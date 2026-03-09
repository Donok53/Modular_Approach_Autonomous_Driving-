#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
import os
import threading
import xml.etree.ElementTree as ET

import rospy
from nav_msgs.msg import Path
from std_msgs.msg import Empty


class TrajectoryOsmExporter:
    def __init__(self):
        self.path_topic = rospy.get_param("~path_topic", "/lio_sam/mapping/path_pose_wgs84")
        self.output_csv = os.path.expanduser(rospy.get_param("~output_csv", "~/.ros/trajectory_wgs84.csv"))
        self.output_osm = os.path.expanduser(rospy.get_param("~output_osm", "~/.ros/trajectory_wgs84.osm"))
        self.write_period_s = max(0.2, float(rospy.get_param("~write_period_s", 2.0)))
        self.min_points = max(2, int(rospy.get_param("~min_points", 5)))
        self.min_step_m = max(0.0, float(rospy.get_param("~min_step_m", 0.25)))
        self.max_step_m = max(0.0, float(rospy.get_param("~max_step_m", 4.0)))
        self.max_points = max(self.min_points, int(rospy.get_param("~max_points", 50000)))
        # Some upstream path topics occasionally replay the whole prefix repeatedly
        # (e.g. p0,p1,p0,p1,p2,...) which creates radial false edges in OSM.
        self.replay_filter_enable = bool(rospy.get_param("~replay_filter_enable", True))
        self.replay_match_m = max(0.01, float(rospy.get_param("~replay_match_m", 0.5)))
        self.replay_min_resets = max(2, int(rospy.get_param("~replay_min_resets", 8)))
        self.auto_write = bool(rospy.get_param("~auto_write", True))
        self.save_on_shutdown = bool(rospy.get_param("~save_on_shutdown", True))
        self.manual_save_topic = rospy.get_param("~manual_save_topic", "/lio_sam/trajectory_export/save")
        self.publish_reload = bool(rospy.get_param("~publish_reload", False))
        self.reload_topic = rospy.get_param("~reload_topic", "/astar/reload_map")
        self.reload_on_auto_write = bool(rospy.get_param("~reload_on_auto_write", False))
        self.min_new_points_to_write = max(1, int(rospy.get_param("~min_new_points_to_write", 1500)))
        self.min_end_shift_m_to_write = max(0.0, float(rospy.get_param("~min_end_shift_m_to_write", 3.0)))
        self.min_seconds_between_writes = max(0.1, float(rospy.get_param("~min_seconds_between_writes", 10.0)))

        self._lock = threading.RLock()
        self._latest_points = []
        self._dirty = False
        self._last_sig = None
        self._last_written_count = 0
        self._last_written_end = None
        self._last_write_time = 0.0

        self.sub_path = rospy.Subscriber(self.path_topic, Path, self.path_callback, queue_size=2)
        self.sub_save = rospy.Subscriber(self.manual_save_topic, Empty, self.manual_save_callback, queue_size=2)
        self.pub_reload = None
        if self.publish_reload:
            self.pub_reload = rospy.Publisher(self.reload_topic, Empty, queue_size=1, latch=False)

        self.timer = None
        if self.auto_write:
            self.timer = rospy.Timer(rospy.Duration(self.write_period_s), self.on_timer)
        rospy.on_shutdown(self.on_shutdown)

        rospy.loginfo(
            "trajectory_osm_exporter started | topic=%s -> csv=%s, osm=%s",
            self.path_topic,
            self.output_csv,
            self.output_osm,
        )

    @staticmethod
    def _ll_dist_m(lat1, lon1, lat2, lon2):
        # Equirectangular approximation (sufficient for short segment filtering)
        r = 6371000.0
        p1 = math.radians(lat1)
        p2 = math.radians(lat2)
        dp = p2 - p1
        dl = math.radians(lon2 - lon1)
        x = dl * math.cos(0.5 * (p1 + p2))
        y = dp
        return r * math.sqrt(x * x + y * y)

    def _extract_points(self, msg):
        raw = []
        for ps in msg.poses:
            lat = float(ps.pose.position.x)
            lon = float(ps.pose.position.y)
            if not math.isfinite(lat) or not math.isfinite(lon):
                continue
            if lat < -90.0 or lat > 90.0 or lon < -180.0 or lon > 180.0:
                continue
            raw.append((lat, lon))

        if not raw:
            return []

        points = self._collapse_replayed_prefix(raw)

        pts = []
        last = None
        for lat, lon in points:
            if last is not None and self.min_step_m > 0.0:
                step = self._ll_dist_m(last[0], last[1], lat, lon)
                if self.max_step_m > 0.0 and step > self.max_step_m:
                    continue
                if step < self.min_step_m:
                    continue
            pts.append((lat, lon))
            last = (lat, lon)
            if len(pts) >= self.max_points:
                break
        return pts

    def _collapse_replayed_prefix(self, points):
        if (not self.replay_filter_enable) or len(points) < self.min_points:
            return points

        start = points[0]
        cur = []
        segments = []
        reset_count = 0
        for p in points:
            if cur and self._ll_dist_m(start[0], start[1], p[0], p[1]) <= self.replay_match_m:
                if len(cur) >= 2:
                    segments.append(cur)
                    cur = [p]
                    reset_count += 1
                    continue
            cur.append(p)
        if cur:
            segments.append(cur)

        if reset_count < self.replay_min_resets or len(segments) <= 1:
            return points

        best = max(segments, key=len)
        if len(best) < self.min_points:
            return points

        rospy.logwarn_throttle(
            2.0,
            "trajectory_osm_exporter: replayed-prefix pattern detected (raw=%d, resets=%d) -> collapsed to %d points",
            len(points),
            reset_count,
            len(best),
        )
        return best

    @staticmethod
    def _signature(pts):
        if not pts:
            return None
        n = len(pts)
        p0 = pts[0]
        p1 = pts[-1]
        return (n, round(p0[0], 7), round(p0[1], 7), round(p1[0], 7), round(p1[1], 7))

    def path_callback(self, msg):
        pts = self._extract_points(msg)
        if len(pts) < self.min_points:
            return
        sig = self._signature(pts)
        with self._lock:
            if sig == self._last_sig:
                return
            self._latest_points = pts
            self._last_sig = sig
            self._dirty = True

    def _write_csv(self, points):
        out = self.output_csv
        d = os.path.dirname(out)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = out + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["lat", "lon"])
            for lat, lon in points:
                w.writerow([("{:.8f}".format(lat)), ("{:.8f}".format(lon))])
        os.replace(tmp, out)

    def _write_osm(self, points):
        out = self.output_osm
        d = os.path.dirname(out)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = out + ".tmp"

        root = ET.Element("osm", attrib={"version": "0.6", "generator": "trajectory_osm_exporter.py"})
        node_ids = []
        for i, (lat, lon) in enumerate(points):
            nid = str(-(i + 1))
            node_ids.append(nid)
            ET.SubElement(
                root,
                "node",
                attrib={
                    "id": nid,
                    "action": "modify",
                    "visible": "true",
                    "lat": "{:.8f}".format(lat),
                    "lon": "{:.8f}".format(lon),
                },
            )

        way = ET.SubElement(root, "way", attrib={"id": "-1", "action": "modify", "visible": "true"})
        for nid in node_ids:
            ET.SubElement(way, "nd", attrib={"ref": nid})
        ET.SubElement(way, "tag", attrib={"k": "highway", "v": "service"})
        ET.SubElement(way, "tag", attrib={"k": "name", "v": "auto_generated"})

        xml = ET.tostring(root, encoding="utf-8")
        with open(tmp, "wb") as f:
            f.write(b"<?xml version='1.0' encoding='UTF-8'?>\n")
            f.write(xml)
            f.write(b"\n")
        os.replace(tmp, out)

    def write_files(self, reason):
        with self._lock:
            points = list(self._latest_points)
            self._dirty = False

        if len(points) < self.min_points:
            return False
        self._write_csv(points)
        self._write_osm(points)
        now = rospy.Time.now().to_sec()
        with self._lock:
            self._last_written_count = len(points)
            self._last_written_end = points[-1]
            self._last_write_time = now
        rospy.loginfo(
            "trajectory_osm_exporter saved (%s): points=%d, csv=%s, osm=%s",
            reason,
            len(points),
            self.output_csv,
            self.output_osm,
        )
        if self.pub_reload is not None and (reason != "auto" or self.reload_on_auto_write):
            self.pub_reload.publish(Empty())
            rospy.loginfo("trajectory_osm_exporter requested astar reload on %s", self.reload_topic)
        return True

    def manual_save_callback(self, _msg):
        try:
            self.write_files("manual")
        except Exception as e:
            rospy.logwarn("trajectory_osm_exporter manual save failed: %s", str(e))

    def on_timer(self, _event):
        with self._lock:
            dirty = self._dirty
            points = list(self._latest_points)
            last_count = self._last_written_count
            last_end = self._last_written_end
            last_write_time = self._last_write_time
        if not dirty:
            return
        now = rospy.Time.now().to_sec()
        if (now - last_write_time) < self.min_seconds_between_writes:
            return
        if len(points) < self.min_points:
            return

        count_delta = abs(len(points) - last_count)
        end_shift = 0.0
        if last_end is not None:
            end_shift = self._ll_dist_m(last_end[0], last_end[1], points[-1][0], points[-1][1])

        if count_delta < self.min_new_points_to_write and end_shift < self.min_end_shift_m_to_write:
            return
        try:
            self.write_files("auto")
        except Exception as e:
            rospy.logwarn_throttle(1.0, "trajectory_osm_exporter auto save failed: %s", str(e))

    def on_shutdown(self):
        if not self.save_on_shutdown:
            return
        with self._lock:
            has_points = len(self._latest_points) >= self.min_points
        if not has_points:
            return
        try:
            self.write_files("shutdown")
        except Exception as e:
            rospy.logwarn("trajectory_osm_exporter shutdown save failed: %s", str(e))


def main():
    rospy.init_node("trajectory_osm_exporter", anonymous=False)
    TrajectoryOsmExporter()
    rospy.spin()


if __name__ == "__main__":
    main()
