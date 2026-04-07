#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rospy
import tf.transformations as transformations
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

from dynamic_window_approach.msg import TrackedObject, TrackedObjectArray


class CloudClusterTracker:
    def __init__(self):
        self.pointcloud_topic = rospy.get_param("~pointcloud_topic", "/ouster/points")
        self.odom_topic = rospy.get_param("~odom_topic", "/lio_localizer/odometry/optimization")
        self.output_topic = rospy.get_param("~output_topic", "/perception/tracked_objects")

        self.min_z = float(rospy.get_param("~min_z", -0.4))
        self.max_z = float(rospy.get_param("~max_z", 1.7))
        self.max_range_m = float(rospy.get_param("~max_range_m", 25.0))
        self.downsample = max(1, int(rospy.get_param("~downsample", 5)))
        self.cell_size_m = max(0.1, float(rospy.get_param("~cell_size_m", 0.40)))
        self.min_points_per_cell = max(1, int(rospy.get_param("~min_points_per_cell", 2)))
        self.min_cluster_cells = max(2, int(rospy.get_param("~min_cluster_cells", 3)))

        self.max_assoc_dist_m = max(0.1, float(rospy.get_param("~max_assoc_dist_m", 1.6)))
        self.track_timeout_s = max(0.1, float(rospy.get_param("~track_timeout_s", 1.0)))
        self.vel_alpha = min(1.0, max(0.01, float(rospy.get_param("~vel_alpha", 0.4))))
        self.publish_static = bool(rospy.get_param("~publish_static", True))
        self.static_speed_thresh_mps = max(0.01, float(rospy.get_param("~static_speed_thresh_mps", 0.15)))
        self.dynamic_min_age = max(1, int(rospy.get_param("~dynamic_min_age", 4)))
        self.pedestrian_static_speed_thresh_mps = max(
            0.01,
            min(
                self.static_speed_thresh_mps,
                float(rospy.get_param("~pedestrian_static_speed_thresh_mps", 0.10)),
            ),
        )
        self.pedestrian_dynamic_min_age = max(
            1,
            min(
                self.dynamic_min_age,
                int(rospy.get_param("~pedestrian_dynamic_min_age", 2)),
            ),
        )
        self.position_jitter_m = max(0.0, float(rospy.get_param("~position_jitter_m", 0.12)))

        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.have_odom = False

        self.next_track_id = 1
        self.tracks = {}

        self.pub_tracks = rospy.Publisher(self.output_topic, TrackedObjectArray, queue_size=2)
        self.sub_odom = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=20)
        self.sub_cloud = rospy.Subscriber(self.pointcloud_topic, PointCloud2, self.cloud_callback, queue_size=1)

        rospy.loginfo(
            "cloud_cluster_tracker started | cloud=%s odom=%s out=%s",
            self.pointcloud_topic,
            self.odom_topic,
            self.output_topic,
        )

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.odom_x = float(p.x)
        self.odom_y = float(p.y)
        self.odom_yaw = transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        self.have_odom = True

    def _local_to_map(self, x, y):
        c = math.cos(self.odom_yaw)
        s = math.sin(self.odom_yaw)
        mx = self.odom_x + c * x - s * y
        my = self.odom_y + s * x + c * y
        return mx, my

    def _extract_clusters(self, msg):
        if not self.have_odom:
            return []

        cells = {}
        rr = self.max_range_m * self.max_range_m
        i = 0
        for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            i += 1
            if self.downsample > 1 and (i % self.downsample != 0):
                continue
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            if z < self.min_z or z > self.max_z:
                continue
            if x * x + y * y > rr:
                continue
            mx, my = self._local_to_map(x, y)
            ix = int(math.floor(mx / self.cell_size_m))
            iy = int(math.floor(my / self.cell_size_m))
            cells[(ix, iy)] = cells.get((ix, iy), 0) + 1

        occ = {k for k, c in cells.items() if c >= self.min_points_per_cell}
        visited = set()
        clusters = []
        for seed in occ:
            if seed in visited:
                continue
            stack = [seed]
            visited.add(seed)
            comp = []
            while stack:
                c = stack.pop()
                comp.append(c)
                cx, cy = c
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nc = (cx + dx, cy + dy)
                        if nc in occ and nc not in visited:
                            visited.add(nc)
                            stack.append(nc)
            if len(comp) < self.min_cluster_cells:
                continue

            xs = []
            ys = []
            min_x = 1e9
            min_y = 1e9
            max_x = -1e9
            max_y = -1e9
            weight_sum = 0
            for (ix, iy) in comp:
                cx = (ix + 0.5) * self.cell_size_m
                cy = (iy + 0.5) * self.cell_size_m
                w = float(cells.get((ix, iy), 1))
                xs.append(cx)
                ys.append(cy)
                min_x = min(min_x, cx)
                min_y = min(min_y, cy)
                max_x = max(max_x, cx)
                max_y = max(max_y, cy)
                weight_sum += w
            if weight_sum <= 0:
                continue
            cx = sum(xs) / float(len(xs))
            cy = sum(ys) / float(len(ys))
            clusters.append(
                {
                    "x": cx,
                    "y": cy,
                    "size_x": max(0.2, max_x - min_x + self.cell_size_m),
                    "size_y": max(0.2, max_y - min_y + self.cell_size_m),
                    "score": min(1.0, len(comp) / 20.0),
                }
            )
        return clusters

    def _associate_and_update(self, clusters, now_sec):
        track_ids = list(self.tracks.keys())
        unmatched_tracks = set(track_ids)
        unmatched_clusters = set(range(len(clusters)))
        matches = []

        for ci, c in enumerate(clusters):
            best_tid = None
            best_d = 1e9
            for tid in track_ids:
                if tid not in unmatched_tracks:
                    continue
                t = self.tracks[tid]
                d = math.hypot(c["x"] - t["x"], c["y"] - t["y"])
                if d < best_d:
                    best_d = d
                    best_tid = tid
            if best_tid is not None and best_d <= self.max_assoc_dist_m:
                matches.append((ci, best_tid))
                unmatched_tracks.discard(best_tid)
                unmatched_clusters.discard(ci)

        for ci, tid in matches:
            c = clusters[ci]
            t = self.tracks[tid]
            dt = max(1e-3, now_sec - t["last_t"])
            dx = c["x"] - t["x"]
            dy = c["y"] - t["y"]
            if math.hypot(dx, dy) < self.position_jitter_m:
                vx_obs = 0.0
                vy_obs = 0.0
            else:
                vx_obs = dx / dt
                vy_obs = dy / dt
            t["vx"] = (1.0 - self.vel_alpha) * t["vx"] + self.vel_alpha * vx_obs
            t["vy"] = (1.0 - self.vel_alpha) * t["vy"] + self.vel_alpha * vy_obs
            t["x"] = c["x"]
            t["y"] = c["y"]
            t["size_x"] = c["size_x"]
            t["size_y"] = c["size_y"]
            t["score"] = c["score"]
            t["last_t"] = now_sec
            t["age"] += 1

        for ci in unmatched_clusters:
            c = clusters[ci]
            tid = self.next_track_id
            self.next_track_id += 1
            self.tracks[tid] = {
                "x": c["x"],
                "y": c["y"],
                "vx": 0.0,
                "vy": 0.0,
                "size_x": c["size_x"],
                "size_y": c["size_y"],
                "score": c["score"],
                "last_t": now_sec,
                "age": 1,
            }

        stale = [tid for tid, t in self.tracks.items() if (now_sec - t["last_t"]) > self.track_timeout_s]
        for tid in stale:
            self.tracks.pop(tid, None)

    @staticmethod
    def _label_track(size_x, size_y, speed):
        area = size_x * size_y
        if area >= 2.5 or max(size_x, size_y) >= 1.8:
            return "vehicle" if speed > 0.2 else "static_vehicle"
        if area >= 0.5:
            return "pedestrian" if speed > 0.15 else "static_person"
        return "unknown"

    @staticmethod
    def _is_person_like_label(label):
        text = (label or "").lower()
        return ("ped" in text) or ("person" in text) or ("walker" in text)

    def cloud_callback(self, msg):
        try:
            now_sec = msg.header.stamp.to_sec() if msg.header.stamp.to_sec() > 0.0 else rospy.Time.now().to_sec()
            clusters = self._extract_clusters(msg)
            self._associate_and_update(clusters, now_sec)
            self.publish_tracks(msg.header.stamp)
        except Exception as e:
            rospy.logwarn_throttle(1.0, "cloud_cluster_tracker error: %s", str(e))

    def publish_tracks(self, stamp):
        out = TrackedObjectArray()
        out.header.stamp = stamp if stamp.to_sec() > 0.0 else rospy.Time.now()
        out.header.frame_id = "map"
        for tid, t in self.tracks.items():
            speed = math.hypot(t["vx"], t["vy"])
            raw_label = self._label_track(t["size_x"], t["size_y"], speed)
            is_person_like = self._is_person_like_label(raw_label)
            dynamic_min_age = (
                self.pedestrian_dynamic_min_age
                if is_person_like
                else self.dynamic_min_age
            )
            static_speed_thresh = (
                self.pedestrian_static_speed_thresh_mps
                if is_person_like
                else self.static_speed_thresh_mps
            )
            effective_vx = float(t["vx"])
            effective_vy = float(t["vy"])
            effective_speed = speed
            if t["age"] < dynamic_min_age and effective_speed < (2.0 * static_speed_thresh):
                effective_vx = 0.0
                effective_vy = 0.0
                effective_speed = 0.0
            if effective_speed < static_speed_thresh:
                effective_vx = 0.0
                effective_vy = 0.0
                effective_speed = 0.0
            if (not self.publish_static) and effective_speed < static_speed_thresh:
                continue
            obj = TrackedObject()
            obj.id = int(tid)
            obj.label = self._label_track(t["size_x"], t["size_y"], effective_speed)
            obj.confidence = float(t["score"])
            obj.pose.position.x = float(t["x"])
            obj.pose.position.y = float(t["y"])
            obj.pose.position.z = 0.0
            obj.pose.orientation.w = 1.0
            obj.twist.linear.x = effective_vx
            obj.twist.linear.y = effective_vy
            obj.twist.linear.z = 0.0
            obj.size.x = float(t["size_x"])
            obj.size.y = float(t["size_y"])
            obj.size.z = 1.5
            out.objects.append(obj)
        self.pub_tracks.publish(out)


def main():
    rospy.init_node("cloud_cluster_tracker", anonymous=False)
    CloudClusterTracker()
    rospy.spin()


if __name__ == "__main__":
    main()
