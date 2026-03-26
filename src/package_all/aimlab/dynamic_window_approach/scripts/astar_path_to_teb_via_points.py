#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rospy
from nav_msgs.msg import Path


class AStarPathToTebViaPoints(object):
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/astar/path")
        self.output_topic = rospy.get_param(
            "~output_topic", "/move_base/TebLocalPlannerROS/via_points"
        )
        self.min_spacing_m = max(
            0.0, float(rospy.get_param("~min_spacing_m", 0.50))
        )
        self.max_points = max(2, int(rospy.get_param("~max_points", 200)))

        self._last_sig = None
        self._last_rx_time = 0.0
        self.pub = rospy.Publisher(self.output_topic, Path, queue_size=1, latch=True)
        self.sub = rospy.Subscriber(
            self.input_topic, Path, self.path_callback, queue_size=2
        )
        self.watchdog = rospy.Timer(rospy.Duration(2.0), self._watchdog_callback)

        rospy.loginfo(
            "astar_path_to_teb_via_points started | in=%s out=%s spacing=%.2fm max_points=%d",
            self.input_topic,
            self.output_topic,
            self.min_spacing_m,
            self.max_points,
        )

    def _watchdog_callback(self, _event):
        if self._last_rx_time > 0.0:
            return
        rospy.logwarn_throttle(
            5.0,
            "astar_path_to_teb_via_points: waiting for input path on %s",
            self.input_topic,
        )

    @staticmethod
    def _dist(a, b):
        dx = float(a.pose.position.x) - float(b.pose.position.x)
        dy = float(a.pose.position.y) - float(b.pose.position.y)
        return math.hypot(dx, dy)

    @staticmethod
    def _path_signature(msg):
        if not msg.poses:
            return ()
        sig = [len(msg.poses)]
        stride = max(1, len(msg.poses) // 16)
        for pose in msg.poses[::stride]:
            sig.append(round(float(pose.pose.position.x), 2))
            sig.append(round(float(pose.pose.position.y), 2))
        last = msg.poses[-1]
        sig.append(round(float(last.pose.position.x), 2))
        sig.append(round(float(last.pose.position.y), 2))
        return tuple(sig)

    def _downsample(self, msg):
        if len(msg.poses) <= 2 or self.min_spacing_m <= 1e-6:
            out = Path()
            out.header = msg.header
            out.poses = list(msg.poses[: self.max_points])
            return out

        out = Path()
        out.header = msg.header
        out.poses.append(msg.poses[0])
        last = msg.poses[0]

        for pose in msg.poses[1:-1]:
            if self._dist(last, pose) < self.min_spacing_m:
                continue
            out.poses.append(pose)
            last = pose
            if len(out.poses) >= (self.max_points - 1):
                break

        if msg.poses[-1] is not out.poses[-1]:
            out.poses.append(msg.poses[-1])

        if len(out.poses) > self.max_points:
            step = max(1, len(out.poses) // max(1, self.max_points - 1))
            reduced = out.poses[::step]
            if reduced[-1] is not out.poses[-1]:
                reduced.append(out.poses[-1])
            out.poses = reduced[: self.max_points]
            if out.poses[-1] is not msg.poses[-1]:
                out.poses[-1] = msg.poses[-1]
        return out

    def path_callback(self, msg):
        self._last_rx_time = rospy.get_time()
        sig = self._path_signature(msg)
        if sig == self._last_sig:
            return
        self._last_sig = sig

        via = self._downsample(msg)
        if not via.poses:
            rospy.logwarn_throttle(
                2.0,
                "astar_path_to_teb_via_points: received empty path on %s",
                self.input_topic,
            )
            return
        self.pub.publish(via)
        rospy.loginfo_throttle(
            1.0,
            "astar_path_to_teb_via_points: published %d via points from %d path poses",
            len(via.poses),
            len(msg.poses),
        )


def main():
    rospy.init_node("astar_path_to_teb_via_points", anonymous=False)
    AStarPathToTebViaPoints()
    rospy.spin()


if __name__ == "__main__":
    main()
