#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from nav_msgs.msg import Path


class ActivePathMux:
    def __init__(self):
        self.global_path_topic = rospy.get_param("~global_path_topic", "/astar/path")
        self.local_path_topic = rospy.get_param("~local_path_topic", "/planning/local_path")
        self.avoidance_path_topic = rospy.get_param("~avoidance_path_topic", "/planning/avoidance_path")
        self.active_path_topic = rospy.get_param("~active_path_topic", "/planning/active_path")
        self.local_path_timeout_s = max(0.1, float(rospy.get_param("~local_path_timeout_s", 4.0)))
        self.avoidance_path_timeout_s = max(
            0.1,
            float(rospy.get_param("~avoidance_path_timeout_s", self.local_path_timeout_s)),
        )
        self.use_local_path_fallback = bool(rospy.get_param("~use_local_path_fallback", False))

        self.global_path_msg = None
        self.global_path_stamp = rospy.Time(0)
        self.local_path_msg = None
        self.local_path_stamp = rospy.Time(0)
        self.avoidance_path_msg = None
        self.avoidance_path_stamp = rospy.Time(0)
        self.last_source = "none"

        self.pub_active_path = rospy.Publisher(self.active_path_topic, Path, queue_size=2)
        self.sub_global_path = rospy.Subscriber(self.global_path_topic, Path, self.global_path_callback, queue_size=5)
        self.sub_local_path = rospy.Subscriber(self.local_path_topic, Path, self.local_path_callback, queue_size=5)
        self.sub_avoidance_path = rospy.Subscriber(
            self.avoidance_path_topic,
            Path,
            self.avoidance_path_callback,
            queue_size=5,
        )
        self.timer = rospy.Timer(rospy.Duration(0.1), self.on_timer)

        rospy.loginfo(
            "active_path_mux started | global=%s local=%s avoidance=%s active=%s",
            self.global_path_topic,
            self.local_path_topic,
            self.avoidance_path_topic,
            self.active_path_topic,
        )

    def global_path_callback(self, msg):
        self.global_path_msg = msg
        self.global_path_stamp = rospy.Time.now()

    def local_path_callback(self, msg):
        self.local_path_msg = msg
        self.local_path_stamp = rospy.Time.now()

    def avoidance_path_callback(self, msg):
        self.avoidance_path_msg = msg
        self.avoidance_path_stamp = rospy.Time.now()

    @staticmethod
    def _is_valid_path(msg):
        return msg is not None and len(msg.poses) >= 2

    def _select_active_path(self):
        now = rospy.Time.now()
        use_avoidance = (
            self._is_valid_path(self.avoidance_path_msg)
            and (now - self.avoidance_path_stamp).to_sec() <= self.avoidance_path_timeout_s
        )
        if use_avoidance:
            return "avoidance", self.avoidance_path_msg

        # Keep following the last valid global path until a new one arrives.
        # The global planner is goal-driven, not a continuously streaming source.
        if self._is_valid_path(self.global_path_msg):
            return "global", self.global_path_msg

        use_local = (
            self.use_local_path_fallback
            and
            self._is_valid_path(self.local_path_msg)
            and (now - self.local_path_stamp).to_sec() <= self.local_path_timeout_s
        )
        if use_local:
            return "local", self.local_path_msg

        return "none", None

    def on_timer(self, _event):
        source, msg = self._select_active_path()
        if source == "none":
            if self.last_source != "none":
                out = Path()
                out.header.stamp = rospy.Time.now()
                out.header.frame_id = "map"
                self.pub_active_path.publish(out)
                self.last_source = "none"
            return
        self.pub_active_path.publish(msg)
        if source != self.last_source:
            rospy.loginfo("active_path_mux: source=%s", source)
            self.last_source = source


def main():
    rospy.init_node("active_path_mux", anonymous=False)
    ActivePathMux()
    rospy.spin()


if __name__ == "__main__":
    main()
