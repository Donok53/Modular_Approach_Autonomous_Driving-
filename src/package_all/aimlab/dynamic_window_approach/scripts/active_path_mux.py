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
        self.active_path_viz_topic = rospy.get_param(
            "~active_path_viz_topic",
            "/planning/active_path_viz",
        )
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
        self.last_viz_source = "none"

        self.pub_active_path = rospy.Publisher(self.active_path_topic, Path, queue_size=2)
        self.pub_active_path_viz = rospy.Publisher(self.active_path_viz_topic, Path, queue_size=2)
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
            "active_path_mux started | global=%s local=%s avoidance=%s active=%s active_viz=%s",
            self.global_path_topic,
            self.local_path_topic,
            self.avoidance_path_topic,
            self.active_path_topic,
            self.active_path_viz_topic,
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

    @staticmethod
    def _is_fresh(stamp, timeout_s):
        return stamp.to_sec() > 0.0 and (rospy.Time.now() - stamp).to_sec() <= timeout_s

    def _has_explicit_local_hold(self):
        return (
            self.local_path_msg is not None
            and len(self.local_path_msg.poses) < 2
            and self._is_fresh(self.local_path_stamp, self.local_path_timeout_s)
        )

    @staticmethod
    def _empty_path_like(msg=None):
        out = Path()
        if msg is not None:
            out.header = msg.header
        else:
            out.header.stamp = rospy.Time.now()
            out.header.frame_id = "map"
        return out

    def _safe_publish(self, publisher, msg):
        if rospy.is_shutdown():
            return False
        try:
            publisher.publish(msg)
            return True
        except rospy.ROSException:
            return False

    def _select_active_path(self):
        use_avoidance = (
            self._is_valid_path(self.avoidance_path_msg)
            and self._is_fresh(self.avoidance_path_stamp, self.avoidance_path_timeout_s)
        )
        if use_avoidance:
            return "avoidance", self.avoidance_path_msg

        if self._has_explicit_local_hold():
            return "hold", self.local_path_msg

        # Keep following the last valid global path until a new one arrives.
        # The global planner is goal-driven, not a continuously streaming source.
        if self._is_valid_path(self.global_path_msg):
            return "global", self.global_path_msg

        use_local = (
            self.use_local_path_fallback
            and
            self._is_valid_path(self.local_path_msg)
            and self._is_fresh(self.local_path_stamp, self.local_path_timeout_s)
        )
        if use_local:
            return "local", self.local_path_msg

        return "none", None

    def _empty_visualization_path(self):
        return self._empty_path_like(
            self.global_path_msg or self.local_path_msg or self.avoidance_path_msg
        )

    def _select_visualization_path(self, source, msg):
        if source in ("avoidance", "local") and self._is_valid_path(msg):
            return source, msg
        return "none", self._empty_visualization_path()

    def _publish_visualization_path(self, source, msg):
        viz_source, viz_msg = self._select_visualization_path(source, msg)
        if viz_source == "none":
            if self.last_viz_source != viz_source:
                if self._safe_publish(self.pub_active_path_viz, viz_msg):
                    rospy.loginfo("active_path_mux: viz_source=%s", viz_source)
                self.last_viz_source = viz_source
            return
        if not self._safe_publish(self.pub_active_path_viz, viz_msg):
            return
        if viz_source != self.last_viz_source:
            rospy.loginfo("active_path_mux: viz_source=%s", viz_source)
            self.last_viz_source = viz_source

    def on_timer(self, _event):
        source, msg = self._select_active_path()
        if source in ("none", "hold"):
            if self.last_source != source:
                if self._safe_publish(self.pub_active_path, self._empty_path_like(msg)):
                    rospy.loginfo("active_path_mux: source=%s", source)
                self.last_source = source
            self._publish_visualization_path(source, msg)
            return
        if not self._safe_publish(self.pub_active_path, msg):
            return
        if source != self.last_source:
            rospy.loginfo("active_path_mux: source=%s", source)
            self.last_source = source
        self._publish_visualization_path(source, msg)


def main():
    rospy.init_node("active_path_mux", anonymous=False)
    ActivePathMux()
    rospy.spin()


if __name__ == "__main__":
    main()
