#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import String


class MapExtensionOdomMux:
    def __init__(self):
        self.primary_odom_topic = rospy.get_param(
            "~primary_odom_topic", "/lio_localizer/odometry/optimization"
        )
        self.primary_twist_topic = rospy.get_param(
            "~primary_twist_topic", "/lio_localizer/odometry/lidar_incremental"
        )
        self.extension_odom_topic = rospy.get_param(
            "~extension_odom_topic", "/map_extension/transformed_odom"
        )
        self.extension_twist_topic = rospy.get_param(
            "~extension_twist_topic", "/map_extension/transformed_odom_incremental"
        )
        self.status_topic = rospy.get_param("~status_topic", "/map_extension/status")
        self.output_odom_topic = rospy.get_param("~output_odom_topic", "/map_extension/active_odom")
        self.output_twist_topic = rospy.get_param(
            "~output_twist_topic", "/map_extension/active_odom_incremental"
        )
        self.extension_timeout_s = max(0.05, float(rospy.get_param("~extension_timeout_s", 0.60)))
        self.primary_timeout_s = max(0.05, float(rospy.get_param("~primary_timeout_s", 1.50)))
        self.publish_hz = max(1.0, float(rospy.get_param("~publish_hz", 30.0)))
        self.log_period_s = max(0.2, float(rospy.get_param("~log_period_s", 1.0)))

        self.primary_odom = None
        self.primary_twist = None
        self.extension_odom = None
        self.extension_twist = None
        self.primary_odom_rx = rospy.Time(0)
        self.primary_twist_rx = rospy.Time(0)
        self.extension_odom_rx = rospy.Time(0)
        self.extension_twist_rx = rospy.Time(0)
        self.extension_state = "idle"
        self.last_source = ""

        self.pub_odom = rospy.Publisher(self.output_odom_topic, Odometry, queue_size=20)
        self.pub_twist = rospy.Publisher(self.output_twist_topic, Odometry, queue_size=20)

        self.sub_primary_odom = rospy.Subscriber(
            self.primary_odom_topic, Odometry, self._primary_odom_cb, queue_size=10, tcp_nodelay=True
        )
        self.sub_primary_twist = rospy.Subscriber(
            self.primary_twist_topic, Odometry, self._primary_twist_cb, queue_size=10, tcp_nodelay=True
        )
        self.sub_extension_odom = rospy.Subscriber(
            self.extension_odom_topic, Odometry, self._extension_odom_cb, queue_size=10, tcp_nodelay=True
        )
        self.sub_extension_twist = rospy.Subscriber(
            self.extension_twist_topic, Odometry, self._extension_twist_cb, queue_size=10, tcp_nodelay=True
        )
        self.sub_status = rospy.Subscriber(self.status_topic, String, self._status_cb, queue_size=5)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_hz), self._timer_cb)

        rospy.loginfo(
            "map_extension_odom_mux started | primary=%s twist=%s extension=%s ext_twist=%s out=%s out_twist=%s",
            self.primary_odom_topic,
            self.primary_twist_topic,
            self.extension_odom_topic,
            self.extension_twist_topic,
            self.output_odom_topic,
            self.output_twist_topic,
        )

    @staticmethod
    def _age(rx_time, now):
        if rx_time == rospy.Time(0):
            return float("inf")
        return max(0.0, (now - rx_time).to_sec())

    @staticmethod
    def _state_from_status(text):
        return str(text).split("|", 1)[0].strip().lower()

    def _primary_odom_cb(self, msg):
        self.primary_odom = msg
        self.primary_odom_rx = rospy.Time.now()

    def _primary_twist_cb(self, msg):
        self.primary_twist = msg
        self.primary_twist_rx = rospy.Time.now()

    def _extension_odom_cb(self, msg):
        self.extension_odom = msg
        self.extension_odom_rx = rospy.Time.now()

    def _extension_twist_cb(self, msg):
        self.extension_twist = msg
        self.extension_twist_rx = rospy.Time.now()

    def _status_cb(self, msg):
        self.extension_state = self._state_from_status(msg.data)

    def _extension_requested(self):
        return self.extension_state in ("running", "saving", "syncing")

    def _select_source(self, now):
        ext_age = self._age(self.extension_odom_rx, now)
        if self._extension_requested() and self.extension_odom is not None and ext_age <= self.extension_timeout_s:
            return "extension", self.extension_odom, ext_age

        primary_age = self._age(self.primary_odom_rx, now)
        if self.primary_odom is not None and primary_age <= self.primary_timeout_s:
            return "primary", self.primary_odom, primary_age

        if self.extension_odom is not None and ext_age <= self.extension_timeout_s:
            return "extension_fallback", self.extension_odom, ext_age

        return "none", None, float("inf")

    def _select_twist(self, source, pose_msg, now):
        if source.startswith("extension"):
            twist_age = self._age(self.extension_twist_rx, now)
            if self.extension_twist is not None and twist_age <= self.extension_timeout_s:
                return self.extension_twist
            return pose_msg

        twist_age = self._age(self.primary_twist_rx, now)
        if self.primary_twist is not None and twist_age <= self.primary_timeout_s:
            return self.primary_twist
        return pose_msg

    def _timer_cb(self, _event):
        now = rospy.Time.now()
        source, pose_msg, age = self._select_source(now)
        if pose_msg is None:
            rospy.logwarn_throttle(
                self.log_period_s,
                "map_extension_odom_mux: waiting for odometry | state=%s primary_age=%.2fs ext_age=%.2fs",
                self.extension_state,
                self._age(self.primary_odom_rx, now),
                self._age(self.extension_odom_rx, now),
            )
            return

        twist_msg = self._select_twist(source, pose_msg, now)
        self.pub_odom.publish(copy.deepcopy(pose_msg))
        self.pub_twist.publish(copy.deepcopy(twist_msg))

        if source != self.last_source:
            rospy.loginfo(
                "map_extension_odom_mux: source=%s state=%s age=%.2fs primary_age=%.2fs ext_age=%.2fs",
                source,
                self.extension_state,
                age,
                self._age(self.primary_odom_rx, now),
                self._age(self.extension_odom_rx, now),
            )
            self.last_source = source


def main():
    rospy.init_node("map_extension_odom_mux", anonymous=False)
    MapExtensionOdomMux()
    rospy.spin()


if __name__ == "__main__":
    main()
