#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2


class TebCmdVelRelay(object):
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/move_base/teb_cmd_vel_raw")
        self.output_topic = rospy.get_param("~output_topic", "/cmd_vel")
        self.publish_hz = max(1.0, float(rospy.get_param("~publish_hz", 20.0)))
        self.idle_timeout_s = max(0.2, float(rospy.get_param("~idle_timeout_s", 1.0)))
        self.log_period_s = max(0.2, float(rospy.get_param("~log_period_s", 1.0)))
        self.forward_only = bool(rospy.get_param("~forward_only", True))
        self.reverse_replacement_speed = max(
            0.0, float(rospy.get_param("~reverse_replacement_speed", 0.05))
        )
        self.enforce_min_linear_speed = bool(
            rospy.get_param("~enforce_min_linear_speed", True)
        )
        self.min_abs_linear_speed = max(
            0.0, float(rospy.get_param("~min_abs_linear_speed", 0.04))
        )
        self.min_angular_for_linear_boost = max(
            0.0, float(rospy.get_param("~min_angular_for_linear_boost", 0.20))
        )
        self.enable_emergency_stop = bool(
            rospy.get_param("~enable_emergency_stop", True)
        )
        self.enable_obstacle_slowdown = bool(
            rospy.get_param("~enable_obstacle_slowdown", True)
        )
        self.robot_width_m = max(0.1, float(rospy.get_param("~robot_width_m", 0.55)))
        self.robot_length_m = max(0.1, float(rospy.get_param("~robot_length_m", 0.60)))
        self.footprint_padding_m = max(0.0, float(rospy.get_param("~footprint_padding_m", 0.0)))
        self.obstacle_cloud_topic = rospy.get_param(
            "~obstacle_cloud_topic", "/move_base/filtered_obstacles"
        )
        self.obstacle_cloud_timeout_s = max(
            0.1, float(rospy.get_param("~obstacle_cloud_timeout_s", 0.6))
        )
        self.emergency_stop_front_margin = max(
            0.0, float(rospy.get_param("~emergency_stop_front_margin", 0.35))
        )
        self.emergency_stop_side_margin = max(
            0.0, float(rospy.get_param("~emergency_stop_side_margin", 0.10))
        )
        self.slowdown_front_margin = max(
            self.emergency_stop_front_margin + 0.05,
            float(rospy.get_param("~slowdown_front_margin", 0.80)),
        )
        self.slowdown_side_margin = max(
            self.emergency_stop_side_margin,
            float(rospy.get_param("~slowdown_side_margin", 0.20)),
        )
        self.robot_half_length = 0.5 * self.robot_length_m + self.footprint_padding_m
        self.robot_half_width = 0.5 * self.robot_width_m + self.footprint_padding_m
        self.emergency_stop_distance = self.robot_half_length + self.emergency_stop_front_margin
        self.emergency_stop_lateral_y = self.robot_half_width + self.emergency_stop_side_margin
        self.slowdown_distance = self.robot_half_length + self.slowdown_front_margin
        self.slowdown_lateral_y = self.robot_half_width + self.slowdown_side_margin

        self.last_cmd = Twist()
        self.last_rx_time = 0.0
        self.last_obstacle_time = 0.0
        self.closest_stop_obstacle_x = float("inf")
        self.closest_slow_obstacle_x = float("inf")

        self.pub = rospy.Publisher(self.output_topic, Twist, queue_size=10)
        self.sub = rospy.Subscriber(self.input_topic, Twist, self.cmd_callback, queue_size=10)
        self.obstacle_sub = None
        if self.enable_emergency_stop or self.enable_obstacle_slowdown:
            self.obstacle_sub = rospy.Subscriber(
                self.obstacle_cloud_topic,
                PointCloud2,
                self.obstacle_callback,
                queue_size=1,
                buff_size=2**24,
            )
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_hz), self.timer_callback)

        rospy.loginfo(
            "teb_cmd_vel_relay started | in=%s out=%s publish=%.1fHz min|v|=%.3f estop=%s slowdown=%s footprint=%.2fx%.2fm stop=%.2fm/%.2fm slow=%.2fm/%.2fm",
            self.input_topic,
            self.output_topic,
            self.publish_hz,
            self.min_abs_linear_speed,
            "on" if self.enable_emergency_stop else "off",
            "on" if self.enable_obstacle_slowdown else "off",
            self.robot_length_m + 2.0 * self.footprint_padding_m,
            self.robot_width_m + 2.0 * self.footprint_padding_m,
            self.emergency_stop_distance,
            self.emergency_stop_lateral_y,
            self.slowdown_distance,
            self.slowdown_lateral_y,
        )

    def _sanitize_cmd(self, cmd):
        out = Twist()
        out.linear.x = float(cmd.linear.x)
        out.linear.y = float(cmd.linear.y)
        out.linear.z = float(cmd.linear.z)
        out.angular.x = float(cmd.angular.x)
        out.angular.y = float(cmd.angular.y)
        out.angular.z = float(cmd.angular.z)
        if self.forward_only and out.linear.x < 0.0:
            rospy.logwarn_throttle(
                self.log_period_s,
                "teb_cmd_vel_relay: clamping reverse cmd v=%.3f -> %.3f",
                out.linear.x,
                self.reverse_replacement_speed,
            )
            out.linear.x = self.reverse_replacement_speed
        if (
            self.enforce_min_linear_speed
            and abs(out.linear.x) > 1e-4
            and abs(out.linear.x) < self.min_abs_linear_speed
            and abs(out.angular.z) >= self.min_angular_for_linear_boost
        ):
            boosted = math.copysign(self.min_abs_linear_speed, out.linear.x)
            rospy.logwarn_throttle(
                self.log_period_s,
                "teb_cmd_vel_relay: boosting tiny cmd v=%.3f -> %.3f (w=%.3f)",
                out.linear.x,
                boosted,
                out.angular.z,
            )
            out.linear.x = boosted
        return out

    @staticmethod
    def _cmd_mag(cmd):
        return math.hypot(float(cmd.linear.x), float(cmd.angular.z))

    def obstacle_callback(self, msg):
        stop_min_x = float("inf")
        slow_min_x = float("inf")

        for x, y, _z in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            if x <= 0.0:
                continue
            abs_y = abs(float(y))
            if abs_y <= self.slowdown_lateral_y:
                slow_min_x = min(slow_min_x, float(x))
            if abs_y <= self.emergency_stop_lateral_y:
                stop_min_x = min(stop_min_x, float(x))

        self.last_obstacle_time = rospy.get_time()
        self.closest_stop_obstacle_x = stop_min_x
        self.closest_slow_obstacle_x = slow_min_x

    def _has_fresh_obstacle_data(self, now):
        return self.last_obstacle_time > 0.0 and (now - self.last_obstacle_time) <= self.obstacle_cloud_timeout_s

    def _apply_obstacle_safety(self, cmd, now):
        if cmd.linear.x <= 0.0 or not self._has_fresh_obstacle_data(now):
            return cmd

        if self.enable_emergency_stop and self.closest_stop_obstacle_x <= self.emergency_stop_distance:
            stopped = Twist()
            stopped.angular.z = cmd.angular.z
            rospy.logwarn_throttle(
                self.log_period_s,
                "teb_cmd_vel_relay: emergency stop | obstacle_x=%.2f m cmd_v=%.3f cmd_w=%.3f",
                self.closest_stop_obstacle_x,
                float(cmd.linear.x),
                float(cmd.angular.z),
            )
            return stopped

        if self.enable_obstacle_slowdown and self.closest_slow_obstacle_x <= self.slowdown_distance:
            span = max(1e-3, self.slowdown_distance - self.emergency_stop_distance)
            scale = (self.closest_slow_obstacle_x - self.emergency_stop_distance) / span
            scale = max(0.0, min(1.0, scale))
            slowed = Twist()
            slowed.linear.x = cmd.linear.x * scale
            slowed.linear.y = cmd.linear.y
            slowed.linear.z = cmd.linear.z
            slowed.angular.x = cmd.angular.x
            slowed.angular.y = cmd.angular.y
            slowed.angular.z = cmd.angular.z
            rospy.logwarn_throttle(
                self.log_period_s,
                "teb_cmd_vel_relay: slowing for obstacle | obstacle_x=%.2f m v=%.3f -> %.3f",
                self.closest_slow_obstacle_x,
                float(cmd.linear.x),
                float(slowed.linear.x),
            )
            return slowed

        return cmd

    def cmd_callback(self, msg):
        self.last_cmd = msg
        self.last_rx_time = rospy.get_time()
        rospy.loginfo_throttle(
            self.log_period_s,
            "teb_cmd_vel_relay: rx cmd | v=%.3f w=%.3f",
            float(msg.linear.x),
            float(msg.angular.z),
        )

    def timer_callback(self, _event):
        now = rospy.get_time()
        if self.last_rx_time <= 0.0 or (now - self.last_rx_time) > self.idle_timeout_s:
            idle = Twist()
            self.pub.publish(idle)
            rospy.logwarn_throttle(
                self.log_period_s,
                "teb_cmd_vel_relay: no fresh cmd from %s for %.2fs",
                self.input_topic,
                self.idle_timeout_s,
            )
            return

        cmd = self._sanitize_cmd(self.last_cmd)
        cmd = self._apply_obstacle_safety(cmd, now)
        self.pub.publish(cmd)
        if self._cmd_mag(cmd) > 1e-3:
            rospy.loginfo_throttle(
                self.log_period_s,
                "teb_cmd_vel_relay: publish cmd | v=%.3f w=%.3f",
                float(cmd.linear.x),
                float(cmd.angular.z),
            )


def main():
    rospy.init_node("teb_cmd_vel_relay", anonymous=False)
    TebCmdVelRelay()
    rospy.spin()


if __name__ == "__main__":
    main()
