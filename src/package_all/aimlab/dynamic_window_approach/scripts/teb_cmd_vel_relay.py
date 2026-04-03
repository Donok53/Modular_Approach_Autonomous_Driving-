#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

from dynamic_window_approach.msg import BehaviorCommand, ExplainabilityEvent


class TebCmdVelRelay(object):
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/move_base/teb_cmd_vel_raw")
        self.output_topic = rospy.get_param("~output_topic", "/cmd_vel")
        self.behavior_cmd_topic = str(
            rospy.get_param("~behavior_cmd_topic", "/planning/behavior_cmd")
        ).strip()
        self.debug_text_topic = str(
            rospy.get_param("~debug_text_topic", "/planning/teb_debug_text")
        ).strip()
        self.debug_text_period_s = max(
            0.1, float(rospy.get_param("~debug_text_period_s", 0.5))
        )
        self.debug_screen_logging = bool(
            rospy.get_param("~debug_screen_logging", True)
        )
        self.debug_screen_log_period_s = max(
            0.2, float(rospy.get_param("~debug_screen_log_period_s", 1.0))
        )
        self.explainability_topic = str(
            rospy.get_param("~explainability_topic", "/planning/explainability")
        ).strip()
        self.odom_topic = str(
            rospy.get_param("~odom_topic", "/lio_localizer/odometry/optimization")
        ).strip()
        self.local_path_topic = str(
            rospy.get_param("~local_path_topic", "/planning/local_path")
        ).strip()
        self.avoidance_path_topic = str(
            rospy.get_param("~avoidance_path_topic", "/planning/avoidance_path")
        ).strip()
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
        self.reverse_deadband_mps = max(
            0.0, float(rospy.get_param("~reverse_deadband_mps", 0.03))
        )
        self.enable_cmd_smoothing = bool(
            rospy.get_param("~enable_cmd_smoothing", True)
        )
        self.max_linear_slew_mps2 = max(
            0.0, float(rospy.get_param("~max_linear_slew_mps2", 0.70))
        )
        self.max_angular_slew_rps2 = max(
            0.0, float(rospy.get_param("~max_angular_slew_rps2", 1.40))
        )
        self.final_path_pose_threshold = max(
            1, int(rospy.get_param("~final_path_pose_threshold", 2))
        )
        self.final_cmd_linear_stop_threshold = max(
            0.0, float(rospy.get_param("~final_cmd_linear_stop_threshold", 0.15))
        )
        self.final_cmd_angular_stop_threshold = max(
            0.0, float(rospy.get_param("~final_cmd_angular_stop_threshold", 0.35))
        )
        self.final_brake_distance_m = max(
            0.0, float(rospy.get_param("~final_brake_distance_m", 0.18))
        )
        self.ignore_local_hold_near_goal_distance_m = max(
            0.0, float(rospy.get_param("~ignore_local_hold_near_goal_distance_m", 0.45))
        )
        self.enable_emergency_stop = bool(
            rospy.get_param("~enable_emergency_stop", True)
        )
        self.enable_obstacle_slowdown = bool(
            rospy.get_param("~enable_obstacle_slowdown", True)
        )
        self.enable_local_path_hold_stop = bool(
            rospy.get_param("~enable_local_path_hold_stop", True)
        )
        self.hold_requires_avoidance_path = bool(
            rospy.get_param("~hold_requires_avoidance_path", True)
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
        self.behavior_cmd_timeout_s = max(
            0.1, float(rospy.get_param("~behavior_cmd_timeout_s", 0.8))
        )
        self.local_hold_timeout_s = max(
            0.1, float(rospy.get_param("~local_hold_timeout_s", 1.0))
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
        self.allow_in_place_rotation_near_obstacle = bool(
            rospy.get_param("~allow_in_place_rotation_near_obstacle", True)
        )
        self.allow_in_place_rotation_during_local_hold = bool(
            rospy.get_param("~allow_in_place_rotation_during_local_hold", True)
        )
        self.min_in_place_rotation_angular_speed = max(
            0.0, float(rospy.get_param("~min_in_place_rotation_angular_speed", 0.05))
        )
        self.ignore_obstacles_beyond_local_goal = bool(
            rospy.get_param("~ignore_obstacles_beyond_local_goal", True)
        )
        self.near_goal_obstacle_ignore_distance_m = max(
            0.0, float(rospy.get_param("~near_goal_obstacle_ignore_distance_m", 1.2))
        )
        self.obstacle_beyond_goal_slack_m = max(
            0.0, float(rospy.get_param("~obstacle_beyond_goal_slack_m", 0.05))
        )
        self.robot_half_length = 0.5 * self.robot_length_m + self.footprint_padding_m
        self.robot_half_width = 0.5 * self.robot_width_m + self.footprint_padding_m
        self.emergency_stop_distance = self.robot_half_length + self.emergency_stop_front_margin
        self.emergency_stop_lateral_y = self.robot_half_width + self.emergency_stop_side_margin
        self.slowdown_distance = self.robot_half_length + self.slowdown_front_margin
        self.slowdown_lateral_y = self.robot_half_width + self.slowdown_side_margin

        self.last_cmd = Twist()
        self.last_rx_time = 0.0
        self.last_publish_cmd = Twist()
        self.last_publish_time = 0.0
        self.last_obstacle_time = 0.0
        self.last_behavior_time = 0.0
        self.behavior_stop = False
        self.behavior_speed_limit = float("inf")
        self.behavior_reason = "clear"
        self.last_local_path_time = 0.0
        self.local_path_empty = False
        self.local_path_pose_count = 0
        self.local_path_remaining_m = float("inf")
        self.last_nonempty_local_path_remaining_m = float("inf")
        self.last_avoidance_path_time = 0.0
        self.avoidance_path_active = False
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.have_odom = False
        self.closest_stop_obstacle_x = float("inf")
        self.closest_stop_obstacle_y = 0.0
        self.closest_slow_obstacle_x = float("inf")
        self.closest_slow_obstacle_y = 0.0
        self._last_explain_key = None
        self._last_explain_time = 0.0
        self._last_debug_text = ""
        self._last_debug_text_time = 0.0
        self._last_debug_screen_time = 0.0

        self.pub = rospy.Publisher(self.output_topic, Twist, queue_size=10)
        self.pub_debug_text = None
        if self.debug_text_topic:
            self.pub_debug_text = rospy.Publisher(
                self.debug_text_topic, String, queue_size=20
            )
        self.pub_explainability = rospy.Publisher(
            self.explainability_topic, ExplainabilityEvent, queue_size=20
        )
        self.sub = rospy.Subscriber(self.input_topic, Twist, self.cmd_callback, queue_size=10)
        self.behavior_sub = None
        if self.behavior_cmd_topic:
            self.behavior_sub = rospy.Subscriber(
                self.behavior_cmd_topic, BehaviorCommand, self.behavior_callback, queue_size=10
            )
        self.odom_sub = None
        if self.odom_topic:
            self.odom_sub = rospy.Subscriber(
                self.odom_topic, Odometry, self.odom_callback, queue_size=10
            )
        self.local_path_sub = None
        if self.local_path_topic:
            self.local_path_sub = rospy.Subscriber(
                self.local_path_topic, Path, self.local_path_callback, queue_size=5
            )
        self.avoidance_path_sub = None
        if self.avoidance_path_topic:
            self.avoidance_path_sub = rospy.Subscriber(
                self.avoidance_path_topic, Path, self.avoidance_path_callback, queue_size=5
            )
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
            "teb_cmd_vel_relay started | in=%s out=%s debug=%s explain=%s behavior=%s odom=%s local=%s avoidance=%s publish=%.1fHz min|v|=%.3f estop=%s slowdown=%s hold_stop=%s hold_requires_avoid=%s smoothing=%s slew(v=%.2f,w=%.2f) footprint=%.2fx%.2fm stop=%.2fm/%.2fm slow=%.2fm/%.2fm final_brake=%.2fm hold_ignore=%.2fm rotate_near_obs=%s hold_rotate=%s",
            self.input_topic,
            self.output_topic,
            self.debug_text_topic if self.debug_text_topic else "-",
            self.explainability_topic if self.explainability_topic else "-",
            self.behavior_cmd_topic if self.behavior_cmd_topic else "-",
            self.odom_topic if self.odom_topic else "-",
            self.local_path_topic if self.local_path_topic else "-",
            self.avoidance_path_topic if self.avoidance_path_topic else "-",
            self.publish_hz,
            self.min_abs_linear_speed,
            "on" if self.enable_emergency_stop else "off",
            "on" if self.enable_obstacle_slowdown else "off",
            "on" if self.enable_local_path_hold_stop else "off",
            "on" if self.hold_requires_avoidance_path else "off",
            "on" if self.enable_cmd_smoothing else "off",
            self.max_linear_slew_mps2,
            self.max_angular_slew_rps2,
            self.robot_length_m + 2.0 * self.footprint_padding_m,
            self.robot_width_m + 2.0 * self.footprint_padding_m,
            self.emergency_stop_distance,
            self.emergency_stop_lateral_y,
            self.slowdown_distance,
            self.slowdown_lateral_y,
            self.final_brake_distance_m,
            self.ignore_local_hold_near_goal_distance_m,
            "on" if self.allow_in_place_rotation_near_obstacle else "off",
            "on" if self.allow_in_place_rotation_during_local_hold else "off",
        )

    @staticmethod
    def _fmt_debug_float(value, precision=2):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "nan"
        if not math.isfinite(v):
            return "inf"
        return ("{:.%df}" % int(precision)).format(v)

    def _publish_debug_text(self, text, now=None, force=False):
        if rospy.is_shutdown():
            return
        stamp = rospy.get_time() if now is None else float(now)
        if (
            (not force)
            and text == self._last_debug_text
            and (stamp - self._last_debug_text_time) < self.debug_text_period_s
        ):
            return
        self._last_debug_text = text
        self._last_debug_text_time = stamp
        if self.pub_debug_text is not None:
            try:
                self.pub_debug_text.publish(String(data=text))
            except rospy.ROSException:
                pass
        if self.debug_screen_logging and (
            force or (stamp - self._last_debug_screen_time) >= self.debug_screen_log_period_s
        ):
            self._last_debug_screen_time = stamp
            rospy.loginfo("teb_cmd_vel_relay debug | %s", text)

    def _build_debug_text(self, raw_cmd, sanitized_cmd, final_cmd, explain, now):
        trigger_reason = "clear"
        action_taken = "follow_teb"
        if explain is not None:
            trigger_reason = str(explain.get("trigger_reason", "clear") or "clear")
            action_taken = str(explain.get("action_taken", "follow_teb") or "follow_teb")

        behavior_limit = (
            self._fmt_debug_float(self.behavior_speed_limit)
            if self.behavior_speed_limit < float("inf")
            else "inf"
        )
        return (
            "relay reason={} action={} raw(v={},w={}) sanitized(v={},w={}) out(v={},w={}) "
            "local(n={},empty={},remain={}) avoid={} hold={} final_brake={} "
            "behavior(stop={},limit={}) obs(stop_x={},slow_x={})"
        ).format(
            trigger_reason,
            action_taken,
            self._fmt_debug_float(raw_cmd.linear.x, 3),
            self._fmt_debug_float(raw_cmd.angular.z, 3),
            self._fmt_debug_float(sanitized_cmd.linear.x, 3),
            self._fmt_debug_float(sanitized_cmd.angular.z, 3),
            self._fmt_debug_float(final_cmd.linear.x, 3),
            self._fmt_debug_float(final_cmd.angular.z, 3),
            int(self.local_path_pose_count),
            "yes" if self.local_path_empty else "no",
            self._fmt_debug_float(self.local_path_remaining_m),
            "on" if self.avoidance_path_active else "off",
            "on" if self._has_local_hold(now) else "off",
            "on" if self._is_final_goal_brake_active() else "off",
            "yes" if self.behavior_stop else "no",
            behavior_limit,
            self._fmt_debug_float(self.closest_stop_obstacle_x),
            self._fmt_debug_float(self.closest_slow_obstacle_x),
        )

    def _safe_publish_cmd(self, cmd):
        if rospy.is_shutdown():
            return
        try:
            self.pub.publish(cmd)
        except rospy.ROSException:
            pass

    @staticmethod
    def _clamp_delta(current, target, limit):
        if limit <= 0.0:
            return float(target)
        delta = float(target) - float(current)
        if delta > limit:
            delta = limit
        elif delta < -limit:
            delta = -limit
        return float(current) + delta

    def _apply_cmd_smoothing(self, cmd, now, bypass=False):
        if bypass or (not self.enable_cmd_smoothing):
            return cmd
        if self.last_publish_time <= 0.0:
            return cmd

        dt = max(1e-3, float(now) - float(self.last_publish_time))
        linear_limit = self.max_linear_slew_mps2 * dt
        angular_limit = self.max_angular_slew_rps2 * dt

        out = Twist()
        out.linear.x = self._clamp_delta(
            self.last_publish_cmd.linear.x, cmd.linear.x, linear_limit
        )
        out.linear.y = float(cmd.linear.y)
        out.linear.z = float(cmd.linear.z)
        out.angular.x = float(cmd.angular.x)
        out.angular.y = float(cmd.angular.y)
        out.angular.z = self._clamp_delta(
            self.last_publish_cmd.angular.z, cmd.angular.z, angular_limit
        )

        if (
            abs(out.linear.x - float(cmd.linear.x)) > 1e-4
            or abs(out.angular.z - float(cmd.angular.z)) > 1e-4
        ):
            rospy.loginfo_throttle(
                self.log_period_s,
                "teb_cmd_vel_relay: smoothing cmd | target(v=%.3f w=%.3f) -> out(v=%.3f w=%.3f)",
                float(cmd.linear.x),
                float(cmd.angular.z),
                float(out.linear.x),
                float(out.angular.z),
            )
        return out

    def _sanitize_cmd(self, cmd):
        out = Twist()
        out.linear.x = float(cmd.linear.x)
        out.linear.y = float(cmd.linear.y)
        out.linear.z = float(cmd.linear.z)
        out.angular.x = float(cmd.angular.x)
        out.angular.y = float(cmd.angular.y)
        out.angular.z = float(cmd.angular.z)
        final_brake_active = self._is_final_goal_brake_active()
        reverse_clamped_to_stop = False
        if out.linear.x < 0.0 and abs(out.linear.x) <= self.reverse_deadband_mps:
            if final_brake_active:
                rospy.loginfo_throttle(
                    self.log_period_s,
                    "teb_cmd_vel_relay: braking at final goal v=%.3f w=%.3f",
                    out.linear.x,
                    out.angular.z,
                )
                return Twist()
            rospy.loginfo_throttle(
                self.log_period_s,
                "teb_cmd_vel_relay: zeroing tiny reverse cmd v=%.3f (w=%.3f)",
                out.linear.x,
                out.angular.z,
            )
            if self.forward_only and self.reverse_replacement_speed > 1e-4:
                out.linear.x = self.reverse_replacement_speed
                reverse_clamped_to_stop = False
                rospy.loginfo_throttle(
                    self.log_period_s,
                    "teb_cmd_vel_relay: replacing tiny reverse cmd with forward crawl v=%.3f (w=%.3f)",
                    out.linear.x,
                    out.angular.z,
                )
            else:
                out.linear.x = 0.0
                reverse_clamped_to_stop = True
        if self.forward_only and out.linear.x < 0.0:
            rospy.logwarn_throttle(
                self.log_period_s,
                "teb_cmd_vel_relay: clamping reverse cmd v=%.3f -> %.3f",
                out.linear.x,
                self.reverse_replacement_speed,
            )
            out.linear.x = self.reverse_replacement_speed
            reverse_clamped_to_stop = out.linear.x <= 1e-4
        if reverse_clamped_to_stop and abs(out.angular.z) > 1e-4:
            if self._should_preserve_in_place_rotation(out):
                rospy.loginfo_throttle(
                    self.log_period_s,
                    "teb_cmd_vel_relay: converting reverse-clamped cmd to in-place rotation w=%.3f",
                    out.angular.z,
                )
                return self._rotation_only_cmd(out)
            rospy.loginfo_throttle(
                self.log_period_s,
                "teb_cmd_vel_relay: suppressing spin for reverse-clamped cmd w=%.3f",
                out.angular.z,
            )
            out.angular.z = 0.0
            out.angular.x = 0.0
            out.angular.y = 0.0
        if (
            self.enforce_min_linear_speed
            and out.linear.x > 1e-4
            and out.linear.x < self.min_abs_linear_speed
            and abs(out.angular.z) >= self.min_angular_for_linear_boost
            and not final_brake_active
        ):
            boosted = self.min_abs_linear_speed
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

    @staticmethod
    def _pose_dist_xy(x, y, pose_stamped):
        dx = float(pose_stamped.pose.position.x) - float(x)
        dy = float(pose_stamped.pose.position.y) - float(y)
        return math.hypot(dx, dy)

    def _should_preserve_in_place_rotation(self, cmd):
        return abs(float(cmd.angular.z)) >= self.min_in_place_rotation_angular_speed

    @staticmethod
    def _rotation_only_cmd(cmd):
        out = Twist()
        out.angular.z = float(cmd.angular.z)
        return out

    def obstacle_callback(self, msg):
        stop_min_x = float("inf")
        stop_min_y = 0.0
        slow_min_x = float("inf")
        slow_min_y = 0.0

        for x, y, _z in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            if x <= 0.0:
                continue
            abs_y = abs(float(y))
            if abs_y <= self.slowdown_lateral_y:
                if float(x) < slow_min_x:
                    slow_min_x = float(x)
                    slow_min_y = float(y)
            if abs_y <= self.emergency_stop_lateral_y:
                if float(x) < stop_min_x:
                    stop_min_x = float(x)
                    stop_min_y = float(y)

        self.last_obstacle_time = rospy.get_time()
        self.closest_stop_obstacle_x = stop_min_x
        self.closest_stop_obstacle_y = stop_min_y
        self.closest_slow_obstacle_x = slow_min_x
        self.closest_slow_obstacle_y = slow_min_y

    def odom_callback(self, msg):
        self.odom_x = float(msg.pose.pose.position.x)
        self.odom_y = float(msg.pose.pose.position.y)
        self.have_odom = True

    def behavior_callback(self, msg):
        self.last_behavior_time = rospy.get_time()
        self.behavior_stop = bool(msg.stop)
        self.behavior_speed_limit = max(0.0, float(msg.speed_limit))
        self.behavior_reason = str(msg.reason).strip() if str(msg.reason).strip() else "behavior"

    def local_path_callback(self, msg):
        self.last_local_path_time = rospy.get_time()
        self.local_path_pose_count = len(msg.poses)
        self.local_path_empty = len(msg.poses) < 2
        self.local_path_remaining_m = float("inf")
        if len(msg.poses) < 2 or (not self.have_odom):
            return

        nearest_idx = 0
        best_dist = float("inf")
        for idx, pose in enumerate(msg.poses):
            dist = self._pose_dist_xy(self.odom_x, self.odom_y, pose)
            if dist < best_dist:
                best_dist = dist
                nearest_idx = idx

        remain_m = 0.0
        for idx in range(nearest_idx, len(msg.poses) - 1):
            p0 = msg.poses[idx]
            p1 = msg.poses[idx + 1]
            remain_m += self._pose_dist_xy(
                float(p0.pose.position.x),
                float(p0.pose.position.y),
                p1,
            )
        self.local_path_remaining_m = remain_m
        self.last_nonempty_local_path_remaining_m = remain_m

    def avoidance_path_callback(self, msg):
        self.last_avoidance_path_time = rospy.get_time()
        self.avoidance_path_active = len(msg.poses) >= 2

    def _is_final_path_segment_active(self):
        return (
            self.local_path_pose_count > 0
            and self.local_path_pose_count <= self.final_path_pose_threshold
            and not self.avoidance_path_active
        )

    def _is_final_goal_brake_active(self):
        return (
            self._is_final_path_segment_active()
            and math.isfinite(self.local_path_remaining_m)
            and self.local_path_remaining_m <= self.final_brake_distance_m
        )

    def _has_fresh_obstacle_data(self, now):
        return self.last_obstacle_time > 0.0 and (now - self.last_obstacle_time) <= self.obstacle_cloud_timeout_s

    def _has_fresh_behavior(self, now):
        return self.last_behavior_time > 0.0 and (now - self.last_behavior_time) <= self.behavior_cmd_timeout_s

    def _has_local_hold(self, now):
        if not self.enable_local_path_hold_stop:
            return False
        if self.last_local_path_time <= 0.0:
            return False
        if (now - self.last_local_path_time) > self.local_hold_timeout_s:
            return False
        if not self.local_path_empty:
            return False
        if self.hold_requires_avoidance_path and (not self.avoidance_path_active):
            return False
        if (
            (not self.avoidance_path_active)
            and math.isfinite(self.last_nonempty_local_path_remaining_m)
            and self.last_nonempty_local_path_remaining_m <= self.ignore_local_hold_near_goal_distance_m
        ):
            return False
        if self.last_avoidance_path_time > 0.0 and (now - self.last_avoidance_path_time) <= self.local_hold_timeout_s:
            return not self.avoidance_path_active
        return True

    def _effective_local_path_remaining_m(self, now=None):
        if math.isfinite(self.local_path_remaining_m):
            return self.local_path_remaining_m
        if not self.local_path_empty:
            return float("inf")
        if self.last_local_path_time <= 0.0:
            return float("inf")
        if now is None:
            now = rospy.get_time()
        if (now - self.last_local_path_time) > self.local_hold_timeout_s:
            return float("inf")
        if not math.isfinite(self.last_nonempty_local_path_remaining_m):
            return float("inf")
        return self.last_nonempty_local_path_remaining_m

    def _should_ignore_obstacle_for_local_goal(self, obstacle_x, now=None):
        if (not self.ignore_obstacles_beyond_local_goal) or (not self.have_odom):
            return False
        if not math.isfinite(obstacle_x):
            return False
        remain_m = self._effective_local_path_remaining_m(now=now)
        if not math.isfinite(remain_m):
            return False
        if remain_m <= 1e-3:
            return False
        if remain_m > self.near_goal_obstacle_ignore_distance_m:
            return False
        return obstacle_x > (remain_m + self.obstacle_beyond_goal_slack_m)

    def _publish_explainability(
        self,
        event_type,
        stamp=None,
        trigger_reason="",
        action_taken="",
        local_planning_active=False,
        stop_commanded=False,
        slowdown_commanded=False,
        speed_before_mps=-1.0,
        speed_after_mps=-1.0,
        speed_limit_mps=-1.0,
        closest_obstacle_dist_m=-1.0,
        obstacle_lateral_offset_m=-1.0,
        summary_text="",
    ):
        msg = ExplainabilityEvent()
        msg.header.stamp = stamp if stamp is not None else rospy.Time.now()
        msg.source_node = "teb_cmd_vel_relay"
        msg.event_type = str(event_type)
        msg.decision_layer = "control_safety_layer"
        msg.trigger_reason = str(trigger_reason)
        msg.action_taken = str(action_taken)
        msg.avoid_direction = "none"
        msg.local_planning_active = bool(local_planning_active)
        msg.stop_commanded = bool(stop_commanded)
        msg.slowdown_commanded = bool(slowdown_commanded)
        msg.speed_before_mps = float(speed_before_mps)
        msg.speed_after_mps = float(speed_after_mps)
        msg.speed_limit_mps = float(speed_limit_mps)
        msg.closest_obstacle_dist_m = float(closest_obstacle_dist_m)
        msg.obstacle_lateral_offset_m = float(obstacle_lateral_offset_m)
        msg.ttc_s = -1.0
        msg.tracked_object_id = -1
        msg.tracked_object_label = ""
        msg.summary_text = str(summary_text)

        key = (
            msg.event_type,
            msg.trigger_reason,
            msg.action_taken,
            msg.stop_commanded,
            msg.slowdown_commanded,
            round(float(msg.speed_limit_mps), 2),
        )
        if key == self._last_explain_key:
            return
        stamp_sec = msg.header.stamp.to_sec() if msg.header.stamp.to_sec() > 0.0 else rospy.get_time()
        self._last_explain_key = key
        self._last_explain_time = stamp_sec
        if rospy.is_shutdown():
            return
        try:
            self.pub_explainability.publish(msg)
        except rospy.ROSException:
            pass

    def _apply_behavior_safety(self, cmd, now):
        if not self._has_fresh_behavior(now):
            return cmd, None

        if self.behavior_stop:
            rospy.logwarn_throttle(
                self.log_period_s,
                "teb_cmd_vel_relay: behavior stop | reason=%s",
                self.behavior_reason,
            )
            return Twist(), {
                "event_type": "CONTROL_ACTION_CHANGE",
                "trigger_reason": self.behavior_reason,
                "action_taken": "stop",
                "stop_commanded": True,
                "slowdown_commanded": False,
                "speed_limit_mps": 0.0,
                "summary_text": "TEB relay applied a full stop because the behavior layer requested a stop.",
            }

        if self.behavior_speed_limit < float("inf") and abs(cmd.linear.x) > self.behavior_speed_limit:
            limited = Twist()
            limited.linear.x = math.copysign(self.behavior_speed_limit, cmd.linear.x)
            limited.linear.y = cmd.linear.y
            limited.linear.z = cmd.linear.z
            limited.angular.x = cmd.angular.x
            limited.angular.y = cmd.angular.y
            limited.angular.z = cmd.angular.z
            rospy.logwarn_throttle(
                self.log_period_s,
                "teb_cmd_vel_relay: behavior speed limit | reason=%s v=%.3f -> %.3f",
                self.behavior_reason,
                float(cmd.linear.x),
                float(limited.linear.x),
            )
            return limited, {
                "event_type": "CONTROL_ACTION_CHANGE",
                "trigger_reason": self.behavior_reason,
                "action_taken": "slowdown",
                "stop_commanded": False,
                "slowdown_commanded": True,
                "speed_limit_mps": float(self.behavior_speed_limit),
                "summary_text": "TEB relay reduced the speed because the behavior layer requested a speed limit.",
            }
        return cmd, None

    def _apply_local_hold(self, cmd, now):
        if not self._has_local_hold(now):
            return cmd, None
        if (
            self.allow_in_place_rotation_during_local_hold
            and self.avoidance_path_active
            and self._should_preserve_in_place_rotation(cmd)
        ):
            rospy.loginfo_throttle(
                self.log_period_s,
                "teb_cmd_vel_relay: local hold keeps in-place rotation | w=%.3f avoidance=active",
                float(cmd.angular.z),
            )
            return self._rotation_only_cmd(cmd), {
                "event_type": "CONTROL_ACTION_CHANGE",
                "trigger_reason": "local_replanner_hold",
                "action_taken": "hold_rotate",
                "stop_commanded": False,
                "slowdown_commanded": True,
                "speed_limit_mps": 0.0,
                "local_planning_active": True,
                "summary_text": "TEB relay held forward motion for local replanning but kept an in-place rotation command.",
            }
        rospy.logwarn_throttle(
            self.log_period_s,
            "teb_cmd_vel_relay: holding stop for local replanner | local_empty=yes avoidance=%s",
            "active" if self.avoidance_path_active else "none",
        )
        return Twist(), {
            "event_type": "CONTROL_ACTION_CHANGE",
            "trigger_reason": "local_replanner_hold",
            "action_taken": "hold_stop",
            "stop_commanded": True,
            "slowdown_commanded": False,
            "speed_limit_mps": 0.0,
            "local_planning_active": True,
            "summary_text": "TEB relay held the robot stopped because local replanning is waiting before avoidance.",
        }

    def _apply_obstacle_safety(self, cmd, now):
        if cmd.linear.x <= 0.0 or not self._has_fresh_obstacle_data(now):
            return cmd, None

        final_segment_active = self._is_final_path_segment_active()
        remain_m = self._effective_local_path_remaining_m(now=now)
        ignore_stop = self._should_ignore_obstacle_for_local_goal(
            self.closest_stop_obstacle_x, now=now
        )
        ignore_slow = self._should_ignore_obstacle_for_local_goal(
            self.closest_slow_obstacle_x, now=now
        )
        if ignore_stop or ignore_slow:
            obstacle_x = self.closest_stop_obstacle_x
            if (not math.isfinite(obstacle_x)) or (
                math.isfinite(self.closest_slow_obstacle_x)
                and self.closest_slow_obstacle_x < obstacle_x
            ):
                obstacle_x = self.closest_slow_obstacle_x
            rospy.loginfo_throttle(
                self.log_period_s,
                "teb_cmd_vel_relay: ignoring obstacle beyond local goal obstacle_x=%.2f remain=%.2f",
                obstacle_x,
                remain_m,
            )
        if (
            self.enable_emergency_stop
            and (not ignore_stop)
            and self.closest_stop_obstacle_x <= self.emergency_stop_distance
        ):
            stopped = Twist()
            rospy.logwarn_throttle(
                self.log_period_s,
                "teb_cmd_vel_relay: emergency stop | obstacle_x=%.2f m cmd_v=%.3f cmd_w=%.3f",
                self.closest_stop_obstacle_x,
                float(cmd.linear.x),
                float(cmd.angular.z),
            )
            return stopped, {
                "event_type": "CONTROL_ACTION_CHANGE",
                "trigger_reason": "front_obstacle_emergency",
                "action_taken": "emergency_stop",
                "stop_commanded": True,
                "slowdown_commanded": False,
                "speed_limit_mps": 0.0,
                "closest_obstacle_dist_m": float(self.closest_stop_obstacle_x),
                "obstacle_lateral_offset_m": float(self.closest_stop_obstacle_y),
                "summary_text": "TEB relay applied an emergency stop because a close obstacle was detected ahead.",
            }

        if (
            self.enable_obstacle_slowdown
            and (not ignore_slow)
            and self.closest_slow_obstacle_x <= self.slowdown_distance
        ):
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
            if abs(slowed.linear.x) <= self.final_cmd_linear_stop_threshold:
                if (
                    self.allow_in_place_rotation_near_obstacle
                    and (not final_segment_active)
                    and self._should_preserve_in_place_rotation(cmd)
                ):
                    rospy.loginfo_throttle(
                        self.log_period_s,
                        "teb_cmd_vel_relay: converting obstacle slowdown to in-place rotation v=%.3f w=%.3f obstacle_x=%.2f",
                        slowed.linear.x,
                        cmd.angular.z,
                        self.closest_slow_obstacle_x,
                    )
                    slowed = self._rotation_only_cmd(cmd)
                else:
                    if abs(slowed.angular.z) > 1e-4:
                        rospy.loginfo_throttle(
                            self.log_period_s,
                            "teb_cmd_vel_relay: suppressing obstacle-stop spin v=%.3f w=%.3f final=%s",
                            slowed.linear.x,
                            slowed.angular.z,
                            "yes" if final_segment_active else "no",
                        )
                    slowed.angular.z = 0.0
                    slowed.angular.x = 0.0
                    slowed.angular.y = 0.0
            rospy.logwarn_throttle(
                self.log_period_s,
                "teb_cmd_vel_relay: slowing for obstacle | obstacle_x=%.2f m v=%.3f -> %.3f",
                self.closest_slow_obstacle_x,
                float(cmd.linear.x),
                float(slowed.linear.x),
            )
            return slowed, {
                "event_type": "CONTROL_ACTION_CHANGE",
                "trigger_reason": "front_obstacle_slowdown",
                "action_taken": "slowdown",
                "stop_commanded": False,
                "slowdown_commanded": True,
                "speed_limit_mps": float(slowed.linear.x),
                "closest_obstacle_dist_m": float(self.closest_slow_obstacle_x),
                "obstacle_lateral_offset_m": float(self.closest_slow_obstacle_y),
                "summary_text": "TEB relay slowed the robot because an obstacle was detected in the forward slowdown zone.",
            }

        return cmd, None

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
            self._safe_publish_cmd(idle)
            self.last_publish_cmd = idle
            self.last_publish_time = now
            self._publish_explainability(
                event_type="CONTROL_ACTION_CHANGE",
                trigger_reason="stale_cmd_timeout",
                action_taken="idle_stop",
                stop_commanded=True,
                speed_before_mps=0.0,
                speed_after_mps=0.0,
                speed_limit_mps=0.0,
                summary_text="TEB relay published a stop because no fresh command was received within the timeout.",
            )
            rospy.logwarn_throttle(
                self.log_period_s,
                "teb_cmd_vel_relay: no fresh cmd from %s for %.2fs",
                self.input_topic,
                self.idle_timeout_s,
            )
            self._publish_debug_text(
                "relay reason=stale_cmd_timeout action=idle_stop raw(v=0.000,w=0.000) sanitized(v=0.000,w=0.000) out(v=0.000,w=0.000) "
                "local(n={},empty={},remain={}) avoid={} hold={} final_brake={} behavior(stop={},limit={}) obs(stop_x={},slow_x={})".format(
                    int(self.local_path_pose_count),
                    "yes" if self.local_path_empty else "no",
                    self._fmt_debug_float(self.local_path_remaining_m),
                    "on" if self.avoidance_path_active else "off",
                    "on" if self._has_local_hold(now) else "off",
                    "on" if self._is_final_goal_brake_active() else "off",
                    "yes" if self.behavior_stop else "no",
                    self._fmt_debug_float(self.behavior_speed_limit)
                    if self.behavior_speed_limit < float("inf")
                    else "inf",
                    self._fmt_debug_float(self.closest_stop_obstacle_x),
                    self._fmt_debug_float(self.closest_slow_obstacle_x),
                ),
                now=now,
            )
            return

        cmd_before = self._sanitize_cmd(self.last_cmd)
        cmd = cmd_before
        explain = None

        cmd, info = self._apply_behavior_safety(cmd, now)
        if info is not None:
            explain = info

        if not (explain and explain.get("stop_commanded", False)):
            cmd, info = self._apply_local_hold(cmd, now)
            if info is not None:
                explain = info

        if not (explain and explain.get("stop_commanded", False)):
            cmd, info = self._apply_obstacle_safety(cmd, now)
            if info is not None:
                explain = info

        cmd = self._apply_cmd_smoothing(
            cmd,
            now,
            bypass=bool(explain and explain.get("stop_commanded", False))
            or self._is_final_goal_brake_active(),
        )
        self._safe_publish_cmd(cmd)
        self.last_publish_cmd = cmd
        self.last_publish_time = now
        if explain is not None:
            self._publish_explainability(
                event_type=explain.get("event_type", "CONTROL_ACTION_CHANGE"),
                trigger_reason=explain.get("trigger_reason", ""),
                action_taken=explain.get("action_taken", ""),
                local_planning_active=bool(explain.get("local_planning_active", False)),
                stop_commanded=bool(explain.get("stop_commanded", False)),
                slowdown_commanded=bool(explain.get("slowdown_commanded", False)),
                speed_before_mps=float(cmd_before.linear.x),
                speed_after_mps=float(cmd.linear.x),
                speed_limit_mps=float(explain.get("speed_limit_mps", -1.0)),
                closest_obstacle_dist_m=float(explain.get("closest_obstacle_dist_m", -1.0)),
                obstacle_lateral_offset_m=float(explain.get("obstacle_lateral_offset_m", -1.0)),
                summary_text=explain.get("summary_text", ""),
            )
        else:
            self._publish_explainability(
                event_type="CONTROL_ACTION_CHANGE",
                trigger_reason="clear",
                action_taken="follow_teb",
                speed_before_mps=float(cmd_before.linear.x),
                speed_after_mps=float(cmd.linear.x),
                summary_text="TEB relay is forwarding the optimized local planner command without additional intervention.",
            )

        self._publish_debug_text(
            self._build_debug_text(self.last_cmd, cmd_before, cmd, explain, now),
            now=now,
            force=explain is not None,
        )

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
