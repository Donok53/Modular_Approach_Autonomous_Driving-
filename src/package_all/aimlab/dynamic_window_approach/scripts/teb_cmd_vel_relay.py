#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rospy
from geometry_msgs.msg import PoseStamped, Twist
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
        self.final_goal_topic = str(
            rospy.get_param("~final_goal_topic", "/move_base_simple/goal")
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
        self.defer_local_hold_while_avoidance_active = bool(
            rospy.get_param("~defer_local_hold_while_avoidance_active", True)
        )
        self.robot_width_m = max(0.1, float(rospy.get_param("~robot_width_m", 0.58)))
        self.robot_length_m = max(0.1, float(rospy.get_param("~robot_length_m", 0.612)))
        self.footprint_padding_m = max(0.0, float(rospy.get_param("~footprint_padding_m", 0.0)))
        self.obstacle_cloud_topic = rospy.get_param(
            "~obstacle_cloud_topic", "/move_base/filtered_obstacles"
        )
        self.obstacle_cloud_timeout_s = max(
            0.1, float(rospy.get_param("~obstacle_cloud_timeout_s", 0.6))
        )
        self.enable_blind_zone_turn_guard = bool(
            rospy.get_param("~enable_blind_zone_turn_guard", True)
        )
        self.blind_zone_turn_guard_radius_m = max(
            0.0, float(rospy.get_param("~blind_zone_turn_guard_radius_m", 1.40))
        )
        self.blind_zone_turn_guard_ttl_s = max(
            0.0, float(rospy.get_param("~blind_zone_turn_guard_ttl_s", 0.45))
        )
        self.blind_zone_turn_guard_side_margin_m = max(
            0.0, float(rospy.get_param("~blind_zone_turn_guard_side_margin_m", 0.08))
        )
        self.blind_zone_turn_guard_front_margin_m = max(
            0.0, float(rospy.get_param("~blind_zone_turn_guard_front_margin_m", 0.15))
        )
        self.blind_zone_turn_guard_support_radius_m = max(
            0.0, float(rospy.get_param("~blind_zone_turn_guard_support_radius_m", 0.18))
        )
        self.blind_zone_turn_guard_min_points = max(
            1, int(rospy.get_param("~blind_zone_turn_guard_min_points", 2))
        )
        self.blind_zone_front_hold_min_points = max(
            self.blind_zone_turn_guard_min_points,
            int(rospy.get_param("~blind_zone_front_hold_min_points", 3)),
        )
        self.blind_zone_turn_guard_angular_deadband = max(
            0.0,
            float(rospy.get_param("~blind_zone_turn_guard_angular_deadband", 0.03)),
        )
        self.blind_zone_turn_guard_linear_cap_mps = max(
            0.0,
            float(rospy.get_param("~blind_zone_turn_guard_linear_cap_mps", 0.12)),
        )
        self.near_goal_side_turn_guard_stop_distance_m = max(
            self.final_brake_distance_m,
            self.ignore_local_hold_near_goal_distance_m,
            float(
                rospy.get_param(
                    "~near_goal_side_turn_guard_stop_distance_m",
                    0.60,
                )
            ),
        )
        self.behavior_cmd_timeout_s = max(
            0.1, float(rospy.get_param("~behavior_cmd_timeout_s", 0.8))
        )
        self.rear_approach_conservative_enabled = bool(
            rospy.get_param("~rear_approach_conservative_enabled", True)
        )
        self.rear_approach_max_angular_speed = max(
            0.0, float(rospy.get_param("~rear_approach_max_angular_speed", 0.12))
        )
        self.rear_approach_spin_linear_threshold_mps = max(
            0.0, float(rospy.get_param("~rear_approach_spin_linear_threshold_mps", 0.05))
        )
        self.rear_approach_no_reverse = bool(
            rospy.get_param("~rear_approach_no_reverse", True)
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
        self.emergency_stop_far_lateral_ratio = max(
            0.0,
            min(
                1.0,
                float(rospy.get_param("~emergency_stop_far_lateral_ratio", 0.55)),
            ),
        )
        self.slowdown_front_margin = max(
            self.emergency_stop_front_margin + 0.05,
            float(rospy.get_param("~slowdown_front_margin", 0.80)),
        )
        self.slowdown_side_margin = max(
            self.emergency_stop_side_margin,
            float(rospy.get_param("~slowdown_side_margin", 0.20)),
        )
        self.slowdown_far_lateral_ratio = max(
            self.emergency_stop_far_lateral_ratio,
            min(
                1.0,
                float(rospy.get_param("~slowdown_far_lateral_ratio", 0.75)),
            ),
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
        self.near_goal_obstacle_ignore_after_avoidance_s = max(
            0.0,
            float(
                rospy.get_param("~near_goal_obstacle_ignore_after_avoidance_s", 3.0)
            ),
        )
        self.robot_half_length = 0.5 * self.robot_length_m + self.footprint_padding_m
        self.robot_half_width = 0.5 * self.robot_width_m + self.footprint_padding_m
        self.blind_zone_turn_guard_front_limit_x = (
            self.robot_half_length + self.blind_zone_turn_guard_front_margin_m
        )
        self.blind_zone_turn_guard_side_forward_limit_x = max(
            0.0,
            float(
                rospy.get_param(
                    "~blind_zone_turn_guard_side_forward_limit_x",
                    self.blind_zone_turn_guard_front_limit_x + 0.10,
                )
            ),
        )
        self.blind_zone_turn_guard_side_lateral_limit_y = max(
            self.robot_half_width + self.blind_zone_turn_guard_side_margin_m,
            float(
                rospy.get_param(
                    "~blind_zone_turn_guard_side_lateral_limit_y",
                    min(
                        self.blind_zone_turn_guard_radius_m,
                        self.robot_half_width
                        + self.blind_zone_turn_guard_side_margin_m
                        + 0.40,
                    ),
                )
            ),
        )
        self.emergency_stop_distance = self.robot_half_length + self.emergency_stop_front_margin
        self.emergency_stop_lateral_y = self.robot_half_width + self.emergency_stop_side_margin
        self.slowdown_distance = self.robot_half_length + self.slowdown_front_margin
        self.slowdown_lateral_y = self.robot_half_width + self.slowdown_side_margin
        # During an active avoidance branch we still need a hard center-front
        # stop box, but we should not blanket-stop for every obstacle that sits
        # slightly off-center while TEB is trying to arc around it.
        self.avoidance_center_stop_distance = max(
            self.robot_half_length + 0.12,
            self.emergency_stop_distance - 0.15,
        )
        self.avoidance_escape_crawl_distance = max(
            self.robot_half_length + 0.08,
            self.avoidance_center_stop_distance - 0.08,
        )
        self.avoidance_center_lateral_y = max(
            0.06,
            min(self.emergency_stop_lateral_y, 0.20 * self.robot_half_width),
        )
        self.avoidance_close_crawl_speed = max(self.min_abs_linear_speed, 0.08)
        self.enable_tiny_reverse_forward_crawl = bool(
            rospy.get_param("~enable_tiny_reverse_forward_crawl", True)
        )
        self.tiny_reverse_forward_crawl_speed = max(
            0.0,
            float(
                rospy.get_param(
                    "~tiny_reverse_forward_crawl_speed",
                    max(self.reverse_replacement_speed, self.min_abs_linear_speed),
                )
            ),
        )
        self.tiny_reverse_forward_crawl_clearance_m = max(
            0.0,
            float(
                rospy.get_param(
                    "~tiny_reverse_forward_crawl_clearance_m",
                    self.emergency_stop_distance + 0.30,
                )
            ),
        )
        self.tiny_reverse_forward_crawl_max_angular_speed = max(
            0.0,
            float(
                rospy.get_param(
                    "~tiny_reverse_forward_crawl_max_angular_speed", 0.35
                )
            ),
        )
        self.require_recent_nonempty_local_path_for_crawl = bool(
            rospy.get_param("~require_recent_nonempty_local_path_for_crawl", True)
        )
        self.disable_near_goal_forward_crawl = bool(
            rospy.get_param("~disable_near_goal_forward_crawl", True)
        )

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
        self.local_path_empty = True
        self.local_path_pose_count = 0
        self.local_path_remaining_m = float("inf")
        self.last_nonempty_local_path_remaining_m = float("inf")
        self.last_avoidance_path_time = 0.0
        self.last_avoidance_active_time = 0.0
        self.avoidance_path_active = False
        self.avoidance_turn_direction = "none"
        self.avoidance_turn_lateral_m = 0.0
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.have_odom = False
        self.final_goal_x = 0.0
        self.final_goal_y = 0.0
        self.have_final_goal = False
        self.closest_stop_obstacle_x = float("inf")
        self.closest_stop_obstacle_y = 0.0
        self.closest_slow_obstacle_x = float("inf")
        self.closest_slow_obstacle_y = 0.0
        self.blind_zone_memory_time = 0.0
        self.blind_zone_memory_x = float("inf")
        self.blind_zone_memory_y = 0.0
        self.blind_zone_memory_side = 0
        self.blind_zone_memory_range_m = float("inf")
        self.blind_zone_memory_support_points = 0
        self._last_explain_key = None
        self._last_explain_time = 0.0
        self._last_debug_text = ""
        self._last_debug_text_time = 0.0
        self._last_debug_screen_time = 0.0
        self._last_sanitize_reason = "init"
        self._last_safety_reason = "init"
        self._last_smoothing_reason = "init"
        self._last_target_cmd = Twist()

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
        self.final_goal_sub = None
        if self.final_goal_topic:
            self.final_goal_sub = rospy.Subscriber(
                self.final_goal_topic, PoseStamped, self.final_goal_callback, queue_size=2
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
            "teb_cmd_vel_relay started | in=%s out=%s debug=%s explain=%s behavior=%s odom=%s local=%s avoidance=%s final_goal=%s publish=%.1fHz min|v|=%.3f estop=%s slowdown=%s hold_stop=%s hold_requires_avoid=%s defer_hold_on_avoid=%s smoothing=%s slew(v=%.2f,w=%.2f) footprint=%.2fx%.2fm stop=%.2fm/%.2fm(r=%.2f) avoid_stop=%.2fm/%.2fm crawl=%.2f slow=%.2fm/%.2fm(r=%.2f) final_brake=%.2fm hold_ignore=%.2fm rotate_near_obs=%s hold_rotate=%s blind_guard=%s(r=%.2f ttl=%.1f cap=%.2f side_x<=%.2f |y|<=%.2f)",
            self.input_topic,
            self.output_topic,
            self.debug_text_topic if self.debug_text_topic else "-",
            self.explainability_topic if self.explainability_topic else "-",
            self.behavior_cmd_topic if self.behavior_cmd_topic else "-",
            self.odom_topic if self.odom_topic else "-",
            self.local_path_topic if self.local_path_topic else "-",
            self.avoidance_path_topic if self.avoidance_path_topic else "-",
            self.final_goal_topic if self.final_goal_topic else "-",
            self.publish_hz,
            self.min_abs_linear_speed,
            "on" if self.enable_emergency_stop else "off",
            "on" if self.enable_obstacle_slowdown else "off",
            "on" if self.enable_local_path_hold_stop else "off",
            "on" if self.hold_requires_avoidance_path else "off",
            "on" if self.defer_local_hold_while_avoidance_active else "off",
            "on" if self.enable_cmd_smoothing else "off",
            self.max_linear_slew_mps2,
            self.max_angular_slew_rps2,
            self.robot_length_m + 2.0 * self.footprint_padding_m,
            self.robot_width_m + 2.0 * self.footprint_padding_m,
            self.emergency_stop_distance,
            self.emergency_stop_lateral_y,
            self.emergency_stop_far_lateral_ratio,
            self.avoidance_center_stop_distance,
            self.avoidance_center_lateral_y,
            self.avoidance_close_crawl_speed,
            self.slowdown_distance,
            self.slowdown_lateral_y,
            self.slowdown_far_lateral_ratio,
            self.final_brake_distance_m,
            self.ignore_local_hold_near_goal_distance_m,
            "on" if self.allow_in_place_rotation_near_obstacle else "off",
            "on" if self.allow_in_place_rotation_during_local_hold else "off",
            "on" if self.enable_blind_zone_turn_guard else "off",
            self.blind_zone_turn_guard_radius_m,
            self.blind_zone_turn_guard_ttl_s,
            self.blind_zone_turn_guard_linear_cap_mps,
            self.blind_zone_turn_guard_side_forward_limit_x,
            self.blind_zone_turn_guard_side_lateral_limit_y,
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

    @staticmethod
    def _copy_cmd(cmd):
        out = Twist()
        out.linear.x = float(cmd.linear.x)
        out.linear.y = float(cmd.linear.y)
        out.linear.z = float(cmd.linear.z)
        out.angular.x = float(cmd.angular.x)
        out.angular.y = float(cmd.angular.y)
        out.angular.z = float(cmd.angular.z)
        return out

    @staticmethod
    def _is_rear_approach_reason(reason):
        return str(reason).strip().lower().startswith("rear_vehicle_approach")

    def _apply_rear_approach_conservative(self, cmd):
        out = self._copy_cmd(cmd)
        flags = []

        if self.rear_approach_no_reverse and out.linear.x < 0.0:
            out.linear.x = 0.0
            flags.append("block_reverse")

        if (
            self.rear_approach_max_angular_speed > 0.0
            and abs(float(out.angular.z)) > self.rear_approach_max_angular_speed
        ):
            out.angular.z = math.copysign(
                self.rear_approach_max_angular_speed, float(out.angular.z)
            )
            flags.append("limit_angular")

        if (
            abs(float(out.linear.x)) <= self.rear_approach_spin_linear_threshold_mps
            and abs(float(out.angular.z)) > 1e-4
        ):
            out.angular.z = 0.0
            flags.append("suppress_spin")

        return out, flags

    def _forward_zone_lateral_limit(
        self, obstacle_x, base_lateral_y, far_lateral_ratio, zone_distance
    ):
        limit_y = max(0.0, float(base_lateral_y))
        if limit_y <= 0.0:
            return 0.0

        x = max(0.0, float(obstacle_x))
        taper_start_x = min(max(0.0, self.robot_half_length), float(zone_distance))
        if x <= taper_start_x:
            return limit_y

        span = max(1e-3, float(zone_distance) - taper_start_x)
        progress = max(0.0, min(1.0, (x - taper_start_x) / span))
        far_ratio = max(0.0, min(1.0, float(far_lateral_ratio)))
        scale = 1.0 - ((1.0 - far_ratio) * progress)
        return limit_y * scale

    @staticmethod
    def _join_diag_flags(flags):
        clean = [str(flag).strip() for flag in flags if str(flag).strip()]
        return "+".join(clean) if clean else "pass"

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
        goal_dist_text = self._fmt_debug_float(self._final_goal_distance_m())
        return (
            "relay reason={} action={} sanitize={} safety={} smooth={} raw(v={},w={}) sanitized(v={},w={}) target(v={},w={}) out(v={},w={}) "
            "local(n={},empty={},remain={}) goal_dist={} avoid={} dir={} hold={} final_brake={} "
            "behavior(stop={},limit={}) obs(stop_x={},slow_x={})"
        ).format(
            trigger_reason,
            action_taken,
            self._last_sanitize_reason,
            self._last_safety_reason,
            self._last_smoothing_reason,
            self._fmt_debug_float(raw_cmd.linear.x, 3),
            self._fmt_debug_float(raw_cmd.angular.z, 3),
            self._fmt_debug_float(sanitized_cmd.linear.x, 3),
            self._fmt_debug_float(sanitized_cmd.angular.z, 3),
            self._fmt_debug_float(self._last_target_cmd.linear.x, 3),
            self._fmt_debug_float(self._last_target_cmd.angular.z, 3),
            self._fmt_debug_float(final_cmd.linear.x, 3),
            self._fmt_debug_float(final_cmd.angular.z, 3),
            int(self.local_path_pose_count),
            "yes" if self.local_path_empty else "no",
            self._fmt_debug_float(self.local_path_remaining_m),
            goal_dist_text,
            "on" if self.avoidance_path_active else "off",
            self.avoidance_turn_direction,
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
        self._last_smoothing_reason = "pass"
        if bypass or (not self.enable_cmd_smoothing):
            self._last_smoothing_reason = "bypass" if bypass else "disabled"
            return cmd
        if self.last_publish_time <= 0.0:
            self._last_smoothing_reason = "unprimed"
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

        smooth_flags = []
        if abs(float(cmd.linear.x)) <= 1e-4 and abs(float(out.linear.x)) > 1e-4:
            smooth_flags.append("cut_linear")
            rospy.loginfo_throttle(
                self.log_period_s,
                "teb_cmd_vel_relay: cutting linear carry for zero target prev_v=%.3f target_w=%.3f",
                float(self.last_publish_cmd.linear.x),
                float(cmd.angular.z),
            )
            out.linear.x = 0.0
        if (
            abs(float(cmd.linear.x)) <= 1e-4
            and abs(float(cmd.angular.z)) <= 1e-4
            and abs(float(out.angular.z)) > 1e-4
        ):
            smooth_flags.append("cut_angular")
            rospy.loginfo_throttle(
                self.log_period_s,
                "teb_cmd_vel_relay: cutting angular carry for zero target prev_w=%.3f",
                float(self.last_publish_cmd.angular.z),
            )
            out.angular.z = 0.0
        if abs(out.linear.x - float(cmd.linear.x)) > 1e-4:
            if abs(float(cmd.linear.x)) <= 1e-4 and abs(float(out.linear.x)) > 1e-4:
                smooth_flags.append("carry_linear")
                rospy.loginfo_throttle(
                    self.log_period_s,
                    "teb_cmd_vel_relay: smoothing carried linear motion after zero target prev_v=%.3f target_v=%.3f out_v=%.3f",
                    float(self.last_publish_cmd.linear.x),
                    float(cmd.linear.x),
                    float(out.linear.x),
                )
            else:
                smooth_flags.append("limit_linear")
        if abs(out.angular.z - float(cmd.angular.z)) > 1e-4:
            if abs(float(cmd.angular.z)) <= 1e-4 and abs(float(out.angular.z)) > 1e-4:
                smooth_flags.append("carry_angular")
            else:
                smooth_flags.append("limit_angular")
        self._last_smoothing_reason = self._join_diag_flags(smooth_flags)

        if smooth_flags:
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
        now = rospy.get_time()
        final_brake_active = self._is_final_goal_brake_active()
        fresh_empty_local_path = (
            self._has_fresh_empty_local_path(now) and (not self.avoidance_path_active)
        )
        sanitize_flags = []
        if self._should_force_goal_stop(out, final_brake_active):
            self._last_sanitize_reason = "force_goal_stop"
            rospy.loginfo_throttle(
                self.log_period_s,
                "teb_cmd_vel_relay: forcing terminal goal stop remain=%.3f v=%.3f w=%.3f",
                self.local_path_remaining_m,
                out.linear.x,
                out.angular.z,
            )
            return Twist()
        reverse_clamped_to_stop = False
        if out.linear.x < 0.0 and abs(out.linear.x) <= self.reverse_deadband_mps:
            if final_brake_active or fresh_empty_local_path:
                if (
                    (not fresh_empty_local_path)
                    and self._should_replace_tiny_reverse_with_final_brake_crawl(out, now)
                ):
                    self._last_sanitize_reason = "tiny_reverse_final_brake_crawl"
                    rospy.loginfo_throttle(
                        self.log_period_s,
                        "teb_cmd_vel_relay: replacing tiny reverse near final brake with forward crawl v=%.3f -> %.3f (w=%.3f remain=%.2f stop_x=%.2f)",
                        out.linear.x,
                        self.tiny_reverse_forward_crawl_speed,
                        out.angular.z,
                        self.local_path_remaining_m,
                        self.closest_stop_obstacle_x,
                    )
                    return self._forward_crawl_cmd(out)
                if (not fresh_empty_local_path) and self._should_convert_small_reverse_to_rotation(out, now=now):
                    self._last_sanitize_reason = "tiny_reverse_goal_rotate"
                    rospy.loginfo_throttle(
                        self.log_period_s,
                        "teb_cmd_vel_relay: converting tiny reverse near goal to rotation-only v=%.3f w=%.3f",
                        out.linear.x,
                        out.angular.z,
                    )
                    return self._rotation_only_cmd(out)
                self._last_sanitize_reason = "tiny_reverse_goal_stop"
                rospy.loginfo_throttle(
                    self.log_period_s,
                    "teb_cmd_vel_relay: stopping tiny reverse near goal/local hold v=%.3f w=%.3f",
                    out.linear.x,
                    out.angular.z,
                )
                return Twist()
            if self._should_coast_through_tiny_reverse(
                out, final_brake_active, fresh_empty_local_path
            ):
                prev_v = float(self.last_publish_cmd.linear.x)
                self._last_sanitize_reason = "tiny_reverse_coast"
                rospy.loginfo_throttle(
                    self.log_period_s,
                    "teb_cmd_vel_relay: coasting through tiny reverse prev_v=%.3f raw(v=%.3f,w=%.3f)",
                    prev_v,
                    out.linear.x,
                    out.angular.z,
                )
                return self._coast_forward_cmd(out)
            if self._should_replace_tiny_reverse_with_forward_crawl(
                out,
                now,
                final_brake_active=final_brake_active,
                fresh_empty_local_path=fresh_empty_local_path,
            ):
                self._last_sanitize_reason = "tiny_reverse_to_forward_crawl"
                rospy.loginfo_throttle(
                    self.log_period_s,
                    "teb_cmd_vel_relay: replacing tiny reverse with forward crawl v=%.3f -> %.3f (w=%.3f stop_x=%.2f)",
                    out.linear.x,
                    self.tiny_reverse_forward_crawl_speed,
                    out.angular.z,
                    self.closest_stop_obstacle_x,
                )
                return self._forward_crawl_cmd(out)
            if self._should_convert_small_reverse_to_rotation(out, now=now):
                self._last_sanitize_reason = "tiny_reverse_to_rotation"
                rospy.loginfo_throttle(
                    self.log_period_s,
                    "teb_cmd_vel_relay: converting tiny reverse to rotation-only v=%.3f w=%.3f",
                    out.linear.x,
                    out.angular.z,
                )
                return self._rotation_only_cmd(out)
            sanitize_flags.append("tiny_reverse")
            rospy.loginfo_throttle(
                self.log_period_s,
                "teb_cmd_vel_relay: zeroing tiny reverse cmd v=%.3f (w=%.3f)",
                out.linear.x,
                out.angular.z,
            )
            out.linear.x = 0.0
            reverse_clamped_to_stop = True
            sanitize_flags.append("zero")
        if self.forward_only and out.linear.x < 0.0:
            if self._should_convert_small_reverse_to_rotation(out, now=now):
                self._last_sanitize_reason = "reverse_to_rotation"
                rospy.loginfo_throttle(
                    self.log_period_s,
                    "teb_cmd_vel_relay: converting reverse cmd to rotation-only v=%.3f w=%.3f",
                    out.linear.x,
                    out.angular.z,
                )
                return self._rotation_only_cmd(out)
            sanitize_flags.append("reverse_clamp")
            rospy.logwarn_throttle(
                self.log_period_s,
                "teb_cmd_vel_relay: clamping reverse cmd to stop v=%.3f",
                out.linear.x,
            )
            out.linear.x = 0.0
            reverse_clamped_to_stop = True
            sanitize_flags.append("zero")
        if reverse_clamped_to_stop and abs(out.angular.z) > 1e-4:
            if self._should_preserve_in_place_rotation(out):
                self._last_sanitize_reason = "reverse_clamped_to_rotation"
                rospy.loginfo_throttle(
                    self.log_period_s,
                    "teb_cmd_vel_relay: converting reverse-clamped cmd to in-place rotation w=%.3f",
                    out.angular.z,
                )
                return self._rotation_only_cmd(out)
            sanitize_flags.append("suppress_spin")
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
            sanitize_flags.append("min_boost")
        self._last_sanitize_reason = self._join_diag_flags(sanitize_flags)
        return out

    @staticmethod
    def _cmd_mag(cmd):
        return math.hypot(float(cmd.linear.x), float(cmd.angular.z))

    @staticmethod
    def _pose_dist_xy(x, y, pose_stamped):
        dx = float(pose_stamped.pose.position.x) - float(x)
        dy = float(pose_stamped.pose.position.y) - float(y)
        return math.hypot(dx, dy)

    def _pose_to_body_xy(self, pose_stamped):
        dx = float(pose_stamped.pose.position.x) - float(self.odom_x)
        dy = float(pose_stamped.pose.position.y) - float(self.odom_y)
        c = math.cos(self.odom_yaw)
        s = math.sin(self.odom_yaw)
        return (c * dx + s * dy, -s * dx + c * dy)

    def _infer_avoidance_turn_from_path(self, path_msg):
        if (not self.have_odom) or path_msg is None or len(path_msg.poses) < 2:
            return "none", 0.0

        best_lateral = 0.0
        best_score = 0.0
        min_forward_x = max(0.12, 0.5 * self.robot_half_length)
        max_considered_x = max(2.5, self.slowdown_distance + 0.5)
        for pose in path_msg.poses[1:]:
            x_local, y_local = self._pose_to_body_xy(pose)
            if x_local < -0.10 or x_local > max_considered_x:
                continue
            score = abs(y_local)
            if x_local >= min_forward_x:
                score += 0.05
            if score > best_score:
                best_score = score
                best_lateral = y_local

        if abs(best_lateral) < 0.05 and len(path_msg.poses) >= 3:
            sx, sy = self._pose_to_body_xy(path_msg.poses[0])
            ex, ey = self._pose_to_body_xy(path_msg.poses[-1])
            dx = float(ex) - float(sx)
            dy = float(ey) - float(sy)
            norm = math.hypot(dx, dy)
            if norm > 1e-6:
                signed_offset = 0.0
                for pose in path_msg.poses[1:-1]:
                    px, py = self._pose_to_body_xy(pose)
                    signed = (dx * (float(py) - float(sy)) - dy * (float(px) - float(sx))) / norm
                    if abs(signed) > abs(signed_offset):
                        signed_offset = signed
                if abs(signed_offset) > abs(best_lateral):
                    best_lateral = signed_offset

        if abs(best_lateral) < 0.05:
            return "none", best_lateral
        return ("left" if best_lateral > 0.0 else "right"), best_lateral

    def _should_preserve_in_place_rotation(self, cmd):
        return abs(float(cmd.angular.z)) >= self.min_in_place_rotation_angular_speed

    def _preferred_avoidance_turn_direction(self):
        if not self.avoidance_path_active:
            return "none"
        if self.avoidance_turn_direction in ("left", "right"):
            return self.avoidance_turn_direction
        return "none"

    def _preferred_avoidance_turn_sign(self):
        direction = self._preferred_avoidance_turn_direction()
        if direction == "left":
            return 1.0
        if direction == "right":
            return -1.0
        return 0.0

    def _has_preferred_avoidance_turn(self):
        return abs(self._preferred_avoidance_turn_sign()) > 0.0

    def _effective_avoidance_angular_z(self, cmd):
        raw_w = float(cmd.angular.z)
        preferred_sign = self._preferred_avoidance_turn_sign()
        if preferred_sign == 0.0:
            return raw_w
        return preferred_sign * max(abs(raw_w), self.min_in_place_rotation_angular_speed)

    def _is_turning_away_from_obstacle(self, cmd, obstacle_y):
        angular_z = self._effective_avoidance_angular_z(cmd)
        lateral_y = float(obstacle_y)
        return (
            abs(angular_z) >= self.min_in_place_rotation_angular_speed
            and abs(lateral_y) >= 1e-3
            and (angular_z * lateral_y) < 0.0
        )

    def _classify_blind_zone_candidate(self, obstacle_x, obstacle_y):
        if (not self.enable_blind_zone_turn_guard) or (
            self.blind_zone_turn_guard_radius_m <= 0.0
        ):
            return None

        x = float(obstacle_x)
        y = float(obstacle_y)
        if x < -0.10:
            return None

        distance = math.hypot(x, y)
        if distance > self.blind_zone_turn_guard_radius_m:
            return None

        side_limit = self.robot_half_width + self.blind_zone_turn_guard_side_margin_m
        side = 0
        if y >= side_limit:
            side = 1
        elif y <= -side_limit:
            side = -1
        elif (
            x >= 0.0
            and x <= self.blind_zone_turn_guard_front_limit_x
            and abs(y) <= side_limit
        ):
            side = 0
        else:
            return None

        if side != 0 and x <= 0.0:
            return None
        if side != 0 and x > self.blind_zone_turn_guard_side_forward_limit_x:
            # Treat side blind-zone memory as an immediate flank guard instead
            # of a broad forward-side keepout. This avoids vetoing near-goal
            # turns for people standing outside the robot path but slightly
            # ahead of the chassis.
            return None
        if side != 0 and abs(y) > self.blind_zone_turn_guard_side_lateral_limit_y:
            # Keep the side guard focused on the immediate flank. Wide lateral
            # returns can come from walls/people outside the robot path and
            # should not veto near-goal turns.
            return None

        score = distance
        if side == 0:
            score -= min(0.20, max(0.0, x) * 0.05)
        return score, x, y, side, distance

    def _blind_zone_min_support_points(self, side):
        return (
            self.blind_zone_front_hold_min_points
            if int(side) == 0
            else self.blind_zone_turn_guard_min_points
        )

    def _count_blind_zone_support_points(self, candidate, blind_zone_points):
        if candidate is None:
            return 0
        if self.blind_zone_turn_guard_support_radius_m <= 0.0:
            return 1

        _score, cx, cy, side, _distance = candidate
        support_radius_sq = (
            self.blind_zone_turn_guard_support_radius_m
            * self.blind_zone_turn_guard_support_radius_m
        )
        support_points = 0
        for px, py, pside in blind_zone_points:
            if int(pside) != int(side):
                continue
            dx = float(px) - float(cx)
            dy = float(py) - float(cy)
            if (dx * dx + dy * dy) <= support_radius_sq:
                support_points += 1
        return support_points

    def _update_blind_zone_memory(self, candidate, support_points, now):
        if candidate is None:
            return
        _score, x, y, side, distance = candidate
        self.blind_zone_memory_time = float(now)
        self.blind_zone_memory_x = float(x)
        self.blind_zone_memory_y = float(y)
        self.blind_zone_memory_side = int(side)
        self.blind_zone_memory_range_m = float(distance)
        self.blind_zone_memory_support_points = int(max(0, support_points))

    def _get_active_blind_zone_memory(self, now):
        if (not self.enable_blind_zone_turn_guard) or (
            self.blind_zone_turn_guard_ttl_s <= 0.0
        ):
            return None
        if self.blind_zone_memory_time <= 0.0:
            return None
        age = float(now) - float(self.blind_zone_memory_time)
        if age < 0.0 or age > self.blind_zone_turn_guard_ttl_s:
            return None
        return {
            "age_s": age,
            "x": float(self.blind_zone_memory_x),
            "y": float(self.blind_zone_memory_y),
            "side": int(self.blind_zone_memory_side),
            "range_m": float(self.blind_zone_memory_range_m),
            "support_points": int(self.blind_zone_memory_support_points),
        }

    def _safe_turn_direction_from_side(self, side):
        if side > 0:
            return "right"
        if side < 0:
            return "left"
        return "none"

    def _is_turning_toward_side(self, angular_z, side):
        angular = float(angular_z)
        if side > 0:
            return angular > self.blind_zone_turn_guard_angular_deadband
        if side < 0:
            return angular < -self.blind_zone_turn_guard_angular_deadband
        return False

    @staticmethod
    def _rotation_only_cmd(cmd):
        out = Twist()
        out.angular.z = float(cmd.angular.z)
        return out

    def _preferred_rotation_only_cmd(self, cmd):
        out = Twist()
        out.angular.z = self._effective_avoidance_angular_z(cmd)
        return out

    def _rotation_away_from_side_cmd(self, cmd, side):
        out = Twist()
        if side == 0:
            return out
        turn_sign = -1.0 if side > 0 else 1.0
        out.angular.z = turn_sign * max(
            abs(float(cmd.angular.z)), self.min_in_place_rotation_angular_speed
        )
        return out

    def _coast_forward_cmd(self, cmd):
        out = Twist()
        out.linear.x = max(0.0, float(self.last_publish_cmd.linear.x))
        out.angular.z = float(cmd.angular.z)
        return out

    def _forward_crawl_cmd(self, cmd):
        out = Twist()
        crawl_speed = max(
            0.0,
            float(self.tiny_reverse_forward_crawl_speed),
            float(self.reverse_replacement_speed),
        )
        if self.enforce_min_linear_speed:
            crawl_speed = max(crawl_speed, float(self.min_abs_linear_speed))
        out.linear.x = crawl_speed
        out.angular.z = float(cmd.angular.z)
        return out

    def _near_goal_blind_zone_crawl_speed(self, now):
        if self.disable_near_goal_forward_crawl:
            return 0.0
        if self.avoidance_path_active:
            return 0.0
        if not self._is_near_goal_tiny_reverse_crawl_window():
            return 0.0
        if not self._has_recent_nonempty_local_path(now):
            return 0.0
        if not self._has_fresh_obstacle_data(now):
            return 0.0
        if self._is_final_goal_brake_active():
            return 0.0
        if math.isfinite(self.closest_stop_obstacle_x) and (
            self.closest_stop_obstacle_x <= self.emergency_stop_distance
        ):
            return 0.0

        crawl_speed = max(
            0.0,
            float(self.tiny_reverse_forward_crawl_speed),
            float(self.reverse_replacement_speed),
        )
        if self.enforce_min_linear_speed:
            crawl_speed = max(crawl_speed, float(self.min_abs_linear_speed))
        if self.blind_zone_turn_guard_linear_cap_mps > 0.0:
            crawl_speed = min(crawl_speed, self.blind_zone_turn_guard_linear_cap_mps)
        return max(0.0, crawl_speed)

    def _avoidance_close_crawl_cmd(self, cmd):
        out = Twist()
        out.linear.x = min(
            max(0.0, float(cmd.linear.x)),
            self.avoidance_close_crawl_speed,
        )
        out.angular.z = self._effective_avoidance_angular_z(cmd)
        return out

    def _should_allow_avoidance_close_crawl(self, cmd, final_segment_active):
        if (not self.avoidance_path_active) or final_segment_active:
            return False
        if not self._has_preferred_avoidance_turn():
            return False
        if float(cmd.linear.x) <= 0.0:
            return False
        if not math.isfinite(self.closest_stop_obstacle_x):
            return False
        if self.closest_stop_obstacle_x <= self.avoidance_escape_crawl_distance:
            return False
        if abs(float(self.closest_stop_obstacle_y)) < self.avoidance_center_lateral_y:
            return False
        return self._is_turning_away_from_obstacle(cmd, self.closest_stop_obstacle_y)

    def obstacle_callback(self, msg):
        stop_min_x = float("inf")
        stop_min_y = 0.0
        slow_min_x = float("inf")
        slow_min_y = 0.0
        blind_zone_candidate = None
        blind_zone_score = float("inf")
        blind_zone_points = []

        for x, y, _z in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            blind_candidate = self._classify_blind_zone_candidate(x, y)
            if blind_candidate is not None:
                blind_zone_points.append(
                    (
                        float(blind_candidate[1]),
                        float(blind_candidate[2]),
                        int(blind_candidate[3]),
                    )
                )
                if blind_candidate[0] < blind_zone_score:
                    blind_zone_candidate = blind_candidate
                    blind_zone_score = blind_candidate[0]
            if x <= 0.0:
                continue
            abs_y = abs(float(y))
            slow_lateral_limit = self._forward_zone_lateral_limit(
                x,
                self.slowdown_lateral_y,
                self.slowdown_far_lateral_ratio,
                self.slowdown_distance,
            )
            if abs_y <= slow_lateral_limit:
                if float(x) < slow_min_x:
                    slow_min_x = float(x)
                    slow_min_y = float(y)
            stop_lateral_limit = self._forward_zone_lateral_limit(
                x,
                self.emergency_stop_lateral_y,
                self.emergency_stop_far_lateral_ratio,
                self.emergency_stop_distance,
            )
            if abs_y <= stop_lateral_limit:
                if float(x) < stop_min_x:
                    stop_min_x = float(x)
                    stop_min_y = float(y)

        now = rospy.get_time()
        self.last_obstacle_time = now
        self.closest_stop_obstacle_x = stop_min_x
        self.closest_stop_obstacle_y = stop_min_y
        self.closest_slow_obstacle_x = slow_min_x
        self.closest_slow_obstacle_y = slow_min_y
        blind_zone_support_points = self._count_blind_zone_support_points(
            blind_zone_candidate, blind_zone_points
        )
        if blind_zone_candidate is not None:
            required_support_points = self._blind_zone_min_support_points(
                blind_zone_candidate[3]
            )
            if blind_zone_support_points < required_support_points:
                blind_zone_candidate = None
        self._update_blind_zone_memory(
            blind_zone_candidate, blind_zone_support_points, now
        )

    def odom_callback(self, msg):
        self.odom_x = float(msg.pose.pose.position.x)
        self.odom_y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self.odom_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.have_odom = True

    def behavior_callback(self, msg):
        self.last_behavior_time = rospy.get_time()
        self.behavior_stop = bool(msg.stop)
        self.behavior_speed_limit = max(0.0, float(msg.speed_limit))
        self.behavior_reason = str(msg.reason).strip() if str(msg.reason).strip() else "behavior"

    def final_goal_callback(self, msg):
        self.final_goal_x = float(msg.pose.position.x)
        self.final_goal_y = float(msg.pose.position.y)
        self.have_final_goal = True

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
        was_active = bool(self.avoidance_path_active)
        self.avoidance_path_active = len(msg.poses) >= 2
        if not self.avoidance_path_active:
            self.avoidance_turn_direction = "none"
            self.avoidance_turn_lateral_m = 0.0
            return
        self.last_avoidance_active_time = self.last_avoidance_path_time

        direction, lateral = self._infer_avoidance_turn_from_path(msg)
        if direction != "none":
            self.avoidance_turn_lateral_m = float(lateral)
        if (not was_active) and direction != "none":
            self.avoidance_turn_direction = direction
        elif self.avoidance_turn_direction == "none" and direction != "none":
            self.avoidance_turn_direction = direction

    def _is_final_path_segment_active(self):
        return (
            self.local_path_pose_count > 0
            and self.local_path_pose_count <= self.final_path_pose_threshold
            and not self.avoidance_path_active
        )

    def _final_goal_distance_m(self):
        if (not self.have_odom) or (not self.have_final_goal):
            return float("inf")
        return math.hypot(
            float(self.final_goal_x) - float(self.odom_x),
            float(self.final_goal_y) - float(self.odom_y),
        )

    def _is_near_goal_side_turn_guard_stop_zone(self):
        goal_distance_m = self._final_goal_distance_m()
        return math.isfinite(goal_distance_m) and (
            goal_distance_m <= self.near_goal_side_turn_guard_stop_distance_m
        )

    def _is_final_goal_brake_active(self):
        if (
            self._is_final_path_segment_active()
            and math.isfinite(self.local_path_remaining_m)
            and self.local_path_remaining_m <= self.final_brake_distance_m
        ):
            return True
        goal_distance_m = self._final_goal_distance_m()
        return math.isfinite(goal_distance_m) and (
            goal_distance_m <= self.final_brake_distance_m
        )

    def _terminal_goal_stop_distance_m(self):
        return min(
            self.final_brake_distance_m,
            max(0.12, 0.8 * self.final_cmd_linear_stop_threshold),
        )

    def _is_terminal_goal_stop_active(self):
        stop_distance_m = self._terminal_goal_stop_distance_m()
        if (
            self._is_final_path_segment_active()
            and math.isfinite(self.local_path_remaining_m)
            and self.local_path_remaining_m <= stop_distance_m
        ):
            return True
        goal_distance_m = self._final_goal_distance_m()
        return math.isfinite(goal_distance_m) and (goal_distance_m <= stop_distance_m)

    def _should_force_goal_stop(self, cmd, final_brake_active=None):
        if final_brake_active is None:
            final_brake_active = self._is_final_goal_brake_active()
        if not final_brake_active:
            return False
        return self._is_terminal_goal_stop_active()

    def _should_convert_small_reverse_to_rotation(self, cmd, now=None):
        # Converting a tiny reverse into an in-place rotation is safer than
        # letting a forward-only robot stall at startup just because the local
        # planner has not yet published a fresh nonempty path.
        return (
            self.forward_only
            and float(cmd.linear.x) < 0.0
            and abs(float(cmd.linear.x))
            <= max(self.reverse_deadband_mps, self.min_abs_linear_speed)
            and self._should_preserve_in_place_rotation(cmd)
        )

    def _is_near_goal_tiny_reverse_crawl_window(self):
        if self.disable_near_goal_forward_crawl:
            return False
        if self.avoidance_path_active:
            return False
        if not math.isfinite(self.local_path_remaining_m):
            return False

        # Near the last few poses TEB sometimes asks for a tiny reverse even
        # though the robot still has a short forward segment left. If we are
        # outside the terminal brake zone, prefer a controlled forward crawl
        # instead of zeroing the command and stalling short of the goal.
        near_goal_pose_window = (
            self.local_path_pose_count > 0
            and self.local_path_pose_count <= (self.final_path_pose_threshold + 1)
        )
        near_goal_distance_window = (
            self.local_path_remaining_m <= self.ignore_local_hold_near_goal_distance_m
        )
        if (not near_goal_pose_window) and (not near_goal_distance_window):
            return False

        return self.local_path_remaining_m > max(
            self.final_brake_distance_m,
            self._terminal_goal_stop_distance_m(),
        )

    def _should_replace_tiny_reverse_with_forward_crawl(
        self,
        cmd,
        now,
        final_brake_active=False,
        fresh_empty_local_path=False,
    ):
        if (not self.enable_tiny_reverse_forward_crawl) or (not self.forward_only):
            return False
        if final_brake_active or fresh_empty_local_path:
            return False
        if self.require_recent_nonempty_local_path_for_crawl and (
            not self._has_recent_nonempty_local_path(now)
        ):
            return False
        if float(cmd.linear.x) >= 0.0:
            return False
        if abs(float(cmd.linear.x)) > max(
            self.reverse_deadband_mps, self.min_abs_linear_speed
        ):
            return False
        if abs(float(cmd.angular.z)) > self.tiny_reverse_forward_crawl_max_angular_speed:
            return False
        near_goal_crawl_window = self._is_near_goal_tiny_reverse_crawl_window()
        if (
            self.local_path_pose_count > 0
            and self.local_path_pose_count <= (self.final_path_pose_threshold + 1)
            and (not near_goal_crawl_window)
        ):
            return False
        if (
            math.isfinite(self.local_path_remaining_m)
            and self.local_path_remaining_m <= self.ignore_local_hold_near_goal_distance_m
            and (not near_goal_crawl_window)
        ):
            return False
        if not self._has_fresh_obstacle_data(now):
            return False
        if math.isfinite(self.closest_stop_obstacle_x):
            return self.closest_stop_obstacle_x > self.tiny_reverse_forward_crawl_clearance_m
        return True

    def _should_replace_tiny_reverse_with_final_brake_crawl(self, cmd, now):
        if (not self.enable_tiny_reverse_forward_crawl) or (not self.forward_only):
            return False
        if self.disable_near_goal_forward_crawl:
            return False
        if self.require_recent_nonempty_local_path_for_crawl and (
            not self._has_recent_nonempty_local_path(now)
        ):
            return False
        if not self._is_final_goal_brake_active():
            return False
        if not math.isfinite(self.local_path_remaining_m):
            return False
        if self.local_path_remaining_m <= self._terminal_goal_stop_distance_m():
            return False
        if float(cmd.linear.x) >= 0.0:
            return False
        if abs(float(cmd.linear.x)) > max(
            self.reverse_deadband_mps, self.min_abs_linear_speed
        ):
            return False
        if abs(float(cmd.angular.z)) > self.tiny_reverse_forward_crawl_max_angular_speed:
            return False
        if not self._has_fresh_obstacle_data(now):
            return False
        return bool(
            self._should_ignore_obstacle_for_local_goal(self.closest_stop_obstacle_x)
            or self._should_ignore_obstacle_for_local_goal(self.closest_slow_obstacle_x)
        )

    def _should_coast_through_tiny_reverse(
        self, cmd, final_brake_active=False, fresh_empty_local_path=False
    ):
        if not self.forward_only:
            return False
        if final_brake_active or fresh_empty_local_path:
            return False
        if self.require_recent_nonempty_local_path_for_crawl and (
            not self._has_recent_nonempty_local_path()
        ):
            return False
        if float(cmd.linear.x) >= 0.0:
            return False
        if abs(float(cmd.linear.x)) > max(self.reverse_deadband_mps, self.min_abs_linear_speed):
            return False
        prev_v = float(self.last_publish_cmd.linear.x)
        if prev_v <= max(self.reverse_replacement_speed, self.min_abs_linear_speed):
            return False
        if (
            self.local_path_pose_count > 0
            and self.local_path_pose_count <= (self.final_path_pose_threshold + 1)
        ):
            return False
        if (
            math.isfinite(self.local_path_remaining_m)
            and self.local_path_remaining_m <= self.ignore_local_hold_near_goal_distance_m
        ):
            return False
        return True

    def _has_fresh_empty_local_path(self, now=None):
        now_sec = rospy.get_time() if now is None else float(now)
        return (
            self.last_local_path_time > 0.0
            and (now_sec - self.last_local_path_time) <= self.local_hold_timeout_s
            and self.local_path_empty
        )

    def _has_recent_nonempty_local_path(self, now=None):
        now_sec = rospy.get_time() if now is None else float(now)
        return (
            self.last_local_path_time > 0.0
            and (now_sec - self.last_local_path_time) <= self.local_hold_timeout_s
            and (not self.local_path_empty)
            and self.local_path_pose_count >= 2
            and math.isfinite(self.local_path_remaining_m)
        )

    def _has_seen_nonempty_local_path(self):
        return math.isfinite(self.last_nonempty_local_path_remaining_m)

    def _has_fresh_near_goal_empty_local_path(self, now=None):
        if not self._has_fresh_empty_local_path(now):
            return False
        return (
            (not self.avoidance_path_active)
            and math.isfinite(self.last_nonempty_local_path_remaining_m)
            and self.last_nonempty_local_path_remaining_m
            <= self.ignore_local_hold_near_goal_distance_m
        )

    def _should_hold_before_avoidance_path(self):
        # constrained_local_replanner publishes an empty local path during the
        # "hold before avoidance" phase. If we keep waiting for an avoidance
        # path to appear before honoring that hold, stale TEB commands can keep
        # driving the robot into the obstacle that triggered replanning.
        if self.avoidance_path_active:
            return False
        if not math.isfinite(self.last_nonempty_local_path_remaining_m):
            return False
        return (
            self.last_nonempty_local_path_remaining_m
            > self.ignore_local_hold_near_goal_distance_m
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
        if not self._has_seen_nonempty_local_path():
            # Do not arm the local-hold stop before the planner has ever
            # produced a real local path. Otherwise the robot can freeze right
            # after a new goal while TEB is still spinning up.
            return False
        if self._has_fresh_near_goal_empty_local_path(now):
            return True
        # During doorway / branch detours the nominal local path is expected to stay
        # empty while the avoidance branch remains active. In that case we should
        # keep following the avoidance branch instead of zeroing cmd_vel.
        if self.defer_local_hold_while_avoidance_active and self.avoidance_path_active:
            return False
        if self.hold_requires_avoidance_path and (not self.avoidance_path_active):
            return self._should_hold_before_avoidance_path()
        if (
            (not self.avoidance_path_active)
            and math.isfinite(self.last_nonempty_local_path_remaining_m)
            and self.last_nonempty_local_path_remaining_m <= self.ignore_local_hold_near_goal_distance_m
        ):
            return False
        if self.last_avoidance_path_time > 0.0 and (now - self.last_avoidance_path_time) <= self.local_hold_timeout_s:
            return not self.avoidance_path_active
        return True

    def _had_recent_avoidance_path(self, now=None):
        if self.avoidance_path_active:
            return True
        if self.near_goal_obstacle_ignore_after_avoidance_s <= 0.0:
            return False
        last_active_time = float(self.last_avoidance_active_time)
        if last_active_time <= 0.0:
            return False
        if now is None:
            now = rospy.get_time()
        return now > 0.0 and (
            now - last_active_time
        ) <= self.near_goal_obstacle_ignore_after_avoidance_s

    def _should_ignore_obstacle_for_local_goal(self, obstacle_x):
        if (not self.ignore_obstacles_beyond_local_goal) or (not self.have_odom):
            return False
        if not math.isfinite(obstacle_x):
            return False
        if not math.isfinite(self.local_path_remaining_m):
            return False
        if self.local_path_remaining_m <= 1e-3:
            return False
        if self._had_recent_avoidance_path():
            return False
        if self.local_path_remaining_m > self.near_goal_obstacle_ignore_distance_m:
            return False
        return obstacle_x > (self.local_path_remaining_m + self.obstacle_beyond_goal_slack_m)

    def _publish_explainability(
        self,
        event_type,
        stamp=None,
        trigger_reason="",
        action_taken="",
        avoid_direction="none",
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
        msg.avoid_direction = str(avoid_direction)
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
            self._last_safety_reason = "behavior_stop"
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

        limited = self._copy_cmd(cmd)
        changed = False
        slowdown_commanded = False
        speed_limit_mps = -1.0
        action_taken = "follow_teb"
        summary_parts = []
        rear_flags = []

        if self.behavior_speed_limit < float("inf") and abs(limited.linear.x) > self.behavior_speed_limit:
            raw_linear = float(limited.linear.x)
            limited.linear.x = math.copysign(self.behavior_speed_limit, raw_linear)
            rospy.logwarn_throttle(
                self.log_period_s,
                "teb_cmd_vel_relay: behavior speed limit | reason=%s v=%.3f -> %.3f",
                self.behavior_reason,
                raw_linear,
                float(limited.linear.x),
            )
            changed = True
            slowdown_commanded = True
            speed_limit_mps = float(self.behavior_speed_limit)
            action_taken = "slowdown"
            summary_parts.append(
                "TEB relay reduced the linear speed because the behavior layer requested a speed limit."
            )

        if (
            self.rear_approach_conservative_enabled
            and self._is_rear_approach_reason(self.behavior_reason)
        ):
            adjusted, rear_flags = self._apply_rear_approach_conservative(limited)
            if rear_flags:
                rospy.loginfo_throttle(
                    self.log_period_s,
                    "teb_cmd_vel_relay: rear-approach conservative | reason=%s flags=%s cmd(v=%.3f,w=%.3f) -> out(v=%.3f,w=%.3f)",
                    self.behavior_reason,
                    ",".join(rear_flags),
                    float(limited.linear.x),
                    float(limited.angular.z),
                    float(adjusted.linear.x),
                    float(adjusted.angular.z),
                )
                limited = adjusted
                changed = True
                action_taken = (
                    "slowdown+rear_approach_conservative"
                    if slowdown_commanded
                    else "rear_approach_conservative"
                )
                summary_parts.append(
                    "TEB relay kept the command conservative for a rear vehicle approach ({}).".format(
                        ",".join(rear_flags)
                    )
                )

        if changed:
            if rear_flags and slowdown_commanded:
                self._last_safety_reason = "behavior_limit+rear_approach"
            elif rear_flags:
                self._last_safety_reason = "behavior_rear_approach"
            else:
                self._last_safety_reason = "behavior_limit"
            return limited, {
                "event_type": "CONTROL_ACTION_CHANGE",
                "trigger_reason": self.behavior_reason,
                "action_taken": action_taken,
                "stop_commanded": False,
                "slowdown_commanded": slowdown_commanded,
                "speed_limit_mps": speed_limit_mps,
                "summary_text": " ".join(summary_parts),
            }
        return cmd, None

    def _apply_local_hold(self, cmd, now):
        if not self._has_local_hold(now):
            return cmd, None
        if (
            self.allow_in_place_rotation_during_local_hold
            and self.avoidance_path_active
            and (
                self._should_preserve_in_place_rotation(cmd)
                or self._has_preferred_avoidance_turn()
            )
        ):
            rotate_cmd = self._preferred_rotation_only_cmd(cmd)
            self._last_safety_reason = "local_hold_rotate"
            rospy.loginfo_throttle(
                self.log_period_s,
                "teb_cmd_vel_relay: local hold keeps in-place rotation | raw_w=%.3f out_w=%.3f avoid_dir=%s",
                float(cmd.angular.z),
                float(rotate_cmd.angular.z),
                self._preferred_avoidance_turn_direction(),
            )
            return rotate_cmd, {
                "event_type": "CONTROL_ACTION_CHANGE",
                "trigger_reason": "local_replanner_hold",
                "action_taken": "hold_rotate",
                "avoid_direction": self._preferred_avoidance_turn_direction(),
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
        self._last_safety_reason = "local_hold_stop"
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
        near_goal_guard_zone = self._is_near_goal_side_turn_guard_stop_zone()
        ignore_stop = self._should_ignore_obstacle_for_local_goal(
            self.closest_stop_obstacle_x
        )
        ignore_slow = self._should_ignore_obstacle_for_local_goal(
            self.closest_slow_obstacle_x
        )
        if ignore_stop or ignore_slow:
            self._last_safety_reason = "ignore_obstacle_beyond_goal"
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
                self.local_path_remaining_m,
            )
        if (
            self.enable_emergency_stop
            and (not ignore_stop)
            and self.closest_stop_obstacle_x <= self.emergency_stop_distance
        ):
            if self._should_allow_avoidance_close_crawl(cmd, final_segment_active):
                crawl_cmd = self._avoidance_close_crawl_cmd(cmd)
                rospy.loginfo_throttle(
                    self.log_period_s,
                    "teb_cmd_vel_relay: keeping avoidance crawl | obstacle=(%.2f,%.2f) m center_box=%.2f/%.2f cmd(v=%.3f,w=%.3f) -> crawl(v=%.3f,w=%.3f) dir=%s",
                    self.closest_stop_obstacle_x,
                    self.closest_stop_obstacle_y,
                    self.avoidance_center_stop_distance,
                    self.avoidance_center_lateral_y,
                    float(cmd.linear.x),
                    float(cmd.angular.z),
                    float(crawl_cmd.linear.x),
                    float(crawl_cmd.angular.z),
                    self._preferred_avoidance_turn_direction(),
                )
                self._last_safety_reason = "obstacle_avoidance_crawl"
                return crawl_cmd, {
                    "event_type": "CONTROL_ACTION_CHANGE",
                    "trigger_reason": "front_obstacle_avoidance_crawl",
                    "action_taken": "avoidance_crawl",
                    "avoid_direction": self._preferred_avoidance_turn_direction(),
                    "stop_commanded": False,
                    "slowdown_commanded": True,
                    "speed_limit_mps": float(crawl_cmd.linear.x),
                    "closest_obstacle_dist_m": float(self.closest_stop_obstacle_x),
                    "obstacle_lateral_offset_m": float(self.closest_stop_obstacle_y),
                    "summary_text": "TEB relay kept a slow forward crawl because an active avoidance path was turning away from a close obstacle that was outside the center-front stop box.",
                }
            if (
                self.allow_in_place_rotation_near_obstacle
                and (not near_goal_guard_zone)
                and (not final_segment_active)
                and (
                    self._should_preserve_in_place_rotation(cmd)
                    or self._has_preferred_avoidance_turn()
                )
            ):
                rotate_only = self._preferred_rotation_only_cmd(cmd)
                stop_lateral_limit = self._forward_zone_lateral_limit(
                    self.closest_stop_obstacle_x,
                    self.emergency_stop_lateral_y,
                    self.emergency_stop_far_lateral_ratio,
                    self.emergency_stop_distance,
                )
                preferred_dir = self._preferred_avoidance_turn_direction()
                rospy.loginfo_throttle(
                    self.log_period_s,
                    "teb_cmd_vel_relay: emergency obstacle -> rotate-only | obstacle=(%.2f,%.2f) m limit_y=%.2f cmd_w=%.3f out_w=%.3f avoid_dir=%s",
                    self.closest_stop_obstacle_x,
                    self.closest_stop_obstacle_y,
                    stop_lateral_limit,
                    float(cmd.angular.z),
                    float(rotate_only.angular.z),
                    preferred_dir,
                )
                self._last_safety_reason = "obstacle_emergency_rotate"
                return rotate_only, {
                    "event_type": "CONTROL_ACTION_CHANGE",
                    "trigger_reason": "front_obstacle_emergency",
                    "action_taken": "emergency_rotate",
                    "avoid_direction": preferred_dir,
                    "stop_commanded": False,
                    "slowdown_commanded": True,
                    "speed_limit_mps": 0.0,
                    "closest_obstacle_dist_m": float(self.closest_stop_obstacle_x),
                    "obstacle_lateral_offset_m": float(self.closest_stop_obstacle_y),
                    "summary_text": "TEB relay suppressed forward motion and kept an in-place rotation because a close obstacle was detected ahead during a turning maneuver.",
                }
            stopped = Twist()
            stop_lateral_limit = self._forward_zone_lateral_limit(
                self.closest_stop_obstacle_x,
                self.emergency_stop_lateral_y,
                self.emergency_stop_far_lateral_ratio,
                self.emergency_stop_distance,
            )
            rospy.logwarn_throttle(
                self.log_period_s,
                "teb_cmd_vel_relay: emergency stop | obstacle=(%.2f,%.2f) m limit_y=%.2f cmd_v=%.3f cmd_w=%.3f",
                self.closest_stop_obstacle_x,
                self.closest_stop_obstacle_y,
                stop_lateral_limit,
                float(cmd.linear.x),
                float(cmd.angular.z),
            )
            self._last_safety_reason = "obstacle_emergency_stop"
            return stopped, {
                "event_type": "CONTROL_ACTION_CHANGE",
                "trigger_reason": "front_obstacle_emergency",
                "action_taken": "emergency_stop",
                "avoid_direction": self._preferred_avoidance_turn_direction(),
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
                    and (not near_goal_guard_zone)
                    and (not final_segment_active)
                    and (
                        self._should_preserve_in_place_rotation(cmd)
                        or self._has_preferred_avoidance_turn()
                    )
                ):
                    rotate_only = self._preferred_rotation_only_cmd(cmd)
                    rospy.loginfo_throttle(
                        self.log_period_s,
                        "teb_cmd_vel_relay: converting obstacle slowdown to in-place rotation v=%.3f w=%.3f -> %.3f obstacle_x=%.2f dir=%s",
                        slowed.linear.x,
                        cmd.angular.z,
                        rotate_only.angular.z,
                        self.closest_slow_obstacle_x,
                        self._preferred_avoidance_turn_direction(),
                    )
                    slowed = rotate_only
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
            self._last_safety_reason = "obstacle_slowdown"
            return slowed, {
                "event_type": "CONTROL_ACTION_CHANGE",
                "trigger_reason": "front_obstacle_slowdown",
                "action_taken": "slowdown",
                "avoid_direction": self._preferred_avoidance_turn_direction(),
                "stop_commanded": False,
                "slowdown_commanded": True,
                "speed_limit_mps": float(slowed.linear.x),
                "closest_obstacle_dist_m": float(self.closest_slow_obstacle_x),
                "obstacle_lateral_offset_m": float(self.closest_slow_obstacle_y),
                "summary_text": "TEB relay slowed the robot because an obstacle was detected in the forward slowdown zone.",
            }

        return cmd, None

    def _apply_blind_zone_turn_guard(self, cmd, now):
        memory = self._get_active_blind_zone_memory(now)
        if memory is None:
            return cmd, None

        side = int(memory["side"])
        obstacle_x = float(memory["x"])
        obstacle_y = float(memory["y"])
        age_s = float(memory["age_s"])
        turn_away_direction = self._safe_turn_direction_from_side(side)

        if side == 0:
            if float(cmd.linear.x) <= 0.0:
                return cmd, None
            self._last_safety_reason = "blind_zone_center_stop"
            rospy.logwarn_throttle(
                self.log_period_s,
                "teb_cmd_vel_relay: blind-zone front hold | obstacle=(%.2f,%.2f) age=%.2fs cmd(v=%.3f,w=%.3f)",
                obstacle_x,
                obstacle_y,
                age_s,
                float(cmd.linear.x),
                float(cmd.angular.z),
            )
            return Twist(), {
                "event_type": "CONTROL_ACTION_CHANGE",
                "trigger_reason": "blind_zone_obstacle_memory",
                "action_taken": "blind_zone_hold_stop",
                "avoid_direction": turn_away_direction,
                "stop_commanded": True,
                "slowdown_commanded": False,
                "speed_limit_mps": 0.0,
                "closest_obstacle_dist_m": float(memory["range_m"]),
                "obstacle_lateral_offset_m": obstacle_y,
                "summary_text": "TEB relay held the robot because a recently observed close obstacle is now inside the LiDAR blind zone ahead of the robot.",
            }

        if not self._is_turning_toward_side(cmd.angular.z, side):
            return cmd, None

        if self._is_near_goal_side_turn_guard_stop_zone():
            self._last_safety_reason = "blind_zone_turn_goal_stop"
            rospy.logwarn_throttle(
                self.log_period_s,
                "teb_cmd_vel_relay: near-goal blind-zone stop | obstacle=(%.2f,%.2f) age=%.2fs goal_dist=%.2f cmd(v=%.3f,w=%.3f)",
                obstacle_x,
                obstacle_y,
                age_s,
                self._final_goal_distance_m(),
                float(cmd.linear.x),
                float(cmd.angular.z),
            )
            return Twist(), {
                "event_type": "CONTROL_ACTION_CHANGE",
                "trigger_reason": "blind_zone_obstacle_memory",
                "action_taken": "blind_zone_turn_goal_stop",
                "avoid_direction": turn_away_direction,
                "stop_commanded": True,
                "slowdown_commanded": False,
                "speed_limit_mps": 0.0,
                "closest_obstacle_dist_m": float(memory["range_m"]),
                "obstacle_lateral_offset_m": obstacle_y,
                "summary_text": "TEB relay held the robot stopped near the final goal instead of rotating or drifting away because a side obstacle remained inside the blind zone.",
            }

        # When the local replanner has temporarily dropped the path during an
        # active avoidance branch, do not convert that hold window into a small
        # forward crawl just because the blind-zone turn guard vetoed rotation.
        if self.local_path_empty and self.avoidance_path_active:
            self._last_safety_reason = "blind_zone_turn_hold"
            rospy.logwarn_throttle(
                self.log_period_s,
                "teb_cmd_vel_relay: blind-zone turn hold | obstacle=(%.2f,%.2f) age=%.2fs local_empty=yes avoid=active cmd(v=%.3f,w=%.3f)",
                obstacle_x,
                obstacle_y,
                age_s,
                float(cmd.linear.x),
                float(cmd.angular.z),
            )
            return Twist(), {
                "event_type": "CONTROL_ACTION_CHANGE",
                "trigger_reason": "blind_zone_obstacle_memory",
                "action_taken": "blind_zone_turn_hold_stop",
                "avoid_direction": turn_away_direction,
                "stop_commanded": True,
                "slowdown_commanded": False,
                "speed_limit_mps": 0.0,
                "closest_obstacle_dist_m": float(memory["range_m"]),
                "obstacle_lateral_offset_m": obstacle_y,
                "summary_text": "TEB relay held the robot stopped because a side obstacle remained inside the blind zone while the avoidance branch had no fresh local path to follow.",
            }

        if float(cmd.linear.x) <= 0.0:
            rotate_away = self._rotation_away_from_side_cmd(cmd, side)
            self._last_safety_reason = "blind_zone_turn_rotate_away"
            rospy.loginfo_throttle(
                self.log_period_s,
                "teb_cmd_vel_relay: blind-zone turn redirect | obstacle=(%.2f,%.2f) age=%.2fs cmd(v=%.3f,w=%.3f) -> out(v=%.3f,w=%.3f)",
                obstacle_x,
                obstacle_y,
                age_s,
                float(cmd.linear.x),
                float(cmd.angular.z),
                float(rotate_away.linear.x),
                float(rotate_away.angular.z),
            )
            return rotate_away, {
                "event_type": "CONTROL_ACTION_CHANGE",
                "trigger_reason": "blind_zone_obstacle_memory",
                "action_taken": "blind_zone_turn_rotate_away",
                "avoid_direction": turn_away_direction,
                "stop_commanded": False,
                "slowdown_commanded": True,
                "speed_limit_mps": 0.0,
                "closest_obstacle_dist_m": float(memory["range_m"]),
                "obstacle_lateral_offset_m": obstacle_y,
                "summary_text": "TEB relay redirected an in-place turn away from a recently observed side obstacle instead of vetoing the escape rotation.",
            }

        guarded = self._copy_cmd(cmd)
        guarded.angular.z = 0.0
        guarded.angular.x = 0.0
        guarded.angular.y = 0.0

        slowed = False
        if (
            self.blind_zone_turn_guard_linear_cap_mps > 0.0
            and float(guarded.linear.x) > self.blind_zone_turn_guard_linear_cap_mps
        ):
            guarded.linear.x = self.blind_zone_turn_guard_linear_cap_mps
            slowed = True

        crawl_applied = False
        crawl_speed = self._near_goal_blind_zone_crawl_speed(now)
        if crawl_speed > 0.0 and float(guarded.linear.x) < crawl_speed:
            guarded.linear.x = crawl_speed
            crawl_applied = True

        self._last_safety_reason = "blind_zone_turn_veto"
        rospy.loginfo_throttle(
            self.log_period_s,
            "teb_cmd_vel_relay: blind-zone turn veto | obstacle=(%.2f,%.2f) age=%.2fs cmd(v=%.3f,w=%.3f) -> out(v=%.3f,w=%.3f)",
            obstacle_x,
            obstacle_y,
            age_s,
            float(cmd.linear.x),
            float(cmd.angular.z),
            float(guarded.linear.x),
            float(guarded.angular.z),
        )
        return guarded, {
            "event_type": "CONTROL_ACTION_CHANGE",
            "trigger_reason": "blind_zone_obstacle_memory",
            "action_taken": (
                "blind_zone_turn_veto_crawl" if crawl_applied else "blind_zone_turn_veto"
            ),
            "avoid_direction": turn_away_direction,
            "stop_commanded": False,
            "slowdown_commanded": slowed or crawl_applied,
            "speed_limit_mps": float(guarded.linear.x),
            "closest_obstacle_dist_m": float(memory["range_m"]),
            "obstacle_lateral_offset_m": obstacle_y,
            "summary_text": (
                "TEB relay suppressed a turn toward a recently observed side obstacle and kept a small near-goal forward crawl so the robot can clear the LiDAR blind zone without stop-go."
                if crawl_applied
                else "TEB relay suppressed a turn toward a recently observed side obstacle and temporarily capped forward speed so the robot can clear the LiDAR blind zone without freezing."
            ),
        }

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
            self._last_sanitize_reason = "stale_cmd"
            self._last_safety_reason = "stale_cmd"
            self._last_smoothing_reason = "idle"
            self._last_target_cmd = Twist()
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
                "relay reason=stale_cmd_timeout action=idle_stop sanitize=stale_cmd safety=stale_cmd smooth=idle raw(v=0.000,w=0.000) sanitized(v=0.000,w=0.000) target(v=0.000,w=0.000) out(v=0.000,w=0.000) "
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

        self._last_safety_reason = "clear"
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

        if not (explain and explain.get("stop_commanded", False)):
            cmd, info = self._apply_blind_zone_turn_guard(cmd, now)
            if info is not None:
                explain = info

        self._last_target_cmd = self._copy_cmd(cmd)
        cmd = self._apply_cmd_smoothing(
            cmd,
            now,
            bypass=bool(explain and explain.get("stop_commanded", False))
            or bool(explain and explain.get("action_taken", "") == "blind_zone_turn_veto")
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
                avoid_direction=explain.get("avoid_direction", "none"),
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
