#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DWA local planner — stable path tracking for A* overlap/crossing segments

Fixes vs. original:
  • Ignore /astar/path if it hasn't actually changed (hash signature).
  • Track along-path progress with arc-length s (monotonic, jitter tolerant).
  • Segment-following target: stay on the selected active path and advance
    vertex-by-vertex instead of shortcutting across the polyline.
  • Progress cost to discourage backwards/sideways motions near junctions.
  • Rotate-only mode with clean hysteresis and path-tangent gating.
  • Minimal forward obstacle stop using front ROI from PointCloud2.
  • Snap-to-path when lateral error is large, then follow lookahead once on track.
"""

import math
import time
import numpy as np
import rospy
import tf2_ros
import tf.transformations as transformations

from geometry_msgs.msg import Twist, Point, PoseStamped
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool, Float32MultiArray, Header, String

from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

from dynamic_window_approach.msg import BehaviorCommand, server_to_robot

# ------------------------------------- utils -------------------------------------
def angdiff(a, b):
    """Wrap-safe angle diff a-b in [-pi, pi]."""
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


def wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))

# ----------------------------------- DWA node -----------------------------------
class DWAControl:
    def __init__(self):
        # ===== Dynamics / Sampling =====
        self.max_speed = 1.20
        self.min_speed = float(rospy.get_param("~min_speed", 0.0))
        self.low_speed = 0.35
        self.max_yaw_rate = math.radians(180.0)
        self.max_accel = float(rospy.get_param("~max_accel", 0.45))
        self.max_delta_yaw_rate = math.radians(450.0)
        self.v_resolution = 0.00125
        self.yaw_rate_resolution = math.radians(5.0)
        self.dt = 0.1
        self.predict_time = 3.0

        # ===== Costs =====
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = rospy.get_param("~speed_cost_gain", 1.8)
        self.progress_cost_gain = rospy.get_param("~progress_cost_gain", 2.0)
        self.lateral_cost_gain  = rospy.get_param("~lateral_cost_gain", 6.0)
        self.target_cost_gain   = rospy.get_param("~target_cost_gain", 2.0)
        self.progress_forward_gain = rospy.get_param("~progress_forward_gain", 0.5)
        self.obstacle_cost_gain = rospy.get_param("~obstacle_cost_gain", 2.0)
        self.robot_stuck_flag_cons = 0.001

        self.server_cmd_drive_mode = 0

        # ===== Inputs =====
        self.pose_topic = rospy.get_param("~pose_topic", "/lio_sam/mapping/odometry")
        self.global_path_topic = rospy.get_param("~global_path_topic", "/astar/path")
        self.local_path_topic = rospy.get_param("~local_path_topic", "/planning/local_path")
        self.path_mode_topic = rospy.get_param("~path_mode_topic", "/planning/path_mode")
        self.follow_global_path_only = bool(
            rospy.get_param("~follow_global_path_only", False)
        )
        self.local_path_timeout_s = float(rospy.get_param("~local_path_timeout_s", 4.0))
        self.local_path_source_timeout_s = max(
            0.05,
            float(
                rospy.get_param(
                    "~local_path_source_timeout_s",
                    self.local_path_timeout_s,
                )
            ),
        )
        self.enforce_local_path_source_stamp = bool(
            rospy.get_param("~enforce_local_path_source_stamp", False)
        )
        self.local_path_signature_start_resolution_m = max(
            0.01,
            float(
                rospy.get_param(
                    "~local_path_signature_start_resolution_m",
                    0.10,
                )
            ),
        )
        self.behavior_cmd_topic = rospy.get_param("~behavior_cmd_topic", "/planning/behavior_cmd")
        self.drivable_grid_topic = rospy.get_param("~drivable_grid_topic", "/lio_sam/drivable_area/grid")
        self.use_drivable_grid = bool(rospy.get_param("~use_drivable_grid", True))
        self.grid_unknown_is_occupied = bool(rospy.get_param("~grid_unknown_is_occupied", True))
        self.dynamic_risk_grid_topic = rospy.get_param("~dynamic_risk_grid_topic", "/planning/dynamic_risk_grid")
        self.use_dynamic_risk_grid = bool(rospy.get_param("~use_dynamic_risk_grid", True))
        self.risk_unknown_is_occupied = bool(rospy.get_param("~risk_unknown_is_occupied", False))
        self.risk_occupied_threshold = int(rospy.get_param("~risk_occupied_threshold", 45))
        self.tracking_reference_topic = rospy.get_param(
            "~tracking_reference_topic",
            "/planning/tracking_reference_path",
        )
        self.launch_profile_label = str(rospy.get_param("~launch_profile_label", "unknown"))
        self.launch_real_mode = str(rospy.get_param("~launch_real_mode", "unknown"))
        self.launch_localization_environment = str(
            rospy.get_param("~launch_localization_environment", "unknown")
        )
        self.launch_map_profile_name = str(
            rospy.get_param("~launch_map_profile_name", "unknown")
        )
        self.launch_localizer_map_relative_path = str(
            rospy.get_param("~launch_localizer_map_relative_path", "unknown")
        )
        self.launch_runtime_drivable_state_file = str(
            rospy.get_param("~launch_runtime_drivable_state_file", "unknown")
        )
        self.pointcloud_static_blocking_enabled = bool(
            rospy.get_param("~pointcloud_static_blocking_enabled", False)
        )
        self.debug_stop_logging = bool(rospy.get_param("~debug_stop_logging", True))
        self.debug_dwa_stats = bool(rospy.get_param("~debug_dwa_stats", True))
        self.stop_log_period_s = max(0.2, float(rospy.get_param("~stop_log_period_s", 1.0)))

        # ===== Obstacle handling (avoid + emergency stop) =====
        self.cloud_topic = rospy.get_param("~pointcloud_topic", "/ouster/points")
        self.emergency_stop_distance = rospy.get_param(
            "~emergency_stop_distance",
            rospy.get_param("~stop_distance", 0.6),
        )
        self.avoidance_hard_stop_distance = max(
            0.05, float(rospy.get_param("~avoidance_hard_stop_distance", 0.30))
        )
        self.obstacle_influence_distance = rospy.get_param("~obstacle_influence_distance", 1.8)
        legacy_robot_radius = float(rospy.get_param("~robot_radius", 0.35))
        self.robot_width_m = max(
            0.05, float(rospy.get_param("~robot_width_m", 0.58))
        )
        self.robot_length_m = max(
            0.05, float(rospy.get_param("~robot_length_m", 0.612))
        )
        self.robot_half_width_m = 0.5 * self.robot_width_m
        self.robot_half_length_m = 0.5 * self.robot_length_m
        self.robot_radius = 0.5 * math.hypot(self.robot_length_m, self.robot_width_m)
        self.footprint_padding_m = max(0.0, float(rospy.get_param("~footprint_padding_m", 0.0)))
        self.safety_margin = rospy.get_param("~safety_margin", 0.12)
        self.use_pointcloud_obstacle_cost = bool(rospy.get_param("~use_pointcloud_obstacle_cost", False))
        self.obstacle_collision_radius = max(
            0.0, float(rospy.get_param("~obstacle_collision_radius", 0.0))
        )
        self.obstacle_consider_side_m = max(
            0.2, float(rospy.get_param("~obstacle_consider_side_m", 1.1))
        )
        self.obstacle_consider_back_m = float(rospy.get_param("~obstacle_consider_back_m", -0.15))
        self.obstacle_ignore_traj_steps = max(
            0, int(rospy.get_param("~obstacle_ignore_traj_steps", 3))
        )
        self.stop_width = rospy.get_param("~stop_width", self.robot_width_m)   # total width (|y|<=width/2)
        self.min_z = rospy.get_param("~min_z", -0.3)
        self.max_z = rospy.get_param("~max_z", 2.2)
        self.enable_self_filter = bool(rospy.get_param("~enable_self_filter", True))
        self.self_filter_margin_m = max(
            0.0, float(rospy.get_param("~self_filter_margin_m", 0.08))
        )
        self.self_filter_radius_x = max(
            0.0,
            float(
                rospy.get_param(
                    "~self_filter_radius_x",
                    self.robot_half_length_m + self.footprint_padding_m + self.self_filter_margin_m,
                )
            ),
        )
        self.self_filter_radius_y = max(
            0.0,
            float(
                rospy.get_param(
                    "~self_filter_radius_y",
                    self.robot_half_width_m + self.footprint_padding_m + self.self_filter_margin_m,
                )
            ),
        )
        self.cloud_downsample = rospy.get_param("~cloud_downsample", 4)
        self.traj_check_step = max(1, int(rospy.get_param("~traj_check_step", 2)))
        self.max_obstacle_points = max(20, int(rospy.get_param("~max_obstacle_points", 300)))
        self.emergency_bin_size_m = max(
            0.05, float(rospy.get_param("~emergency_bin_size_m", 0.10))
        )
        self.emergency_min_close_points = max(
            1, int(rospy.get_param("~emergency_min_close_points", 2))
        )
        self.emergency_immediate_contact_min_points = max(
            1,
            int(rospy.get_param("~emergency_immediate_contact_min_points", 1)),
        )
        self.emergency_passable_width_m = max(
            0.10,
            float(
                rospy.get_param(
                    "~emergency_passable_width_m",
                    self.robot_width_m + 2.0 * self.footprint_padding_m + 0.05,
                )
            ),
        )
        self.enable_emergency_stop = bool(
            rospy.get_param("~enable_emergency_stop", True)
        )
        self.block_on_count = rospy.get_param("~block_on_count", 2)
        self.block_off_count = rospy.get_param("~block_off_count", 3)
        self.emergency_blocked = False
        self._blk_on = 0
        self._blk_off = 0
        self.front_obstacle_clearance = float("inf")
        self._last_emergency_close_points = 0
        self._last_emergency_intrusion_points = 0
        self.use_global_obstacle_overlay_boxes_for_stop = bool(
            rospy.get_param("~use_global_obstacle_overlay_boxes_for_stop", True)
        )
        self.global_obstacle_overlay_boxes_topic = str(
            rospy.get_param(
                "~global_obstacle_overlay_boxes_topic",
                "/planning/global_obstacle_overlay_boxes",
            )
        ).strip()
        self.global_obstacle_overlay_stop_locked_only = bool(
            rospy.get_param("~global_obstacle_overlay_stop_locked_only", True)
        )
        self.global_obstacle_overlay_box_timeout_s = max(
            0.0,
            float(rospy.get_param("~global_obstacle_overlay_box_timeout_s", 1.5)),
        )
        self.global_obstacle_overlay_boxes = []
        self.global_obstacle_overlay_boxes_stamp = rospy.Time(0)
        self.overlay_box_front_clearance = float("inf")
        self.overlay_box_blocked = False
        self._overlay_box_match_count = 0
        self._raw_front_obstacle_clearance = float("inf")
        self._raw_emergency_band_blocked = False
        self._raw_immediate_contact_blocked = False
        self.near_field_raw_stop_enabled = bool(
            rospy.get_param("~near_field_raw_stop_enabled", False)
        )
        self.near_field_raw_stop_topic = str(
            rospy.get_param("~near_field_raw_stop_topic", "")
        ).strip()
        self.near_field_raw_stop_base_frame = str(
            rospy.get_param("~near_field_raw_stop_base_frame", "base_link")
        ).strip()
        self.near_field_raw_stop_tf_timeout_s = max(
            0.0, float(rospy.get_param("~near_field_raw_stop_tf_timeout_s", 0.02))
        )
        self.near_field_raw_stop_tf_fallback_latest = bool(
            rospy.get_param("~near_field_raw_stop_tf_fallback_latest", True)
        )
        self.near_field_raw_stop_allow_raw_frame_on_tf_failure = bool(
            rospy.get_param("~near_field_raw_stop_allow_raw_frame_on_tf_failure", False)
        )
        self.near_field_raw_stop_min_x_m = float(
            rospy.get_param("~near_field_raw_stop_min_x_m", 0.30)
        )
        self.near_field_raw_stop_max_x_m = max(
            self.near_field_raw_stop_min_x_m + 0.05,
            float(rospy.get_param("~near_field_raw_stop_max_x_m", 1.20)),
        )
        self.near_field_raw_stop_half_width_m = max(
            0.05, float(rospy.get_param("~near_field_raw_stop_half_width_m", 0.45))
        )
        self.near_field_raw_stop_min_z_m = float(
            rospy.get_param("~near_field_raw_stop_min_z_m", -0.20)
        )
        self.near_field_raw_stop_max_z_m = max(
            self.near_field_raw_stop_min_z_m + 0.05,
            float(rospy.get_param("~near_field_raw_stop_max_z_m", 1.20)),
        )
        self.near_field_raw_stop_min_points = max(
            1, int(rospy.get_param("~near_field_raw_stop_min_points", 3))
        )
        self.near_field_raw_stop_cell_size_m = max(
            0.03, float(rospy.get_param("~near_field_raw_stop_cell_size_m", 0.12))
        )
        self.near_field_raw_stop_min_cells = max(
            1, int(rospy.get_param("~near_field_raw_stop_min_cells", 1))
        )
        self.near_field_raw_stop_close_override_distance_m = max(
            0.0,
            float(
                rospy.get_param(
                    "~near_field_raw_stop_close_override_distance_m",
                    min(self.emergency_stop_distance, max(0.10, self.avoidance_hard_stop_distance)),
                )
            ),
        )
        self.near_field_raw_stop_close_override_half_width_m = max(
            0.05,
            min(
                self.near_field_raw_stop_half_width_m,
                float(
                    rospy.get_param(
                        "~near_field_raw_stop_close_override_half_width_m",
                        min(
                            self.near_field_raw_stop_half_width_m,
                            max(
                                0.10,
                                self.robot_half_width_m - 0.12,
                            ),
                        ),
                    )
                ),
            ),
        )
        self.near_field_raw_stop_close_override_min_points = max(
            1, int(rospy.get_param("~near_field_raw_stop_close_override_min_points", 3))
        )
        self.near_field_raw_stop_close_override_min_cells = max(
            1, int(rospy.get_param("~near_field_raw_stop_close_override_min_cells", 2))
        )
        self.near_field_raw_stop_downsample = max(
            1, int(rospy.get_param("~near_field_raw_stop_downsample", 1))
        )
        self.near_field_raw_stop_max_update_hz = max(
            0.0, float(rospy.get_param("~near_field_raw_stop_max_update_hz", 0.0))
        )
        self.near_field_raw_stop_on_count = max(
            1, int(rospy.get_param("~near_field_raw_stop_on_count", 1))
        )
        self.near_field_raw_stop_off_count = max(
            1, int(rospy.get_param("~near_field_raw_stop_off_count", 3))
        )
        self.near_field_raw_stop_timeout_s = max(
            0.05, float(rospy.get_param("~near_field_raw_stop_timeout_s", 0.35))
        )
        self.near_field_raw_stop_hold_s = max(
            0.0, float(rospy.get_param("~near_field_raw_stop_hold_s", 0.45))
        )
        self.near_field_raw_stop_marker_topic = str(
            rospy.get_param(
                "~near_field_raw_stop_marker_topic",
                "/planning/near_field_stop_marker",
            )
        ).strip()
        self.near_field_raw_stop_hit_cloud_topic = str(
            rospy.get_param(
                "~near_field_raw_stop_hit_cloud_topic",
                "/planning/near_field_stop_hits",
            )
        ).strip()
        self.near_field_raw_stop_log_period_s = max(
            0.0, float(rospy.get_param("~near_field_raw_stop_log_period_s", 0.5))
        )
        self.near_field_raw_stop_publish_max_points = max(
            0, int(rospy.get_param("~near_field_raw_stop_publish_max_points", 600))
        )
        self.emergency_bypass_enabled = bool(
            rospy.get_param("~emergency_bypass_enabled", True)
        )
        self.emergency_bypass_preview_m = max(
            0.10, float(rospy.get_param("~emergency_bypass_preview_m", 0.90))
        )
        self.emergency_bypass_target_lateral_m = max(
            0.02, float(rospy.get_param("~emergency_bypass_target_lateral_m", 0.10))
        )
        self.emergency_bypass_bearing_rad = math.radians(
            max(1.0, float(rospy.get_param("~emergency_bypass_bearing_deg", 8.0)))
        )
        self.emergency_bypass_yaw_err_rad = math.radians(
            max(1.0, float(rospy.get_param("~emergency_bypass_yaw_err_deg", 10.0)))
        )
        self.emergency_bypass_clearance_margin_m = max(
            0.0,
            float(rospy.get_param("~emergency_bypass_clearance_margin_m", 0.03)),
        )
        self.active_avoidance_bypass_clearance_m = max(
            0.05,
            min(
                self.avoidance_hard_stop_distance,
                float(
                    rospy.get_param(
                        "~active_avoidance_bypass_clearance_m",
                        max(0.10, self.avoidance_hard_stop_distance - 0.10),
                    )
                ),
            ),
        )
        self.emergency_bypass_goal_window_m = max(
            0.0,
            float(rospy.get_param("~emergency_bypass_goal_window_m", 0.60)),
        )
        self.emergency_bypass_speed_limit_mps = max(
            0.05,
            float(rospy.get_param("~emergency_bypass_speed_limit_mps", 0.18)),
        )
        self.enable_reverse_recovery = bool(
            rospy.get_param("~enable_reverse_recovery", False)
        )
        self.reverse_recovery_hold_trigger_s = max(
            0.1, float(rospy.get_param("~reverse_recovery_hold_trigger_s", 1.2))
        )
        self.reverse_recovery_emergency_trigger_s = max(
            0.0,
            float(rospy.get_param("~reverse_recovery_emergency_trigger_s", 0.35)),
        )
        self.reverse_recovery_distance_m = max(
            0.05, float(rospy.get_param("~reverse_recovery_distance_m", 0.40))
        )
        self.reverse_recovery_speed_mps = max(
            0.02, float(rospy.get_param("~reverse_recovery_speed_mps", 0.16))
        )
        self.reverse_recovery_clearance_margin_m = max(
            0.0,
            float(rospy.get_param("~reverse_recovery_clearance_margin_m", 0.08)),
        )
        self.reverse_recovery_max_distance_m = max(
            self.reverse_recovery_distance_m,
            float(rospy.get_param("~reverse_recovery_max_distance_m", 1.00)),
        )
        self.reverse_recovery_min_active_time_s = max(
            0.0,
            float(rospy.get_param("~reverse_recovery_min_active_time_s", 0.70)),
        )
        self.reverse_recovery_min_distance_before_resume_m = max(
            0.0,
            float(
                rospy.get_param(
                    "~reverse_recovery_min_distance_before_resume_m", 0.08
                )
            ),
        )
        default_reverse_timeout_s = max(
            2.0, (self.reverse_recovery_distance_m / self.reverse_recovery_speed_mps) * 2.5
        )
        self.reverse_recovery_timeout_s = max(
            0.5,
            float(
                rospy.get_param(
                    "~reverse_recovery_timeout_s", default_reverse_timeout_s
                )
            ),
        )
        self.reverse_recovery_pause_s = max(
            0.0, float(rospy.get_param("~reverse_recovery_pause_s", 0.45))
        )
        self.reverse_recovery_cooldown_s = max(
            0.0, float(rospy.get_param("~reverse_recovery_cooldown_s", 4.0))
        )
        self.reverse_recovery_rear_check_distance_m = max(
            0.10,
            float(rospy.get_param("~reverse_recovery_rear_check_distance_m", 0.60)),
        )
        self.reverse_recovery_rear_half_width_m = max(
            0.10,
            float(rospy.get_param("~reverse_recovery_rear_half_width_m", 0.42)),
        )
        self.reverse_recovery_rear_min_points = max(
            1, int(rospy.get_param("~reverse_recovery_rear_min_points", 2))
        )
        self.reverse_recovery_rear_drivable_margin_m = max(
            0.0,
            float(rospy.get_param("~reverse_recovery_rear_drivable_margin_m", 0.05)),
        )
        # When near-field raw stop is enabled, a second full /ouster/points pass
        # inside this node can delay the critical stop callback under load.
        self.enable_raw_cloud_fallback_stop = bool(
            rospy.get_param(
                "~enable_raw_cloud_fallback_stop",
                not self.near_field_raw_stop_enabled,
            )
        )
        self.enable_raw_cloud_processing = bool(
            self.use_pointcloud_obstacle_cost
            or self.enable_raw_cloud_fallback_stop
            or self.enable_reverse_recovery
        )
        self._near_field_raw_stop_blocked = False
        self._near_field_raw_stop_on = 0
        self._near_field_raw_stop_off = 0
        self._near_field_raw_stop_last_stamp = rospy.Time(0)
        self._near_field_raw_stop_last_count = 0
        self._near_field_raw_stop_last_cells = 0
        self._near_field_raw_stop_last_min_x = float("inf")
        self._near_field_raw_stop_last_log = rospy.Time(0)
        self._near_field_raw_stop_tf_warn_sec = 0.0
        self._near_field_raw_stop_last_frame = ""
        self._near_field_raw_stop_tf_status = "not_started"
        self._near_field_raw_stop_last_marker_refresh = rospy.Time(0)
        self._near_field_raw_stop_hold_until = rospy.Time(0)
        self._near_field_raw_stop_last_reason = "clear"
        self._near_field_raw_stop_last_process_ms = 0.0
        self._near_field_raw_stop_last_process_wall_sec = 0.0
        self._last_emergency_source = "none"
        self._emergency_bypass_active = False
        self._emergency_bypass_debug = {}
        self.obstacle_local_points = np.empty((0, 2), dtype=np.float32)
        self._footprint_sample_cache = {}

        # ===== Drivable area grid =====
        self.grid_resolution = None
        self.grid_width = 0
        self.grid_height = 0
        self.grid_origin_x = 0.0
        self.grid_origin_y = 0.0
        self.grid_data = None
        self.risk_grid_resolution = None
        self.risk_grid_width = 0
        self.risk_grid_height = 0
        self.risk_grid_origin_x = 0.0
        self.risk_grid_origin_y = 0.0
        self.risk_grid_data = None

        # ===== Rotate-only mode =====
        self.rotate_only_deg = rospy.get_param("~rotate_only_deg", 150.0)
        self.rotate_exit_deg = rospy.get_param("~rotate_exit_deg", 40.0)
        self.rotate_kp = rospy.get_param("~rotate_kp", 2.0)
        self.rotate_w_max_deg = rospy.get_param("~rotate_w_max_deg", 28.0)
        self.rotate_ok_count = rospy.get_param("~rotate_ok_count", 3)
        self.rotate_max_spin_deg = rospy.get_param("~rotate_max_spin_deg", 420.0)
        self.rotate_max_time_s = rospy.get_param("~rotate_max_time_s", 6.0)
        self.rotate_reentry_cooldown_s = max(
            0.0, float(rospy.get_param("~rotate_reentry_cooldown_s", 4.0))
        )
        self.rotate_reentry_target_delta_deg = max(
            0.0, float(rospy.get_param("~rotate_reentry_target_delta_deg", 12.0))
        )
        self._ROT_HIGH = math.radians(self.rotate_only_deg)
        self._ROT_LOW  = math.radians(self.rotate_exit_deg)
        self._ROT_WMAX = math.radians(self.rotate_w_max_deg)
        self._ROT_RETARGET = math.radians(self.rotate_reentry_target_delta_deg)
        self._rot_mode = False
        self._rot_yaw_target = None
        self._rot_prev_yaw = None
        self._rot_accum = 0.0
        self._rot_ok = 0
        self._rot_start_time = None
        self._rot_cooldown_until = rospy.Time(0)
        self._rot_last_timeout_target = None

        # ===== Path tracking (s-based) =====
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 1.40)
        self.lookahead_speed_gain = max(
            0.0, float(rospy.get_param("~lookahead_speed_gain", 1.80))
        )
        self.lookahead_error_gain = max(
            0.0, float(rospy.get_param("~lookahead_error_gain", 0.80))
        )
        self.lookahead_max_distance = max(
            self.lookahead_distance,
            float(rospy.get_param("~lookahead_max_distance", 3.50)),
        )
        self.obstacle_response_clearance_m = max(
            self.emergency_stop_distance,
            float(rospy.get_param("~obstacle_response_clearance_m", 1.80)),
        )
        self.obstacle_response_lookahead_scale = max(
            0.45,
            min(
                1.60,
                float(rospy.get_param("~obstacle_response_lookahead_scale", 0.72)),
            ),
        )
        self.obstacle_response_tracking_kp_scale = max(
            1.0,
            float(rospy.get_param("~obstacle_response_tracking_kp_scale", 1.30)),
        )
        self.obstacle_response_yaw_rate_scale = max(
            1.0,
            float(rospy.get_param("~obstacle_response_yaw_rate_scale", 1.25)),
        )
        self.back_jitter_m = rospy.get_param("~back_jitter_m", 0.3)
        self.goal_thresh_m = rospy.get_param("~goal_thresh_m", 0.25)
        self.final_approach_window_m = rospy.get_param("~final_approach_window_m", 0.0)
        self.final_speed_k = rospy.get_param("~final_speed_k", 0.95)
        self.final_speed_min = rospy.get_param("~final_speed_min", 0.34)
        self.lat_goal_slop = rospy.get_param("~lat_goal_slop", 0.6)
        self.near_goal_no_rotate_m = rospy.get_param("~near_goal_no_rotate_m", 1.5)
        self.forward_motion_deadband = rospy.get_param("~forward_motion_deadband", 0.01)
        self.min_forward_cmd = rospy.get_param("~min_forward_cmd", 0.08)
        self.min_forward_cmd_distance = rospy.get_param("~min_forward_cmd_distance", 0.8)
        self.cruise_min_speed = rospy.get_param("~cruise_min_speed", 0.0)
        self.cruise_distance_m = rospy.get_param("~cruise_distance_m", 1.8)
        self.cruise_lat_err_m = rospy.get_param("~cruise_lat_err_m", 0.25)
        self.cruise_max_heading_err = math.radians(
            rospy.get_param("~cruise_max_heading_err_deg", 12.0)
        )
        self.cruise_max_yaw_rate = math.radians(rospy.get_param("~cruise_max_yaw_rate_deg", 45.0))
        self.current_point_search_radius_m = 5.0  # legacy (kept for /traj_info)
        # 경로에서 이 정도 이상 벗어나면 일단 경로로 붙는 스냅 단계
        self.snap_lat_err = rospy.get_param("~snap_lat_err", 0.25)
        self.snap_target_ahead_m = max(
            self.lookahead_distance,
            float(rospy.get_param("~snap_target_ahead_m", 2.20)),
        )
        self.tracking_projection_back_window_m = max(
            0.0, float(rospy.get_param("~tracking_projection_back_window_m", 0.35))
        )
        self.tracking_projection_forward_window_m = max(
            self.lookahead_distance,
            float(rospy.get_param("~tracking_projection_forward_window_m", 1.20))
        )
        self.tracking_path_smoothing_passes = max(
            0, int(rospy.get_param("~tracking_path_smoothing_passes", 2))
        )
        self.local_tracking_smoothing_enabled = bool(
            rospy.get_param("~local_tracking_smoothing_enabled", False)
        )
        self.local_tracking_preview_m = max(
            0.25, float(rospy.get_param("~local_tracking_preview_m", 0.85))
        )
        self.local_tracking_segment_step_scale = max(
            0.20, float(rospy.get_param("~local_tracking_segment_step_scale", 0.65))
        )
        self.local_tracking_target_step_cap_m = max(
            0.05, float(rospy.get_param("~local_tracking_target_step_cap_m", 0.22))
        )
        self.local_tracking_obstacle_speed_cap_mps = max(
            0.05,
            float(rospy.get_param("~local_tracking_obstacle_speed_cap_mps", 0.42)),
        )
        self.path_tracking_only = bool(rospy.get_param("~path_tracking_only", True))
        self.path_tracking_kp = float(rospy.get_param("~path_tracking_kp", 0.92))
        self.path_tracking_yaw_rate_max = math.radians(
            rospy.get_param("~path_tracking_yaw_rate_max_deg", 42.0)
        )
        self.path_tracking_in_place_yaw_rate_max = math.radians(
            rospy.get_param(
                "~path_tracking_in_place_yaw_rate_max_deg",
                26.0,
            )
        )
        self.path_tracking_yaw_accel_max = math.radians(
            rospy.get_param("~path_tracking_yaw_accel_max_deg", 120.0)
        )
        self.path_tracking_speed_cap = max(
            0.05, float(rospy.get_param("~path_tracking_speed_cap", 0.90))
        )
        self.path_tracking_steer_filter_gain = min(
            1.0, max(0.05, float(rospy.get_param("~path_tracking_steer_filter_gain", 0.55)))
        )
        self.cmd_smoothing_enabled = bool(rospy.get_param("~cmd_smoothing_enabled", True))
        self.cmd_linear_accel_max = max(
            0.05, float(rospy.get_param("~cmd_linear_accel_max_mps2", 0.68))
        )
        self.cmd_linear_decel_max = max(
            self.cmd_linear_accel_max,
            float(rospy.get_param("~cmd_linear_decel_max_mps2", 0.90)),
        )
        self.cmd_angular_accel_max = math.radians(
            max(10.0, float(rospy.get_param("~cmd_angular_accel_max_degps2", 85.0)))
        )
        self.cmd_angular_decel_max = math.radians(
            max(
                math.degrees(self.cmd_angular_accel_max),
                float(rospy.get_param("~cmd_angular_decel_max_degps2", 120.0)),
            )
        )
        self.cmd_smoothing_zero_snap = max(
            0.0, float(rospy.get_param("~cmd_smoothing_zero_snap", 0.015))
        )
        self.path_tracking_slowdown_yaw = math.radians(
            rospy.get_param("~path_tracking_slowdown_yaw_deg", 65.0)
        )
        self.path_tracking_stop_yaw = math.radians(
            rospy.get_param("~path_tracking_stop_yaw_deg", 95.0)
        )
        self.path_tracking_cte_gain = float(rospy.get_param("~path_tracking_cte_gain", 0.38))
        self.path_tracking_cte_soft_mps = max(
            0.05, float(rospy.get_param("~path_tracking_cte_soft_mps", 0.55))
        )
        self.path_tracking_cte_deadband_m = max(
            0.0, float(rospy.get_param("~path_tracking_cte_deadband_m", 0.10))
        )
        self.path_tracking_cte_filter_gain = min(
            1.0,
            max(0.05, float(rospy.get_param("~path_tracking_cte_filter_gain", 0.25))),
        )
        self.path_tracking_cte_yaw_cap = math.radians(
            rospy.get_param("~path_tracking_cte_yaw_cap_deg", 15.0)
        )
        self.path_tracking_heading_filter_gain = min(
            1.0,
            max(0.05, float(rospy.get_param("~path_tracking_heading_filter_gain", 0.35))),
        )
        self.local_tracking_heading_filter_gain = min(
            1.0,
            max(
                0.05,
                float(
                    rospy.get_param(
                        "~local_tracking_heading_filter_gain",
                        1.0,
                    )
                ),
            ),
        )
        self.local_tracking_heading_filter_reset = math.radians(
            max(
                0.0,
                float(
                    rospy.get_param(
                        "~local_tracking_heading_filter_reset_deg",
                        35.0,
                    )
                ),
            )
        )
        self.path_tracking_goal_bearing_gain = min(
            1.0,
            max(0.0, float(rospy.get_param("~path_tracking_goal_bearing_gain", 0.50))),
        )
        self.path_tracking_goal_bearing_cap = math.radians(
            rospy.get_param("~path_tracking_goal_bearing_cap_deg", 35.0)
        )
        self.path_tracking_target_step_m = max(
            0.05, float(rospy.get_param("~path_tracking_target_step_m", 0.25))
        )
        self.path_tracking_tangent_window_m = max(
            0.0,
            float(
                rospy.get_param(
                    "~path_tracking_tangent_window_m",
                    0.75,
                )
            ),
        )
        self.local_tracking_tangent_window_m = max(
            0.0,
            min(
                self.path_tracking_tangent_window_m,
                float(
                    rospy.get_param(
                        "~local_tracking_tangent_window_m",
                        min(self.path_tracking_tangent_window_m, 0.20),
                    )
                ),
            ),
        )
        self.path_tracking_crawl_speed = max(
            0.0, float(rospy.get_param("~path_tracking_crawl_speed", 0.12))
        )
        self.path_tracking_large_yaw_crawl_speed = max(
            0.0,
            float(
                rospy.get_param(
                    "~path_tracking_large_yaw_crawl_speed",
                    0.10,
                )
            ),
        )
        self.path_tracking_stop_distance_m = max(
            0.0, float(rospy.get_param("~path_tracking_stop_distance_m", 0.35))
        )
        self.local_stop_turn_go_enabled = bool(
            rospy.get_param("~local_stop_turn_go_enabled", True)
        )
        self.local_stop_turn_speed_cap = max(
            0.05,
            float(rospy.get_param("~local_stop_turn_speed_cap_mps", 0.26)),
        )
        self.local_stop_turn_align_rad = math.radians(
            float(rospy.get_param("~local_stop_turn_align_deg", 6.0))
        )
        self.local_stop_turn_corner_trigger_rad = math.radians(
            float(rospy.get_param("~local_stop_turn_corner_trigger_deg", 45.0))
        )
        self.local_stop_turn_corner_arrival_m = max(
            0.05,
            float(rospy.get_param("~local_stop_turn_corner_arrival_m", 0.22)),
        )
        self.path_tracking_goal_align_window_m = max(
            self.goal_thresh_m,
            float(
                rospy.get_param(
                    "~path_tracking_goal_align_window_m",
                    max(self.path_tracking_stop_distance_m, self.final_approach_window_m),
                )
            ),
        )
        self.path_tracking_goal_cte_scale_min = min(
            1.0,
            max(
                0.0,
                float(rospy.get_param("~path_tracking_goal_cte_scale_min", 0.35)),
            ),
        )
        self.path_tracking_goal_yaw_rate_max = min(
            self.path_tracking_yaw_rate_max,
            max(
                math.radians(5.0),
                math.radians(
                    rospy.get_param(
                        "~path_tracking_goal_yaw_rate_max_deg",
                        18.0,
                    )
                ),
            ),
        )
        self.local_tracking_stop_turn_only_yaw = min(
            math.pi,
            max(
                self.path_tracking_stop_yaw,
                math.radians(
                    float(
                        rospy.get_param(
                            "~local_tracking_stop_turn_only_yaw_deg",
                            120.0,
                        )
                    )
                ),
            ),
        )
        self.path_tracking_drivable_ignore_start_distance_m = max(
            0.0,
            float(
                rospy.get_param(
                    "~path_tracking_drivable_ignore_start_distance_m",
                    0.90,
                )
            ),
        )
        self.path_tracking_recovery_ignore_start_distance_m = max(
            self.path_tracking_drivable_ignore_start_distance_m,
            float(
                rospy.get_param(
                    "~path_tracking_recovery_ignore_start_distance_m",
                    1.80,
                )
            ),
        )
        self.path_tracking_drivable_boundary_tolerance_m = max(
            0.0,
            float(
                rospy.get_param(
                    "~path_tracking_drivable_boundary_tolerance_m",
                    0.12,
                )
            ),
        )
        self.path_tracking_enforce_drivable_grid = bool(
            rospy.get_param("~path_tracking_enforce_drivable_grid", False)
        )
        self.path_tracking_reset_goal_delta_m = max(
            0.0, float(rospy.get_param("~path_tracking_reset_goal_delta_m", 0.80))
        )
        self.path_tracking_minor_replan_delta_m = max(
            0.0, float(rospy.get_param("~path_tracking_minor_replan_delta_m", 0.12))
        )
        self.goal_completion_stuck_distance_m = max(
            self.goal_thresh_m,
            float(
                rospy.get_param(
                    "~goal_completion_stuck_distance_m",
                    max(
                        self.path_tracking_stop_distance_m,
                        self.avoidance_hard_stop_distance,
                        0.95,
                    ),
                )
            ),
        )
        self.goal_completion_stuck_arc_m = max(
            self.goal_thresh_m,
            float(
                rospy.get_param(
                    "~goal_completion_stuck_arc_m",
                    max(
                        self.path_tracking_stop_distance_m,
                        0.55,
                    ),
                )
            ),
        )
        self.goal_completion_stuck_hold_s = max(
            0.0,
            float(rospy.get_param("~goal_completion_stuck_hold_s", 0.6)),
        )
        self._path_tracking_prev_w = 0.0
        self._path_tracking_prev_desired_yaw = None
        self._path_tracking_filtered_lat_err = 0.0
        self._last_tracking_debug = {}
        self._last_target_debug = {}
        self._local_turn_rotate_kind = None
        self._local_turn_rotate_seg_idx = -1
        self._local_turn_rotate_heading = None
        self._local_turn_rotate_advance_after = False

        # Internal path buffers
        self.global_path_msg = None
        self.global_path_sig = None
        self.global_goal_xy = None
        self.local_path_msg = None
        self.local_path_sig = None
        self.local_path_stamp = rospy.Time(0)
        self.local_path_source_stamp = rospy.Time(0)
        self.active_path_source = "none"
        self.path_msg = None
        self.path_sig = None
        self.path_pts = []          # [(x,y), ...]
        self.seg_lens = []          # [len_i]
        self.cum_len = [0.0]        # [0, s1, s2, ...]
        self.s_total = 0.0
        self.s_cur = 0.0
        self.s_prev_published_idx = 0

        self.reach_goal_flag = False
        self.prev_goal_flag = False
        self._last_nav_reason = None
        self._last_nav_log_sec = 0.0
        self._no_valid_traj_since_sec = 0.0
        self._no_valid_traj_dist_m = float("inf")
        self._no_valid_traj_arc_m = float("inf")
        self._no_valid_traj_lat_m = float("inf")
        self._last_eval_stats = {
            "sampled": 0,
            "skip_spin": 0,
            "skip_grid": 0,
            "collision": 0,
            "valid": 0,
        }

        # ===== State =====
        self.current_pose = Odometry()
        self.warm_up_flag = False
        self.behavior_stop = False
        self.behavior_speed_limit = self.max_speed
        self.behavior_reason = "clear"
        self.current_path_mode = (
            "follow_global" if self.follow_global_path_only else "follow_local"
        )
        self._hold_mode_enter_sec = 0.0
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.emergency_stop_topic = rospy.get_param("~emergency_stop_topic", "/planning/emergency_stop")
        self.cmd_publish_hz = max(5.0, float(rospy.get_param("~cmd_publish_hz", 20.0)))
        self.last_cmd = Twist()
        self._last_cmd_smooth_stamp = rospy.Time(0)
        self._reverse_recovery_active = False
        self._reverse_recovery_start_time = rospy.Time(0)
        self._reverse_recovery_start_xy = None
        self._reverse_recovery_pause_until = rospy.Time(0)
        self._reverse_recovery_cooldown_until = rospy.Time(0)
        self._reverse_recovery_last_exit_reason = ""
        self._reverse_recovery_last_log_sec = 0.0
        self._reverse_recovery_trigger_since_sec = 0.0
        self._reverse_recovery_resume_distance_m = self.reverse_recovery_distance_m
        self._reverse_recovery_required_front_clearance_m = (
            self.emergency_stop_distance + self.reverse_recovery_clearance_margin_m
        )
        self.reverse_recovery_rear_local_points = np.empty((0, 2), dtype=np.float32)

        # ===== ROS I/O =====
        self.sub_path_global = rospy.Subscriber(self.global_path_topic, Path, self.path_callback_global, queue_size=5)
        self.sub_path_local = None
        if not self.follow_global_path_only:
            self.sub_path_local = rospy.Subscriber(
                self.local_path_topic,
                Path,
                self.path_callback_local,
                queue_size=1,
                tcp_nodelay=True,
            )
        self.sub_pose = rospy.Subscriber(
            self.pose_topic,
            Odometry,
            self.pose_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.sub_server_cmd = rospy.Subscriber("server_to_robot_topic", server_to_robot, self.server_to_robot_callback)
        self.sub_behavior = rospy.Subscriber(
            self.behavior_cmd_topic,
            BehaviorCommand,
            self.behavior_cmd_callback,
            queue_size=10,
        )
        self.sub_path_mode = None
        if not self.follow_global_path_only:
            self.sub_path_mode = rospy.Subscriber(
                self.path_mode_topic,
                String,
                self.path_mode_callback,
                queue_size=10,
            )
        self.near_field_raw_stop_marker_pub = None
        if self.near_field_raw_stop_marker_topic:
            self.near_field_raw_stop_marker_pub = rospy.Publisher(
                self.near_field_raw_stop_marker_topic,
                MarkerArray,
                queue_size=2,
                latch=True,
            )
        self.near_field_raw_stop_hit_cloud_pub = None
        if self.near_field_raw_stop_hit_cloud_topic:
            self.near_field_raw_stop_hit_cloud_pub = rospy.Publisher(
                self.near_field_raw_stop_hit_cloud_topic,
                PointCloud2,
                queue_size=2,
                latch=True,
            )
        self.near_field_raw_stop_tf_buffer = None
        self.near_field_raw_stop_tf_listener = None
        if self.near_field_raw_stop_enabled and self.near_field_raw_stop_base_frame:
            self.near_field_raw_stop_tf_buffer = tf2_ros.Buffer(
                cache_time=rospy.Duration(5.0)
            )
            self.near_field_raw_stop_tf_listener = tf2_ros.TransformListener(
                self.near_field_raw_stop_tf_buffer
            )
        self.sub_cloud = None
        if self.enable_raw_cloud_processing:
            self.sub_cloud = rospy.Subscriber(
                self.cloud_topic,
                PointCloud2,
                self.cloud_callback,
                queue_size=1,
                buff_size=2**24,
                tcp_nodelay=True,
            )
        self.sub_near_field_raw_stop = None
        if self.near_field_raw_stop_enabled and self.near_field_raw_stop_topic:
            self.sub_near_field_raw_stop = rospy.Subscriber(
                self.near_field_raw_stop_topic,
                PointCloud2,
                self.near_field_raw_stop_callback,
                queue_size=1,
                buff_size=2**24,
                tcp_nodelay=True,
            )
        self.sub_global_obstacle_overlay_boxes = None
        if (
            self.use_global_obstacle_overlay_boxes_for_stop
            and self.global_obstacle_overlay_boxes_topic
        ):
            self.sub_global_obstacle_overlay_boxes = rospy.Subscriber(
                self.global_obstacle_overlay_boxes_topic,
                MarkerArray,
                self.global_obstacle_overlay_boxes_callback,
                queue_size=2,
            )
        self.sub_grid = None
        if self.use_drivable_grid:
            self.sub_grid = rospy.Subscriber(self.drivable_grid_topic, OccupancyGrid, self.drivable_grid_callback, queue_size=5)
        self.sub_risk_grid = None
        if self.use_dynamic_risk_grid:
            self.sub_risk_grid = rospy.Subscriber(self.dynamic_risk_grid_topic, OccupancyGrid, self.risk_grid_callback, queue_size=5)

        self.cmd_vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=10)
        self.emergency_stop_pub = rospy.Publisher(self.emergency_stop_topic, Bool, queue_size=2)
        self.target_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
        self.current_pub = rospy.Publisher('visualization_marker_2', Marker, queue_size=10)
        self.trajectory_pub = rospy.Publisher('predicted_trajectory', Marker, queue_size=10)
        self.tracking_reference_pub = rospy.Publisher(
            self.tracking_reference_topic, Path, queue_size=2, latch=True
        )
        self.traj_info_pub = rospy.Publisher('/traj_info', Float32MultiArray, queue_size=10)
        self.cmd_timer = rospy.Timer(rospy.Duration(1.0 / self.cmd_publish_hz), self._cmd_timer_callback)
        if self.near_field_raw_stop_enabled:
            self._publish_near_field_raw_stop_debug(self._near_field_status_header(), [])

    def _cmd_timer_callback(self, _event):
        if rospy.is_shutdown():
            return
        try:
            if self._publish_emergency_stop_state():
                self.last_cmd = Twist()
                self._last_cmd_smooth_stamp = rospy.Time.now()
            self.cmd_vel_pub.publish(self.last_cmd)
            self._refresh_near_field_raw_stop_marker_if_waiting()
        except rospy.ROSException:
            pass

    def _hard_stop_active(self):
        # A planned avoidance detour may continue through the outer emergency band
        # when the controller has already verified the path bends around the obstacle.
        emergency_hard_stop = (
            self.enable_emergency_stop
            and self.emergency_blocked
            and (not self._emergency_bypass_active)
        )
        return bool(emergency_hard_stop or self.behavior_stop)

    def _publish_emergency_stop_state(self):
        hard_stop_active = self._hard_stop_active()
        pub = getattr(self, "emergency_stop_pub", None)
        if pub is not None:
            pub.publish(Bool(data=hard_stop_active))
        return hard_stop_active

    @staticmethod
    def _transform_world_to_local(wx, wy, pose_x, pose_y, yaw):
        dx = float(wx) - float(pose_x)
        dy = float(wy) - float(pose_y)
        c = math.cos(float(yaw))
        s = math.sin(float(yaw))
        return c * dx + s * dy, -s * dx + c * dy

    @staticmethod
    def _transform_local_to_world(lx, ly, pose_x, pose_y, yaw):
        c = math.cos(float(yaw))
        s = math.sin(float(yaw))
        return (
            float(pose_x) + c * float(lx) - s * float(ly),
            float(pose_y) + s * float(lx) + c * float(ly),
        )

    def _world_box_to_local_bounds(self, box, pose_x, pose_y, yaw):
        corners = (
            (float(box["min_x"]), float(box["min_y"])),
            (float(box["min_x"]), float(box["max_y"])),
            (float(box["max_x"]), float(box["min_y"])),
            (float(box["max_x"]), float(box["max_y"])),
        )
        local_pts = [
            self._transform_world_to_local(wx, wy, pose_x, pose_y, yaw)
            for wx, wy in corners
        ]
        xs = [pt[0] for pt in local_pts]
        ys = [pt[1] for pt in local_pts]
        return min(xs), max(xs), min(ys), max(ys)

    def _overlay_boxes_are_fresh(self):
        if not self.use_global_obstacle_overlay_boxes_for_stop:
            return False
        stamp_sec = self.global_obstacle_overlay_boxes_stamp.to_sec()
        if stamp_sec <= 0.0:
            return False
        if self.global_obstacle_overlay_box_timeout_s <= 0.0:
            return True
        return (
            rospy.Time.now() - self.global_obstacle_overlay_boxes_stamp
        ).to_sec() <= self.global_obstacle_overlay_box_timeout_s

    def _evaluate_overlay_box_blocking(self):
        if not self._overlay_boxes_are_fresh() or not self.global_obstacle_overlay_boxes:
            return False, float("inf"), 0

        pose = self.current_pose.pose.pose.position
        yaw = self.get_yaw_from_quaternion(self.current_pose.pose.pose.orientation)
        stop_half_w = 0.5 * max(self.stop_width, self.robot_width_m) + self.footprint_padding_m
        footprint_front = self.robot_half_length_m + self.footprint_padding_m
        relevant_boxes = 0
        min_clearance = float("inf")
        blocked = False

        for box in self.global_obstacle_overlay_boxes:
            if self.global_obstacle_overlay_stop_locked_only and (not box.get("locked", False)):
                continue
            local_min_x, local_max_x, local_min_y, local_max_y = self._world_box_to_local_bounds(
                box,
                pose.x,
                pose.y,
                yaw,
            )
            if local_max_x < self.obstacle_consider_back_m:
                continue
            if local_min_y > stop_half_w or local_max_y < -stop_half_w:
                continue
            relevant_boxes += 1
            clearance = local_min_x - footprint_front
            if clearance < min_clearance:
                min_clearance = clearance
            if local_max_x >= footprint_front and clearance <= self.emergency_stop_distance:
                blocked = True

        return blocked, min_clearance, relevant_boxes

    def _update_emergency_stop_state(self):
        overlay_blocked, overlay_clearance, overlay_match_count = self._evaluate_overlay_box_blocking()
        self.overlay_box_blocked = bool(overlay_blocked)
        self.overlay_box_front_clearance = overlay_clearance
        self._overlay_box_match_count = int(overlay_match_count)
        near_raw_fresh = (
            self._near_field_raw_stop_last_stamp.to_sec() > 0.0
            and (
                rospy.Time.now() - self._near_field_raw_stop_last_stamp
            ).to_sec() <= self.near_field_raw_stop_timeout_s
        )
        near_raw_hold_active = rospy.Time.now() < self._near_field_raw_stop_hold_until
        near_raw_blocked = near_raw_hold_active or (
            self._near_field_raw_stop_blocked and near_raw_fresh
        )

        raw_fallback_blocked = self.enable_raw_cloud_fallback_stop and (
            self._raw_immediate_contact_blocked
            or (
                self._raw_emergency_band_blocked
                and self._raw_front_obstacle_clearance
                <= self.avoidance_hard_stop_distance
            )
        )
        if self.use_global_obstacle_overlay_boxes_for_stop:
            near = self.overlay_box_blocked or raw_fallback_blocked or near_raw_blocked
            source_parts = []
            if self.overlay_box_blocked:
                source_parts.append("overlay_boxes")
            if near_raw_blocked:
                source_parts.append("near_raw")
            if self._raw_immediate_contact_blocked:
                source_parts.append("raw_intrusion")
            elif raw_fallback_blocked:
                source_parts.append("raw_near_fallback")
        else:
            near = (
                self._raw_immediate_contact_blocked
                or self._raw_emergency_band_blocked
                or near_raw_blocked
            )
            source_parts = []
            if near_raw_blocked:
                source_parts.append("near_raw")
            if self._raw_immediate_contact_blocked:
                source_parts.append("raw_intrusion")
            elif self._raw_emergency_band_blocked:
                source_parts.append("raw_band")

        self.front_obstacle_clearance = min(
            self._raw_front_obstacle_clearance,
            self.overlay_box_front_clearance,
            self._near_field_raw_stop_last_min_x
            - (self.robot_half_length_m + self.footprint_padding_m)
            if near_raw_fresh
            else float("inf"),
        )
        self._last_emergency_source = ",".join(source_parts) if source_parts else "clear"

        if near:
            self._blk_on += 1
            self._blk_off = 0
        else:
            self._blk_off += 1
            self._blk_on = 0
        state_changed = False
        if not self.emergency_blocked and self._blk_on >= self.block_on_count:
            self.emergency_blocked = True
            state_changed = True
            self.last_cmd = Twist()
            self._last_cmd_smooth_stamp = rospy.Time.now()
            if getattr(self, "cmd_vel_pub", None) is not None:
                self.cmd_vel_pub.publish(self.last_cmd)
            rospy.logwarn(
                "Emergency STOP: obstacle <= %.2fm | clearance=%.2f raw=%.2f overlay=%.2f close=%d intrusion=%d overlay_boxes=%d source=%s",
                self.emergency_stop_distance,
                self.front_obstacle_clearance if math.isfinite(self.front_obstacle_clearance) else float("inf"),
                self._raw_front_obstacle_clearance if math.isfinite(self._raw_front_obstacle_clearance) else float("inf"),
                self.overlay_box_front_clearance if math.isfinite(self.overlay_box_front_clearance) else float("inf"),
                self._last_emergency_close_points,
                self._last_emergency_intrusion_points,
                self._overlay_box_match_count,
                self._last_emergency_source,
            )
        elif self.emergency_blocked and self._blk_off >= self.block_off_count:
            self.emergency_blocked = False
            state_changed = True
            rospy.loginfo("Emergency STOP cleared")
        if state_changed:
            self._publish_emergency_stop_state()

    @staticmethod
    def _same_frame(frame_a, frame_b):
        return str(frame_a).strip().lstrip("/") == str(frame_b).strip().lstrip("/")

    def _near_field_raw_stop_transform_matrix(self, msg):
        source_frame = str(msg.header.frame_id).strip()
        target_frame = str(self.near_field_raw_stop_base_frame).strip()
        self._near_field_raw_stop_last_frame = source_frame
        if (not target_frame) or self._same_frame(source_frame, target_frame):
            self._near_field_raw_stop_tf_status = "raw_frame"
            return None, True, source_frame
        if self.near_field_raw_stop_tf_buffer is None:
            self._near_field_raw_stop_tf_status = "tf_buffer_missing"
            return None, False, target_frame

        stamp = msg.header.stamp
        if stamp.to_sec() <= 0.0:
            stamp = rospy.Time(0)
        first_error = None
        try:
            tf_msg = self.near_field_raw_stop_tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                stamp,
                rospy.Duration(self.near_field_raw_stop_tf_timeout_s),
            )
            self._near_field_raw_stop_tf_status = "tf_stamp"
        except Exception as exc:
            first_error = exc
            tf_msg = None

        if tf_msg is None and self.near_field_raw_stop_tf_fallback_latest:
            try:
                tf_msg = self.near_field_raw_stop_tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    rospy.Time(0),
                    rospy.Duration(self.near_field_raw_stop_tf_timeout_s),
                )
                self._near_field_raw_stop_tf_status = "tf_latest"
            except Exception as exc:
                first_error = exc

        if tf_msg is None:
            if self.near_field_raw_stop_allow_raw_frame_on_tf_failure:
                self._near_field_raw_stop_tf_status = "raw_fallback_tf_missing"
                now_sec = rospy.Time.now().to_sec()
                if now_sec - self._near_field_raw_stop_tf_warn_sec > 1.0:
                    self._near_field_raw_stop_tf_warn_sec = now_sec
                    rospy.logwarn(
                        "near_field_raw_stop: missing TF %s <- %s; using raw cloud frame ROI for debug: %s",
                        target_frame,
                        source_frame,
                        str(first_error),
                    )
                return None, True, target_frame

            self._near_field_raw_stop_tf_status = "tf_missing"
            now_sec = rospy.Time.now().to_sec()
            if now_sec - self._near_field_raw_stop_tf_warn_sec > 1.0:
                self._near_field_raw_stop_tf_warn_sec = now_sec
                rospy.logwarn(
                    "near_field_raw_stop: missing TF %s <- %s: %s",
                    target_frame,
                    source_frame,
                    str(first_error),
                )
            return None, False, target_frame

        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        mat = transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        mat[0, 3] = t.x
        mat[1, 3] = t.y
        mat[2, 3] = t.z
        return mat, True, target_frame

    @staticmethod
    def _transform_point_xyz(mat, x, y, z):
        if mat is None:
            return float(x), float(y), float(z)
        return (
            float(mat[0, 0] * x + mat[0, 1] * y + mat[0, 2] * z + mat[0, 3]),
            float(mat[1, 0] * x + mat[1, 1] * y + mat[1, 2] * z + mat[1, 3]),
            float(mat[2, 0] * x + mat[2, 1] * y + mat[2, 2] * z + mat[2, 3]),
        )

    @staticmethod
    def _pointcloud_xyz_offsets(msg):
        offsets = {}
        for field in msg.fields:
            name = str(field.name)
            if name in ("x", "y", "z"):
                offsets[name] = int(field.offset)
        if len(offsets) != 3:
            return None
        return offsets["x"], offsets["y"], offsets["z"]

    def _pointcloud_xyz_numpy(self, msg, downsample=1):
        offsets = self._pointcloud_xyz_offsets(msg)
        if offsets is None:
            return None
        point_step = int(msg.point_step)
        count = max(0, int(msg.width) * max(1, int(msg.height)))
        if point_step <= 0 or count <= 0:
            return None
        if len(msg.data) < count * point_step:
            return None

        endian = ">" if bool(msg.is_bigendian) else "<"
        dtype = np.dtype(
            {
                "names": ("x", "y", "z"),
                "formats": (endian + "f4", endian + "f4", endian + "f4"),
                "offsets": offsets,
                "itemsize": point_step,
            }
        )
        try:
            raw = np.frombuffer(msg.data, dtype=dtype, count=count)
        except (TypeError, ValueError):
            return None

        step = max(1, int(downsample))
        raw = raw[::step]
        points = np.empty((raw.shape[0], 3), dtype=np.float32)
        points[:, 0] = raw["x"]
        points[:, 1] = raw["y"]
        points[:, 2] = raw["z"]
        return points

    @staticmethod
    def _transform_points_xyz(mat, points):
        if mat is None or points.size == 0:
            return points
        rot = np.asarray(mat[:3, :3], dtype=np.float32)
        trans = np.asarray(mat[:3, 3], dtype=np.float32)
        return points @ rot.T + trans

    def _points_to_publish_xyz(self, points):
        if points.size == 0:
            return []
        max_points = int(self.near_field_raw_stop_publish_max_points)
        if max_points > 0 and points.shape[0] > max_points:
            idx = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int32)
            points = points[idx]
        return [tuple(map(float, row)) for row in points]

    @staticmethod
    def _transform_local_to_world_xy(local_x, local_y, pose_x, pose_y, yaw):
        c = math.cos(float(yaw))
        s = math.sin(float(yaw))
        return (
            float(pose_x) + c * float(local_x) - s * float(local_y),
            float(pose_y) + s * float(local_x) + c * float(local_y),
        )

    def _near_field_header(self, msg, frame_id=None):
        header = Header()
        # These debug markers/point clouds are already transformed into the
        # local stop frame. Publishing them with the original LiDAR stamp makes
        # RViz render them against an older TF snapshot, which looks like the
        # near-stop box is lagging behind the robot in map-fixed view.
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id or self.near_field_raw_stop_base_frame or msg.header.frame_id
        return header

    def _near_field_status_header(self):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.near_field_raw_stop_base_frame or "base_link"
        return header

    def _refresh_near_field_raw_stop_marker_if_waiting(self):
        if not self.near_field_raw_stop_enabled:
            return
        if self.near_field_raw_stop_marker_pub is None:
            return
        now = rospy.Time.now()
        if (now - self._near_field_raw_stop_last_marker_refresh).to_sec() < 0.5:
            return
        self._near_field_raw_stop_last_marker_refresh = now
        fresh = (
            self._near_field_raw_stop_last_stamp.to_sec() > 0.0
            and (now - self._near_field_raw_stop_last_stamp).to_sec()
            <= self.near_field_raw_stop_timeout_s
        )
        if not fresh:
            self._near_field_raw_stop_tf_status = (
                self._near_field_raw_stop_tf_status
                if self._near_field_raw_stop_tf_status != "not_started"
                else "waiting_cloud"
            )
            self._publish_near_field_raw_stop_debug(self._near_field_status_header(), [])

    def _publish_near_field_raw_stop_debug(self, header, hit_points):
        if self.near_field_raw_stop_hit_cloud_pub is not None:
            self.near_field_raw_stop_hit_cloud_pub.publish(
                point_cloud2.create_cloud_xyz32(header, hit_points)
            )

        if self.near_field_raw_stop_marker_pub is None:
            return

        marker_header = Header()
        marker_header.stamp = rospy.Time.now()
        marker_header.frame_id = header.frame_id
        use_world_pose = False
        pose_frame = str(self.current_pose.header.frame_id).strip()
        pose_x = 0.0
        pose_y = 0.0
        pose_z = 0.0
        pose_yaw = 0.0
        pose_orientation = None
        if pose_frame:
            marker_header.frame_id = pose_frame
            pose_x = float(self.current_pose.pose.pose.position.x)
            pose_y = float(self.current_pose.pose.pose.position.y)
            pose_z = float(self.current_pose.pose.pose.position.z)
            pose_orientation = self.current_pose.pose.pose.orientation
            pose_yaw = self.get_yaw_from_quaternion(pose_orientation)
            use_world_pose = True

        markers = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        box = Marker()
        box.header = marker_header
        box.ns = "near_field_raw_stop"
        box.id = 0
        box.type = Marker.CUBE
        box.action = Marker.ADD
        center_x_local = 0.5 * (
            self.near_field_raw_stop_min_x_m + self.near_field_raw_stop_max_x_m
        )
        center_y_local = 0.0
        center_z_local = 0.5 * (
            self.near_field_raw_stop_min_z_m + self.near_field_raw_stop_max_z_m
        )
        if use_world_pose:
            wx, wy = self._transform_local_to_world_xy(
                center_x_local, center_y_local, pose_x, pose_y, pose_yaw
            )
            box.pose.position.x = wx
            box.pose.position.y = wy
            box.pose.position.z = pose_z + center_z_local
            box.pose.orientation = pose_orientation
            box.frame_locked = False
        else:
            box.pose.position.x = center_x_local
            box.pose.position.y = center_y_local
            box.pose.position.z = center_z_local
            box.pose.orientation.w = 1.0
            box.frame_locked = True
        box.scale.x = self.near_field_raw_stop_max_x_m - self.near_field_raw_stop_min_x_m
        box.scale.y = 2.0 * self.near_field_raw_stop_half_width_m
        box.scale.z = self.near_field_raw_stop_max_z_m - self.near_field_raw_stop_min_z_m
        box.color.a = 0.32 if self._near_field_raw_stop_blocked else 0.18
        box.color.r = 1.0 if self._near_field_raw_stop_blocked else 0.0
        box.color.g = 0.05 if self._near_field_raw_stop_blocked else 0.95
        box.color.b = 0.05 if self._near_field_raw_stop_blocked else 0.95
        box.lifetime = rospy.Duration(max(0.2, self.near_field_raw_stop_timeout_s * 2.0))
        markers.markers.append(box)

        outline = Marker()
        outline.header = marker_header
        outline.ns = "near_field_raw_stop"
        outline.id = 2
        outline.type = Marker.LINE_LIST
        outline.action = Marker.ADD
        outline.frame_locked = not use_world_pose
        outline.pose.orientation.w = 1.0
        outline.scale.x = 0.03
        outline.color.a = 1.0
        outline.color.r = 1.0 if self._near_field_raw_stop_blocked else 0.0
        outline.color.g = 0.05 if self._near_field_raw_stop_blocked else 0.95
        outline.color.b = 0.05 if self._near_field_raw_stop_blocked else 1.0
        outline.lifetime = box.lifetime
        min_x = self.near_field_raw_stop_min_x_m
        max_x = self.near_field_raw_stop_max_x_m
        min_y = -self.near_field_raw_stop_half_width_m
        max_y = self.near_field_raw_stop_half_width_m
        min_z = self.near_field_raw_stop_min_z_m
        max_z = self.near_field_raw_stop_max_z_m
        corners_local = [
            (min_x, min_y, min_z),
            (max_x, min_y, min_z),
            (max_x, max_y, min_z),
            (min_x, max_y, min_z),
            (min_x, min_y, max_z),
            (max_x, min_y, max_z),
            (max_x, max_y, max_z),
            (min_x, max_y, max_z),
        ]
        corners = []
        for lx, ly, lz in corners_local:
            if use_world_pose:
                wx, wy = self._transform_local_to_world_xy(
                    lx, ly, pose_x, pose_y, pose_yaw
                )
                corners.append(Point(wx, wy, pose_z + lz))
            else:
                corners.append(Point(lx, ly, lz))
        edge_indices = (
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )
        for start_idx, end_idx in edge_indices:
            outline.points.append(corners[start_idx])
            outline.points.append(corners[end_idx])
        markers.markers.append(outline)

        text = Marker()
        text.header = marker_header
        text.ns = "near_field_raw_stop"
        text.id = 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        if use_world_pose:
            text_x, text_y = self._transform_local_to_world_xy(
                self.near_field_raw_stop_max_x_m, 0.0, pose_x, pose_y, pose_yaw
            )
            text.pose.position.x = text_x
            text.pose.position.y = text_y
            text.pose.position.z = pose_z + self.near_field_raw_stop_max_z_m + 0.25
            text.frame_locked = False
        else:
            text.pose.position.x = self.near_field_raw_stop_max_x_m
            text.pose.position.y = 0.0
            text.pose.position.z = self.near_field_raw_stop_max_z_m + 0.25
            text.frame_locked = True
        text.pose.orientation.w = 1.0
        text.scale.z = 0.22
        text.color.a = 1.0
        text.color.r = 1.0 if self._near_field_raw_stop_blocked else 0.1
        text.color.g = 0.1 if self._near_field_raw_stop_blocked else 1.0
        text.color.b = 0.1
        front_clearance = (
            self._near_field_raw_stop_last_min_x
            - (self.robot_half_length_m + self.footprint_padding_m)
            if math.isfinite(self._near_field_raw_stop_last_min_x)
            else float("inf")
        )
        text.text = "NEAR STOP: {}\npts={} cells={} clr={:.2f}m\nframe={} tf={}".format(
            "STOP" if self._near_field_raw_stop_blocked else "clear",
            self._near_field_raw_stop_last_count,
            self._near_field_raw_stop_last_cells,
            front_clearance if math.isfinite(front_clearance) else -1.0,
            self._near_field_raw_stop_last_frame or "-",
            "{}|{}".format(self._near_field_raw_stop_tf_status, self._near_field_raw_stop_last_reason),
        )
        text.lifetime = box.lifetime
        markers.markers.append(text)

        self.near_field_raw_stop_marker_pub.publish(markers)

    def near_field_raw_stop_callback(self, msg):
        try:
            process_start = time.monotonic()
            if self.near_field_raw_stop_max_update_hz > 0.0:
                min_dt = 1.0 / self.near_field_raw_stop_max_update_hz
                if (
                    self._near_field_raw_stop_last_process_wall_sec > 0.0
                    and (
                        process_start
                        - self._near_field_raw_stop_last_process_wall_sec
                    )
                    < min_dt
                ):
                    return
                self._near_field_raw_stop_last_process_wall_sec = process_start
            transform_mat, transform_ok, debug_frame = self._near_field_raw_stop_transform_matrix(msg)
            header = self._near_field_header(msg, debug_frame)
            if not transform_ok:
                self._near_field_raw_stop_last_stamp = rospy.Time.now()
                self._near_field_raw_stop_last_count = 0
                self._near_field_raw_stop_last_cells = 0
                self._near_field_raw_stop_last_min_x = float("inf")
                self._near_field_raw_stop_last_process_ms = (
                    time.monotonic() - process_start
                ) * 1000.0
                self._publish_near_field_raw_stop_debug(header, [])
                self._update_emergency_stop_state()
                return

            hit_points = []
            min_x = float("inf")
            footprint_front = self.robot_half_length_m + self.footprint_padding_m
            stop_detected = False
            close_override_detected = False
            points = self._pointcloud_xyz_numpy(
                msg, downsample=self.near_field_raw_stop_downsample
            )
            if points is None:
                occupied_cells = set()
                close_override_cells = set()
                i = 0
                for x, y, z in point_cloud2.read_points(
                    msg, field_names=("x", "y", "z"), skip_nans=True
                ):
                    i += 1
                    if (
                        self.near_field_raw_stop_downsample > 1
                        and (i % self.near_field_raw_stop_downsample != 0)
                    ):
                        continue
                    x, y, z = self._transform_point_xyz(
                        transform_mat, float(x), float(y), float(z)
                    )
                    if (
                        self.enable_self_filter
                        and self._point_in_local_rect(
                            x,
                            y,
                            self.self_filter_radius_x,
                            self.self_filter_radius_y,
                        )
                    ):
                        continue
                    if not (
                        self.near_field_raw_stop_min_x_m
                        <= x
                        <= self.near_field_raw_stop_max_x_m
                    ):
                        continue
                    if abs(y) > self.near_field_raw_stop_half_width_m:
                        continue
                    if not (
                        self.near_field_raw_stop_min_z_m
                        <= z
                        <= self.near_field_raw_stop_max_z_m
                    ):
                        continue
                    hit_points.append((x, y, z))
                    occupied_cells.add(
                        (
                            int(math.floor(x / self.near_field_raw_stop_cell_size_m)),
                            int(math.floor(y / self.near_field_raw_stop_cell_size_m)),
                        )
                    )
                    if abs(y) <= self.near_field_raw_stop_close_override_half_width_m:
                        close_override_cells.add(
                            (
                                int(
                                    math.floor(
                                        x / self.near_field_raw_stop_cell_size_m
                                    )
                                ),
                                int(
                                    math.floor(
                                        y / self.near_field_raw_stop_cell_size_m
                                    )
                                ),
                            )
                        )
                    if x < min_x:
                        min_x = x
                hit_count = len(hit_points)
                cell_count = len(occupied_cells)
                close_xy_points = [(float(x), float(y)) for x, y, _ in hit_points]
                close_override_hit_count = sum(
                    1
                    for _, y, _ in hit_points
                    if abs(y) <= self.near_field_raw_stop_close_override_half_width_m
                )
                close_override_cell_count = len(close_override_cells)
                close_override_min_x = min(
                    (
                        float(x)
                        for x, y, _ in hit_points
                        if abs(y) <= self.near_field_raw_stop_close_override_half_width_m
                    ),
                    default=float("inf"),
                )
            else:
                finite_mask = np.isfinite(points).all(axis=1)
                points = points[finite_mask]
                points = self._transform_points_xyz(transform_mat, points)
                if self.enable_self_filter and points.size > 0:
                    self_mask = (
                        (np.abs(points[:, 0]) <= self.self_filter_radius_x)
                        & (np.abs(points[:, 1]) <= self.self_filter_radius_y)
                    )
                    points = points[~self_mask]
                roi_mask = (
                    (points[:, 0] >= self.near_field_raw_stop_min_x_m)
                    & (points[:, 0] <= self.near_field_raw_stop_max_x_m)
                    & (np.abs(points[:, 1]) <= self.near_field_raw_stop_half_width_m)
                    & (points[:, 2] >= self.near_field_raw_stop_min_z_m)
                    & (points[:, 2] <= self.near_field_raw_stop_max_z_m)
                )
                hit_points_arr = points[roi_mask]
                hit_count = int(hit_points_arr.shape[0])
                if hit_count > 0:
                    min_x = float(np.min(hit_points_arr[:, 0]))
                    cells = np.floor(
                        hit_points_arr[:, :2] / self.near_field_raw_stop_cell_size_m
                    ).astype(np.int32)
                    cell_count = int(np.unique(cells, axis=0).shape[0])
                    hit_points = self._points_to_publish_xyz(hit_points_arr)
                    close_xy_points = [
                        (float(x), float(y)) for x, y in hit_points_arr[:, :2]
                    ]
                    close_override_mask = (
                        np.abs(hit_points_arr[:, 1])
                        <= self.near_field_raw_stop_close_override_half_width_m
                    )
                    close_override_arr = hit_points_arr[close_override_mask]
                    close_override_hit_count = int(close_override_arr.shape[0])
                    if close_override_hit_count > 0:
                        close_override_min_x = float(np.min(close_override_arr[:, 0]))
                        close_override_cells = np.floor(
                            close_override_arr[:, :2]
                            / self.near_field_raw_stop_cell_size_m
                        ).astype(np.int32)
                        close_override_cell_count = int(
                            np.unique(close_override_cells, axis=0).shape[0]
                        )
                    else:
                        close_override_min_x = float("inf")
                        close_override_cell_count = 0
                else:
                    cell_count = 0
                    close_xy_points = []
                    close_override_hit_count = 0
                    close_override_cell_count = 0
                    close_override_min_x = float("inf")

            min_front_clearance = (
                min_x - footprint_front if math.isfinite(min_x) else float("inf")
            )
            close_override_front_clearance = (
                close_override_min_x - footprint_front
                if math.isfinite(close_override_min_x)
                else float("inf")
            )
            detected = (
                hit_count >= self.near_field_raw_stop_min_points
                and cell_count >= self.near_field_raw_stop_min_cells
            )
            close_override_detected = (
                close_override_hit_count
                >= self.near_field_raw_stop_close_override_min_points
                and close_override_cell_count
                >= self.near_field_raw_stop_close_override_min_cells
                and close_override_front_clearance
                <= self.near_field_raw_stop_close_override_distance_m
            )
            passable_gap_blocked = self._emergency_band_is_blocked(
                close_xy_points,
                max(
                    self.near_field_raw_stop_half_width_m,
                    self.robot_half_width_m + self.footprint_padding_m,
                ),
            )
            detected = detected or close_override_detected
            stop_detected = (
                detected
                and min_front_clearance <= self.emergency_stop_distance
                and (close_override_detected or passable_gap_blocked)
            )
            if stop_detected:
                self._near_field_raw_stop_last_reason = (
                    "close_override" if close_override_detected else "standard"
                )
            elif detected and math.isfinite(min_front_clearance):
                self._near_field_raw_stop_last_reason = "passable_gap"
            else:
                self._near_field_raw_stop_last_reason = "clear"
            if stop_detected:
                self._near_field_raw_stop_on += 1
                self._near_field_raw_stop_off = 0
                if self.near_field_raw_stop_hold_s > 0.0:
                    self._near_field_raw_stop_hold_until = (
                        rospy.Time.now() + rospy.Duration(self.near_field_raw_stop_hold_s)
                    )
            else:
                self._near_field_raw_stop_off += 1
                self._near_field_raw_stop_on = 0

            if (
                not self._near_field_raw_stop_blocked
                and self._near_field_raw_stop_on >= self.near_field_raw_stop_on_count
            ):
                self._near_field_raw_stop_blocked = True
            elif (
                self._near_field_raw_stop_blocked
                and self._near_field_raw_stop_off >= self.near_field_raw_stop_off_count
            ):
                self._near_field_raw_stop_blocked = False

            # Use receive time for control freshness. Some Ouster bags/live streams
            # can carry sensor/header stamps that drift from ROS time; RViz would
            # still show the STOP marker while the controller considered it stale.
            self._near_field_raw_stop_last_stamp = rospy.Time.now()
            self._near_field_raw_stop_last_count = hit_count
            self._near_field_raw_stop_last_cells = cell_count
            self._near_field_raw_stop_last_min_x = min_x
            self._near_field_raw_stop_last_process_ms = (
                time.monotonic() - process_start
            ) * 1000.0
            self._publish_near_field_raw_stop_debug(header, hit_points)
            self._update_emergency_stop_state()

            if self.near_field_raw_stop_log_period_s > 0.0:
                now = rospy.Time.now()
                if (
                    now - self._near_field_raw_stop_last_log
                ).to_sec() >= self.near_field_raw_stop_log_period_s:
                    self._near_field_raw_stop_last_log = now
                    rospy.loginfo(
                        "near_field_raw_stop: %s reason=%s pts=%d cells=%d min_x=%.2f clearance=%.2f stop_dist=%.2f close_override=%.2f process=%.1fms roi=[x %.2f..%.2f y +/-%.2f z %.2f..%.2f] topic=%s frame=%s tf=%s",
                        "STOP" if self._near_field_raw_stop_blocked else "clear",
                        self._near_field_raw_stop_last_reason,
                        hit_count,
                        cell_count,
                        min_x if math.isfinite(min_x) else float("inf"),
                        min_front_clearance if math.isfinite(min_front_clearance) else float("inf"),
                        self.emergency_stop_distance,
                        self.near_field_raw_stop_close_override_distance_m,
                        self._near_field_raw_stop_last_process_ms,
                        self.near_field_raw_stop_min_x_m,
                        self.near_field_raw_stop_max_x_m,
                        self.near_field_raw_stop_half_width_m,
                        self.near_field_raw_stop_min_z_m,
                        self.near_field_raw_stop_max_z_m,
                        self.near_field_raw_stop_topic,
                        self._near_field_raw_stop_last_frame or "-",
                        self._near_field_raw_stop_tf_status,
                    )
        except Exception as e:
            rospy.logwarn("near_field_raw_stop_callback error: %s", str(e))

    def _point_in_local_rect(self, x, y, half_length, half_width):
        return abs(x) <= half_length and abs(y) <= half_width

    def _point_in_local_footprint(self, x, y, padding=0.0):
        return self._point_in_local_rect(
            x,
            y,
            self.robot_half_length_m + padding,
            self.robot_half_width_m + padding,
        )

    def _rect_clearance_local(self, points_xy, padding=0.0):
        if points_xy.shape[0] == 0:
            return np.empty((0,), dtype=np.float32)
        half_length = self.robot_half_length_m + padding
        half_width = self.robot_half_width_m + padding
        dx = np.maximum(np.abs(points_xy[:, 0]) - half_length, 0.0)
        dy = np.maximum(np.abs(points_xy[:, 1]) - half_width, 0.0)
        return np.hypot(dx, dy)

    def _footprint_sample_offsets(self, step_m):
        step = max(0.05, float(step_m))
        key = round(step, 3)
        cached = self._footprint_sample_cache.get(key)
        if cached is not None:
            return cached

        half_length = self.robot_half_length_m + self.footprint_padding_m
        half_width = self.robot_half_width_m + self.footprint_padding_m
        xs = np.arange(-half_length, half_length + 0.5 * step, step, dtype=np.float32)
        ys = np.arange(-half_width, half_width + 0.5 * step, step, dtype=np.float32)
        if xs.size == 0 or abs(xs[-1] - half_length) > 1e-6:
            xs = np.append(xs, np.float32(half_length))
        if ys.size == 0 or abs(ys[-1] - half_width) > 1e-6:
            ys = np.append(ys, np.float32(half_width))
        grid_x, grid_y = np.meshgrid(xs, ys)
        offsets = np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float32)
        self._footprint_sample_cache[key] = offsets
        return offsets

    def _nearest_drivable_grid_distance_m(self, x, y, max_search_m=None):
        if not self.use_drivable_grid:
            return 0.0
        if (
            self.grid_data is None
            or self.grid_width <= 0
            or self.grid_height <= 0
            or self.grid_resolution is None
            or self.grid_resolution <= 0.0
        ):
            return 0.0

        gx = int(math.floor((x - self.grid_origin_x) / self.grid_resolution))
        gy = int(math.floor((y - self.grid_origin_y) / self.grid_resolution))
        if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
            idx = gy * self.grid_width + gx
            occ = self.grid_data[idx]
            if occ >= 0 and occ == 0:
                return 0.0
            if occ < 0 and (not self.grid_unknown_is_occupied):
                return 0.0

        search_m = (
            max_search_m
            if max_search_m is not None
            else max(self.robot_length_m, self.robot_width_m) + 2.0 * self.footprint_padding_m + 0.25
        )
        max_cells = max(1, int(math.ceil(search_m / max(1e-6, self.grid_resolution))))
        best = float("inf")

        x_min = max(0, gx - max_cells)
        x_max = min(self.grid_width - 1, gx + max_cells)
        y_min = max(0, gy - max_cells)
        y_max = min(self.grid_height - 1, gy + max_cells)
        for ny in range(y_min, y_max + 1):
            row = ny * self.grid_width
            cy = self.grid_origin_y + (float(ny) + 0.5) * self.grid_resolution
            for nx in range(x_min, x_max + 1):
                occ = self.grid_data[row + nx]
                if occ < 0:
                    if self.grid_unknown_is_occupied:
                        continue
                elif occ != 0:
                    continue
                cx = self.grid_origin_x + (float(nx) + 0.5) * self.grid_resolution
                dist = math.hypot(cx - float(x), cy - float(y))
                if dist < best:
                    best = dist
        return best

    def _is_xy_drivable_grid_ok_with_tolerance(self, x, y, tolerance_m=None):
        if self._is_xy_drivable_grid_ok(x, y):
            return True
        tol = (
            self.path_tracking_drivable_boundary_tolerance_m
            if tolerance_m is None
            else max(0.0, float(tolerance_m))
        )
        if tol <= 1e-6:
            return False
        search_m = tol + max(0.05, float(self.grid_resolution or 0.05))
        nearest = self._nearest_drivable_grid_distance_m(x, y, max_search_m=search_m)
        return math.isfinite(nearest) and nearest <= tol

    def _initial_inward_recovery_window_ok(
        self,
        traj,
        offsets,
        ignore_start_distance_m,
        check_drivable_grid=True,
    ):
        if (
            (not check_drivable_grid)
            or ignore_start_distance_m <= 1e-6
            or traj is None
            or len(traj) < 2
        ):
            return True

        tol = max(0.02, 0.5 * max(0.05, float(self.grid_resolution or 0.05)))
        search_m = max(
            self.robot_length_m,
            self.robot_width_m,
        ) + 2.0 * self.footprint_padding_m + tol

        pose0 = traj[0]
        yaw0 = float(pose0[2])
        c0 = math.cos(yaw0)
        s0 = math.sin(yaw0)
        base_ok = []
        prev_dists = []
        prev_outside = 0
        for ox, oy in offsets:
            wx = float(pose0[0]) + c0 * float(ox) - s0 * float(oy)
            wy = float(pose0[1]) + s0 * float(ox) + c0 * float(oy)
            if not self._is_xy_risk_ok(wx, wy):
                return False
            ok = self._is_xy_drivable_grid_ok_with_tolerance(wx, wy)
            base_ok.append(ok)
            if ok:
                prev_dists.append(0.0)
            else:
                dist = self._nearest_drivable_grid_distance_m(wx, wy, max_search_m=search_m)
                if not math.isfinite(dist):
                    return False
                prev_dists.append(dist)
                prev_outside += 1

        traveled_m = 0.0
        prev_row = pose0
        for row in traj[1::self.traj_check_step]:
            traveled_m += math.hypot(float(row[0]) - float(prev_row[0]), float(row[1]) - float(prev_row[1]))
            prev_row = row
            if traveled_m >= ignore_start_distance_m:
                break

            yaw = float(row[2])
            c = math.cos(yaw)
            s = math.sin(yaw)
            row_dists = []
            row_outside = 0
            for idx, (ox, oy) in enumerate(offsets):
                wx = float(row[0]) + c * float(ox) - s * float(oy)
                wy = float(row[1]) + s * float(ox) + c * float(oy)
                if not self._is_xy_risk_ok(wx, wy):
                    return False
                ok = self._is_xy_drivable_grid_ok_with_tolerance(wx, wy)
                if ok:
                    row_dists.append(0.0)
                    continue
                if base_ok[idx]:
                    return False
                dist = self._nearest_drivable_grid_distance_m(wx, wy, max_search_m=search_m)
                if not math.isfinite(dist):
                    return False
                if dist > prev_dists[idx] + tol:
                    return False
                row_dists.append(dist)
                row_outside += 1

            if row_outside > prev_outside:
                return False
            prev_dists = row_dists
            prev_outside = row_outside

        return True

    # ------------------------------- obstacle stop -------------------------------
    def _avoidance_mode_active(self):
        return self.current_path_mode == "follow_avoidance"

    def _active_path_preview_local_xy(self, preview_m=None):
        if self.active_path_source != "local":
            return None
        if len(self.path_pts) < 2:
            return None

        try:
            pose_x, pose_y = self._tracking_anchor_xy()
            yaw = self.get_yaw_from_quaternion(self.current_pose.pose.pose.orientation)
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            s_min = max(0.0, self.s_cur - self.tracking_projection_back_window_m)
            s_max = min(self.s_total, self.s_cur + self.tracking_projection_forward_window_m)
            s_proj, _, _, _ = self._project_to_path(pose_x, pose_y, s_min=s_min, s_max=s_max)
            base_s = max(self.s_cur, s_proj)
            preview_dist = (
                self.emergency_bypass_preview_m
                if preview_m is None
                else max(0.05, float(preview_m))
            )
            preview_s = min(self.s_total, base_s + preview_dist)
            preview_x, preview_y, _ = self._interp_xy_smoothed_tangent_at_s(preview_s)
            dx = float(preview_x) - pose_x
            dy = float(preview_y) - pose_y
            return (
                cos_yaw * dx + sin_yaw * dy,
                -sin_yaw * dx + cos_yaw * dy,
            )
        except Exception:
            return None

    def _emergency_band_is_blocked(self, close_points, lateral_limit):
        if len(close_points) < self.emergency_min_close_points:
            return False

        lateral_limit = max(lateral_limit, self.robot_half_width_m + self.footprint_padding_m)
        bin_size = max(0.05, self.emergency_bin_size_m)
        bin_count = max(1, int(math.ceil((2.0 * lateral_limit) / bin_size)))
        occupied = np.zeros(bin_count, dtype=np.uint8)

        for _, y in close_points:
            idx = int(math.floor((y + lateral_limit) / bin_size))
            if idx < 0 or idx >= bin_count:
                continue
            occupied[idx] = 1

        longest_free_run = 0
        free_run = 0
        for cell in occupied:
            if cell:
                longest_free_run = max(longest_free_run, free_run)
                free_run = 0
            else:
                free_run += 1
        longest_free_run = max(longest_free_run, free_run)
        max_free_gap_m = float(longest_free_run) * bin_size
        if max_free_gap_m + 1e-6 >= self.emergency_passable_width_m:
            return False

        preview_local = self._active_path_preview_local_xy()
        if preview_local is None:
            return True

        preview_local_x, preview_local_y = preview_local
        if preview_local_x <= 0.05:
            return True
        if abs(preview_local_y) < self.emergency_bypass_target_lateral_m:
            return True

        center_idx = int(math.floor((lateral_limit) / bin_size))
        center_idx = max(0, min(bin_count - 1, center_idx))
        side_free_run = 0
        if preview_local_y > 0.0:
            idx_iter = range(center_idx, bin_count)
        else:
            idx_iter = range(center_idx - 1, -1, -1)

        for idx in idx_iter:
            if idx < 0 or idx >= bin_count:
                break
            if occupied[idx]:
                break
            side_free_run += 1

        side_free_gap_m = float(side_free_run) * bin_size
        return side_free_gap_m + 1e-6 < self.emergency_passable_width_m

    def cloud_callback(self, msg):
        if not self.enable_raw_cloud_processing:
            return
        try:
            obs = []
            rear_obs = []
            close_points = []
            intrusion_points = []
            min_front_clearance = float("inf")
            influence_radius = max(
                self.obstacle_influence_distance,
                self.reverse_recovery_rear_check_distance_m
                + self.robot_half_length_m
                + self.footprint_padding_m
                + 0.10,
            )
            influence_sq = influence_radius * influence_radius
            stop_half_w = 0.5 * max(self.stop_width, self.robot_width_m) + self.footprint_padding_m
            i = 0
            for pt in point_cloud2.read_points(msg, field_names=('x','y','z'), skip_nans=True):
                i += 1
                if self.cloud_downsample > 1 and (i % self.cloud_downsample != 0):
                    continue
                x, y, z = pt
                if (
                    self.enable_self_filter
                    and self._point_in_local_rect(
                        x, y, self.self_filter_radius_x, self.self_filter_radius_y
                    )
                ):
                    continue
                if z < self.min_z or z > self.max_z:
                    continue
                d2 = x * x + y * y
                if d2 > influence_sq:
                    continue
                rear_check_limit_x = -(
                    self.robot_half_length_m + self.footprint_padding_m
                ) - self.reverse_recovery_rear_check_distance_m
                if (
                    x >= rear_check_limit_x
                    and x <= 0.10
                    and abs(y) <= self.reverse_recovery_rear_half_width_m
                ):
                    rear_obs.append((x, y))
                if x < self.obstacle_consider_back_m or abs(y) > self.obstacle_consider_side_m:
                    continue
                obs.append((x, y))
                front_clearance = x - (self.robot_half_length_m + self.footprint_padding_m)
                if abs(y) <= stop_half_w and front_clearance < min_front_clearance:
                    min_front_clearance = front_clearance
                if abs(y) > self.obstacle_consider_side_m:
                    continue
                if abs(y) <= stop_half_w and front_clearance < 0.0:
                    intrusion_points.append((x, y))
                    close_points.append((x, y))
                    continue
                if front_clearance < 0.0:
                    continue
                if front_clearance <= self.emergency_stop_distance:
                    close_points.append((x, y))

            immediate_contact = len(intrusion_points) >= self.emergency_immediate_contact_min_points
            self._raw_emergency_band_blocked = self._emergency_band_is_blocked(
                close_points,
                self.obstacle_consider_side_m,
            )
            self._raw_immediate_contact_blocked = bool(immediate_contact)
            self._last_emergency_close_points = len(close_points)
            self._last_emergency_intrusion_points = len(intrusion_points)

            if obs:
                step = max(1, len(obs) // self.max_obstacle_points)
                self.obstacle_local_points = np.array(obs[::step], dtype=np.float32)
            else:
                self.obstacle_local_points = np.empty((0, 2), dtype=np.float32)
            if rear_obs:
                step = max(1, len(rear_obs) // self.max_obstacle_points)
                self.reverse_recovery_rear_local_points = np.array(
                    rear_obs[::step], dtype=np.float32
                )
            else:
                self.reverse_recovery_rear_local_points = np.empty(
                    (0, 2), dtype=np.float32
                )
            self._raw_front_obstacle_clearance = min_front_clearance
            self._update_emergency_stop_state()
        except Exception as e:
            rospy.logwarn("cloud_callback error: %s", str(e))

    def global_obstacle_overlay_boxes_callback(self, msg):
        try:
            boxes = []
            latest_stamp = rospy.Time(0)
            for marker in msg.markers:
                if marker.action == Marker.DELETEALL:
                    continue
                if marker.action != Marker.ADD or marker.type != Marker.CUBE:
                    continue
                if marker.ns and marker.ns != "global_obstacle_overlay_boxes":
                    continue
                size_x = max(0.0, float(marker.scale.x))
                size_y = max(0.0, float(marker.scale.y))
                cx = float(marker.pose.position.x)
                cy = float(marker.pose.position.y)
                boxes.append(
                    {
                        "min_x": cx - 0.5 * size_x,
                        "max_x": cx + 0.5 * size_x,
                        "min_y": cy - 0.5 * size_y,
                        "max_y": cy + 0.5 * size_y,
                        "locked": float(marker.color.a) >= 0.5,
                    }
                )
                if marker.header.stamp.to_sec() > latest_stamp.to_sec():
                    latest_stamp = marker.header.stamp
            self.global_obstacle_overlay_boxes = boxes
            # Freshness is about when this node received the overlay, not the
            # marker's source timestamp. This keeps visual STOP and control STOP
            # on the same clock.
            self.global_obstacle_overlay_boxes_stamp = rospy.Time.now()
            self._update_emergency_stop_state()
        except Exception as e:
            rospy.logwarn("global_obstacle_overlay_boxes_callback error: %s", str(e))

    def drivable_grid_callback(self, msg):
        try:
            self.grid_resolution = float(msg.info.resolution)
            self.grid_width = int(msg.info.width)
            self.grid_height = int(msg.info.height)
            self.grid_origin_x = float(msg.info.origin.position.x)
            self.grid_origin_y = float(msg.info.origin.position.y)
            self.grid_data = msg.data
        except Exception as e:
            rospy.logwarn("drivable_grid_callback error: %s", str(e))

    def risk_grid_callback(self, msg):
        try:
            self.risk_grid_resolution = float(msg.info.resolution)
            self.risk_grid_width = int(msg.info.width)
            self.risk_grid_height = int(msg.info.height)
            self.risk_grid_origin_x = float(msg.info.origin.position.x)
            self.risk_grid_origin_y = float(msg.info.origin.position.y)
            self.risk_grid_data = msg.data
        except Exception as e:
            rospy.logwarn("risk_grid_callback error: %s", str(e))

    def behavior_cmd_callback(self, msg):
        self.behavior_stop = bool(msg.stop)
        self.behavior_speed_limit = max(0.0, float(msg.speed_limit))
        self.behavior_reason = str(msg.reason)
        self._publish_emergency_stop_state()

    def path_mode_callback(self, msg):
        mode = str(getattr(msg, "data", "") or "").strip().lower()
        if not mode:
            mode = "hold"
        now_sec = rospy.Time.now().to_sec()
        if mode == self.current_path_mode:
            if mode == "hold" and self._hold_mode_enter_sec <= 0.0:
                self._hold_mode_enter_sec = now_sec
            return
        self.current_path_mode = mode
        if mode == "hold":
            self._hold_mode_enter_sec = now_sec
        else:
            self._hold_mode_enter_sec = 0.0

    def _log_nav_reason(self, reason, msg, warn=False):
        if not self.debug_stop_logging:
            return
        now_sec = rospy.Time.now().to_sec()
        if reason == self._last_nav_reason and (now_sec - self._last_nav_log_sec) < self.stop_log_period_s:
            return
        self._last_nav_reason = reason
        self._last_nav_log_sec = now_sec
        text = "[nav_reason] %s | %s" % (reason, msg)
        if warn:
            rospy.logwarn(text)
        else:
            rospy.loginfo(text)

    def _finish_reverse_recovery(self, reason, apply_pause=True):
        now = rospy.Time.now()
        self._reverse_recovery_active = False
        self._reverse_recovery_start_time = rospy.Time(0)
        self._reverse_recovery_start_xy = None
        self._reverse_recovery_trigger_since_sec = 0.0
        self._reverse_recovery_last_exit_reason = str(reason or "")
        self._reverse_recovery_resume_distance_m = self.reverse_recovery_distance_m
        self._reverse_recovery_required_front_clearance_m = (
            self.emergency_stop_distance + self.reverse_recovery_clearance_margin_m
        )
        self._reverse_recovery_cooldown_until = now + rospy.Duration(
            self.reverse_recovery_cooldown_s
        )
        if apply_pause and self.reverse_recovery_pause_s > 0.0:
            self._reverse_recovery_pause_until = now + rospy.Duration(
                self.reverse_recovery_pause_s
            )
        else:
            self._reverse_recovery_pause_until = rospy.Time(0)

    def _reverse_recovery_distance_traveled(self, pose_x, pose_y):
        if self._reverse_recovery_start_xy is None:
            return 0.0
        start_x, start_y = self._reverse_recovery_start_xy
        return math.hypot(float(pose_x) - start_x, float(pose_y) - start_y)

    def _compute_reverse_recovery_targets(self):
        required_front_clearance = max(
            self.emergency_stop_distance + self.reverse_recovery_clearance_margin_m,
            self.avoidance_hard_stop_distance + self.reverse_recovery_clearance_margin_m,
        )
        resume_distance = self.reverse_recovery_distance_m
        if math.isfinite(self.front_obstacle_clearance):
            deficit = required_front_clearance - self.front_obstacle_clearance
            if deficit > 0.0:
                resume_distance = max(resume_distance, deficit)
        resume_distance = min(
            max(self.reverse_recovery_distance_m, resume_distance),
            self.reverse_recovery_max_distance_m,
        )
        return resume_distance, required_front_clearance

    def _reverse_recovery_rear_points_blocked(self):
        if self.reverse_recovery_rear_local_points.size == 0:
            return False, 0
        rear_limit_x = -(
            self.robot_half_length_m + self.footprint_padding_m
        )
        check_min_x = rear_limit_x - self.reverse_recovery_rear_check_distance_m
        half_width = self.reverse_recovery_rear_half_width_m
        pts = self.reverse_recovery_rear_local_points
        mask = (
            (pts[:, 0] >= check_min_x)
            & (pts[:, 0] <= rear_limit_x + 0.05)
            & (np.abs(pts[:, 1]) <= half_width)
        )
        count = int(np.count_nonzero(mask))
        return count >= self.reverse_recovery_rear_min_points, count

    def _reverse_recovery_rear_drivable_ok(self, pose_x, pose_y, yaw):
        if not self.use_drivable_grid:
            return True
        rear_extent = (
            self.robot_half_length_m
            + self.footprint_padding_m
            + self.reverse_recovery_rear_drivable_margin_m
        )
        side_extent = self.robot_half_width_m + self.footprint_padding_m
        samples = (
            (-rear_extent, 0.0),
            (-rear_extent - 0.5 * self.reverse_recovery_distance_m, 0.0),
            (-rear_extent - self.reverse_recovery_distance_m, 0.0),
            (-rear_extent - self.reverse_recovery_distance_m, 0.8 * side_extent),
            (-rear_extent - self.reverse_recovery_distance_m, -0.8 * side_extent),
        )
        for lx, ly in samples:
            wx, wy = self._transform_local_to_world(lx, ly, pose_x, pose_y, yaw)
            if not self._is_xy_drivable_grid_ok(wx, wy):
                return False
        return True

    def _reverse_recovery_rear_is_safe(self, pose_x, pose_y, yaw):
        blocked, rear_count = self._reverse_recovery_rear_points_blocked()
        if blocked:
            return False, "rear_points=%d" % rear_count
        if not self._reverse_recovery_rear_drivable_ok(pose_x, pose_y, yaw):
            return False, "rear_drivable=false"
        return True, "rear_clear"

    def _reverse_recovery_forward_resume_ready(self, elapsed_s=0.0, traveled_m=0.0):
        if elapsed_s < self.reverse_recovery_min_active_time_s:
            return False
        required_travel = max(
            self.reverse_recovery_min_distance_before_resume_m,
            self._reverse_recovery_resume_distance_m,
        )
        if traveled_m < required_travel:
            return False
        if (
            math.isfinite(self.front_obstacle_clearance)
            and self.front_obstacle_clearance
            < self._reverse_recovery_required_front_clearance_m
        ):
            return False
        if self.current_path_mode == "hold":
            return False
        if self.active_path_source != "local":
            return False
        if len(self.path_pts) < 2:
            return False
        if self.enable_emergency_stop and self.emergency_blocked:
            return False
        return True

    def _handle_reverse_recovery(self, pose_x, pose_y, yaw):
        if not self.enable_reverse_recovery or self.follow_global_path_only:
            return False

        now = rospy.Time.now()
        if self._reverse_recovery_active:
            elapsed = (now - self._reverse_recovery_start_time).to_sec()
            traveled = self._reverse_recovery_distance_traveled(pose_x, pose_y)
            if self.behavior_stop:
                self._finish_reverse_recovery("hard_stop", apply_pause=True)
                self._log_nav_reason(
                    "reverse_abort",
                    "abort reverse recovery: behavior_stop=%s" % (
                        self.behavior_stop,
                    ),
                    warn=True,
                )
                self.publish_drive([0.0, 0.0])
                return True

            if self._reverse_recovery_forward_resume_ready(elapsed, traveled):
                self._finish_reverse_recovery("forward_path_ready", apply_pause=False)
                self._log_nav_reason(
                    "reverse_done",
                    "resume forward: src=%s mode=%s clr=%.2f" % (
                        self.active_path_source,
                        self.current_path_mode,
                        self.front_obstacle_clearance
                        if math.isfinite(self.front_obstacle_clearance)
                        else float("inf"),
                    ),
                )
                self.publish_drive([0.0, 0.0])
                return True

            rear_safe, rear_reason = self._reverse_recovery_rear_is_safe(
                pose_x, pose_y, yaw
            )
            if not rear_safe:
                self._finish_reverse_recovery(rear_reason, apply_pause=True)
                self._log_nav_reason(
                    "reverse_abort",
                    "rear no longer safe: %s" % rear_reason,
                    warn=True,
                )
                self.publish_drive([0.0, 0.0])
                return True

            if traveled >= self.reverse_recovery_max_distance_m:
                self._finish_reverse_recovery("max_distance_reached", apply_pause=True)
                self._log_nav_reason(
                    "reverse_done",
                    "distance=%.2f/%.2f clr=%.2f req=%.2f" % (
                        traveled,
                        self.reverse_recovery_max_distance_m,
                        self.front_obstacle_clearance
                        if math.isfinite(self.front_obstacle_clearance)
                        else float("inf"),
                        self._reverse_recovery_required_front_clearance_m,
                    ),
                )
                self.publish_drive([0.0, 0.0])
                return True
            if elapsed >= self.reverse_recovery_timeout_s:
                self._finish_reverse_recovery("timeout", apply_pause=True)
                self._log_nav_reason(
                    "reverse_done",
                    "timeout=%.2fs distance=%.2f" % (elapsed, traveled),
                    warn=True,
                )
                self.publish_drive([0.0, 0.0])
                return True

            self._rot_mode = False
            self._log_nav_reason(
                "reverse_recovery",
                "mode=%s dist=%.2f/%.2f speed=%.2f clr=%.2f req=%.2f" % (
                    self.current_path_mode,
                    traveled,
                    self._reverse_recovery_resume_distance_m,
                    self.reverse_recovery_speed_mps,
                    self.front_obstacle_clearance
                    if math.isfinite(self.front_obstacle_clearance)
                    else float("inf"),
                    self._reverse_recovery_required_front_clearance_m,
                ),
            )
            self.publish_drive([-self.reverse_recovery_speed_mps, 0.0])
            return True

        if now < self._reverse_recovery_pause_until:
            remaining = (self._reverse_recovery_pause_until - now).to_sec()
            self._log_nav_reason(
                "reverse_pause",
                "pause %.2fs after %s" % (
                    remaining,
                    self._reverse_recovery_last_exit_reason or "reverse_done",
                ),
            )
            self.publish_drive([0.0, 0.0])
            return True

        if now < self._reverse_recovery_cooldown_until:
            return False
        if self.behavior_stop or self._rot_mode:
            self._reverse_recovery_trigger_since_sec = 0.0
            return False

        trigger_hold = self.current_path_mode == "hold"
        trigger_emergency = bool(self.enable_emergency_stop and self.emergency_blocked)
        trigger_no_valid = (
            self._no_valid_traj_since_sec > 0.0
            and (now.to_sec() - self._no_valid_traj_since_sec) >= self.reverse_recovery_hold_trigger_s
            and (not self._near_goal_stuck_completion_allowed(now.to_sec()))
        )
        if not (trigger_hold or trigger_emergency or trigger_no_valid):
            self._reverse_recovery_trigger_since_sec = 0.0
            return False

        if self._reverse_recovery_trigger_since_sec <= 0.0:
            self._reverse_recovery_trigger_since_sec = now.to_sec()

        held_s = max(0.0, now.to_sec() - self._reverse_recovery_trigger_since_sec)
        if trigger_hold and self._hold_mode_enter_sec > 0.0:
            held_s = max(held_s, now.to_sec() - self._hold_mode_enter_sec)
        required_hold_s = self.reverse_recovery_hold_trigger_s
        if trigger_emergency:
            required_hold_s = min(
                required_hold_s, self.reverse_recovery_emergency_trigger_s
            )
        if held_s < required_hold_s:
            return False

        rear_safe, rear_reason = self._reverse_recovery_rear_is_safe(
            pose_x, pose_y, yaw
        )
        if not rear_safe:
            log_now = now.to_sec()
            if (log_now - self._reverse_recovery_last_log_sec) >= self.stop_log_period_s:
                self._reverse_recovery_last_log_sec = log_now
                self._log_nav_reason(
                    "reverse_blocked",
                    "hold=%.2fs but rear not safe: %s" % (held_s, rear_reason),
                    warn=True,
                )
            return False

        self._reverse_recovery_active = True
        self._reverse_recovery_start_time = now
        self._reverse_recovery_start_xy = (float(pose_x), float(pose_y))
        self._reverse_recovery_last_exit_reason = ""
        self._reverse_recovery_pause_until = rospy.Time(0)
        (
            self._reverse_recovery_resume_distance_m,
            self._reverse_recovery_required_front_clearance_m,
        ) = self._compute_reverse_recovery_targets()
        self._rot_mode = False
        trigger_label = []
        if trigger_hold:
            trigger_label.append("hold")
        if trigger_emergency:
            trigger_label.append("emergency")
        if trigger_no_valid:
            trigger_label.append("no_valid")
        self._log_nav_reason(
            "reverse_start",
            "trigger=%s held=%.2fs dist=%.2f speed=%.2f clr=%.2f req=%.2f max=%.2f" % (
                "+".join(trigger_label) if trigger_label else "unknown",
                held_s,
                self._reverse_recovery_resume_distance_m,
                self.reverse_recovery_speed_mps,
                self.front_obstacle_clearance
                if math.isfinite(self.front_obstacle_clearance)
                else float("inf"),
                self._reverse_recovery_required_front_clearance_m,
                self.reverse_recovery_max_distance_m,
            ),
            warn=True,
        )
        self.publish_drive([-self.reverse_recovery_speed_mps, 0.0])
        return True

    # ------------------------------- rotate-only --------------------------------
    def rotate_only_enter(self, cur_yaw, desired_yaw):
        self._rot_mode = True
        self._rot_yaw_target = desired_yaw
        self._rot_prev_yaw = cur_yaw
        self._rot_accum = 0.0
        self._rot_ok = 0
        self._rot_start_time = rospy.Time.now()
        rospy.loginfo("Rotate-only ENTER: target %.1f°", math.degrees(self._rot_yaw_target))

    def rotate_only_step(self, cur_yaw):
        dyaw = angdiff(cur_yaw, self._rot_prev_yaw)
        self._rot_accum += abs(dyaw)
        self._rot_prev_yaw = cur_yaw
        err = angdiff(self._rot_yaw_target, cur_yaw)
        w_cmd = max(-self._ROT_WMAX, min(self._ROT_WMAX, self.rotate_kp * err))
        u = [0.0, w_cmd]
        if abs(err) < self._ROT_LOW:
            self._rot_ok += 1
        else:
            self._rot_ok = 0
        now = rospy.Time.now()
        time_in = (now - self._rot_start_time).to_sec()
        exit_reason = None
        if self._rot_ok >= self.rotate_ok_count:
            exit_reason = "aligned"
        elif self._rot_accum > math.radians(self.rotate_max_spin_deg):
            exit_reason = "spin_limit"
        elif time_in > self.rotate_max_time_s:
            exit_reason = "timeout"
        if exit_reason is not None:
            self._rot_mode = False
            self._rot_last_timeout_target = self._rot_yaw_target
            self._rot_cooldown_until = now + rospy.Duration(self.rotate_reentry_cooldown_s)
            rospy.loginfo("Rotate-only EXIT: err=%.1f°, accum=%.1f°, t=%.1fs",
                          math.degrees(err), math.degrees(self._rot_accum), time_in)
            return None, True, exit_reason
        return u, False, None

    def _rotate_only_allowed(self, dist_to_goal, obstacle_response_active):
        if dist_to_goal <= self.near_goal_no_rotate_m:
            return False
        if obstacle_response_active:
            return False
        if self.active_path_source == "local":
            return False
        return True

    # ------------------------------ path handling --------------------------------
    def _path_signature(self, path_msg, include_start=False):
        if not path_msg or not path_msg.poses:
            return None
        n = len(path_msg.poses)
        p0 = path_msg.poses[0 if include_start else min(1, n - 1)].pose.position
        p_mid = path_msg.poses[n // 2].pose.position
        p1 = path_msg.poses[-1].pose.position
        if include_start:
            start_res = self.local_path_signature_start_resolution_m
            p0_x = round(round(float(p0.x) / start_res) * start_res, 3)
            p0_y = round(round(float(p0.y) / start_res) * start_res, 3)
        else:
            p0_x = round(p0.x, 3)
            p0_y = round(p0.y, 3)
        return (n,
                p0_x, p0_y,
                round(p_mid.x, 3), round(p_mid.y, 3),
                round(p1.x, 3), round(p1.y, 3))

    def _build_geometry_from_path_msg(self, path_msg):
        if path_msg is None or len(path_msg.poses) < 2:
            return [], [], [0.0], 0.0

        path_pts = []
        for ps in path_msg.poses:
            p = ps.pose.position
            path_pts.append((float(p.x), float(p.y)))
        if self._should_smooth_tracking_path():
            path_pts = self._smooth_tracking_polyline(path_pts)

        seg_lens = []
        cum_len = [0.0]
        s_total = 0.0
        for i in range(len(path_pts) - 1):
            dx = path_pts[i + 1][0] - path_pts[i][0]
            dy = path_pts[i + 1][1] - path_pts[i][1]
            seg_len = math.hypot(dx, dy)
            seg_lens.append(seg_len)
            s_total += seg_len
            cum_len.append(s_total)
        return path_pts, seg_lens, cum_len, s_total

    @staticmethod
    def _goal_xy_from_path_msg(path_msg):
        if path_msg is None or not path_msg.poses:
            return None
        goal = path_msg.poses[-1].pose.position
        return (float(goal.x), float(goal.y))

    def _navigation_goal_xy(self):
        if self.global_goal_xy is not None:
            return self.global_goal_xy
        return self._goal_xy_from_path_msg(self.path_msg)

    def _distance_to_navigation_goal(self, x, y, active_goal_xy):
        if self.active_path_source == "local":
            nav_goal_xy = self._navigation_goal_xy()
            if nav_goal_xy is not None:
                return math.hypot(nav_goal_xy[0] - float(x), nav_goal_xy[1] - float(y))
        return math.hypot(active_goal_xy[0] - float(x), active_goal_xy[1] - float(y))

    def _should_preserve_goal_reached_on_path_refresh(self, next_path_msg):
        if not self.reach_goal_flag:
            return False
        if self.active_path_source == "local":
            return False
        current_goal_xy = self._goal_xy_from_path_msg(self.path_msg)
        next_goal_xy = self._goal_xy_from_path_msg(next_path_msg)
        if current_goal_xy is None or next_goal_xy is None:
            return False

        completion_window = max(self.goal_thresh_m, self.path_tracking_stop_distance_m)
        goal_refresh_tolerance = max(
            completion_window,
            self.path_tracking_minor_replan_delta_m,
        )
        if (
            math.hypot(
                next_goal_xy[0] - current_goal_xy[0],
                next_goal_xy[1] - current_goal_xy[1],
            )
            > goal_refresh_tolerance
        ):
            return False

        pose = self.current_pose.pose.pose.position
        return (
            math.hypot(
                float(pose.x) - next_goal_xy[0],
                float(pose.y) - next_goal_xy[1],
            )
            <= completion_window
        )

    def _interp_geometry_xy_at_s(self, path_pts, seg_lens, cum_len, s_total, s):
        if len(path_pts) < 2:
            return 0.0, 0.0
        if s <= 0.0:
            return float(path_pts[0][0]), float(path_pts[0][1])
        if s >= s_total:
            return float(path_pts[-1][0]), float(path_pts[-1][1])

        i = 0
        while i < len(seg_lens) and cum_len[i + 1] < s:
            i += 1
        seg_len = seg_lens[i]
        ds = s - cum_len[i]
        t = 0.0 if seg_len < 1e-9 else (ds / seg_len)
        x0, y0 = path_pts[i]
        x1, y1 = path_pts[i + 1]
        return (
            float(x0 + t * (x1 - x0)),
            float(y0 + t * (y1 - y0)),
        )

    def _incoming_path_requires_activation(self, path_msg, sig, source):
        if source != self.active_path_source:
            return True
        if path_msg is None or self.path_msg is None:
            return True
        if source == "local":
            return self.path_sig != sig
        if self.path_sig == sig:
            return False
        if (
            self.path_tracking_minor_replan_delta_m <= 0.0
            or len(self.path_pts) < 2
            or not path_msg.poses
            or not self.path_msg.poses
        ):
            return True

        new_goal = path_msg.poses[-1].pose.position
        old_goal = self.path_msg.poses[-1].pose.position
        goal_delta = math.hypot(
            float(new_goal.x) - float(old_goal.x),
            float(new_goal.y) - float(old_goal.y),
        )
        if goal_delta > self.path_tracking_reset_goal_delta_m:
            return True

        self._sync_progress_to_current_pose()
        new_pts, new_seg_lens, new_cum_len, new_s_total = self._build_geometry_from_path_msg(path_msg)
        if len(new_pts) < 2:
            return True

        sample_offsets = []
        for d in (1.0, 2.5, 4.0):
            if self.s_cur + d <= self.s_total - 0.05 and d <= new_s_total - 0.05:
                sample_offsets.append(d)
        if not sample_offsets:
            self.path_sig = sig
            return False

        for d in sample_offsets:
            cur_x, cur_y = self._interp_geometry_xy_at_s(
                self.path_pts,
                self.seg_lens,
                self.cum_len,
                self.s_total,
                self.s_cur + d,
            )
            new_x, new_y = self._interp_geometry_xy_at_s(
                new_pts,
                new_seg_lens,
                new_cum_len,
                new_s_total,
                d,
            )
            if math.hypot(cur_x - new_x, cur_y - new_y) > self.path_tracking_minor_replan_delta_m:
                return True

        self.path_sig = sig
        return False

    def _rebuild_path_geometry(self):
        self.path_pts = []
        for ps in self.path_msg.poses:
            p = ps.pose.position
            self.path_pts.append((p.x, p.y))
        if self._should_smooth_tracking_path():
            self.path_pts = self._smooth_tracking_polyline(self.path_pts)
        # seg lengths & cumulative
        self.seg_lens = []
        self.cum_len = [0.0]
        s = 0.0
        for i in range(len(self.path_pts) - 1):
            dx = self.path_pts[i + 1][0] - self.path_pts[i][0]
            dy = self.path_pts[i + 1][1] - self.path_pts[i][1]
            L = math.hypot(dx, dy)
            self.seg_lens.append(L)
            s += L
            self.cum_len.append(s)
        self.s_total = s
        self.s_cur = 0.0

    def _should_smooth_tracking_path(self):
        if self.active_path_source == "local":
            return self.local_tracking_smoothing_enabled and (not self._local_stop_turn_go_active())
        return not self._local_stop_turn_go_active()

    def _local_stop_turn_go_active(self):
        return (
            self.local_stop_turn_go_enabled
            and self.active_path_source == "local"
            and len(self.path_pts) >= 2
        )

    def _clear_local_turn_rotate_state(self):
        self._local_turn_rotate_kind = None
        self._local_turn_rotate_seg_idx = -1
        self._local_turn_rotate_heading = None
        self._local_turn_rotate_advance_after = False

    def _segment_index_at_s(self, s_val):
        if not self.seg_lens:
            return 0
        idx = 0
        while idx + 1 < len(self.seg_lens) and self.cum_len[idx + 1] < (s_val - 1e-6):
            idx += 1
        return max(0, min(len(self.seg_lens) - 1, idx))

    def _segment_heading_tangent(self, seg_idx):
        if len(self.path_pts) < 2:
            return 0.0, (1.0, 0.0)
        seg_idx = max(0, min(len(self.path_pts) - 2, int(seg_idx)))
        x0, y0 = self.path_pts[seg_idx]
        x1, y1 = self.path_pts[seg_idx + 1]
        dx = float(x1) - float(x0)
        dy = float(y1) - float(y0)
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-9:
            return 0.0, (1.0, 0.0)
        return math.atan2(dy, dx), (dx / seg_len, dy / seg_len)

    def _step_local_turn_rotate(self, yaw):
        if self._local_turn_rotate_heading is None:
            return None
        desired_heading = float(self._local_turn_rotate_heading)
        yaw_err = angdiff(desired_heading, yaw)
        if abs(yaw_err) <= self.local_stop_turn_align_rad:
            if (
                self._local_turn_rotate_advance_after
                and 0 <= self._local_turn_rotate_seg_idx < len(self.cum_len)
            ):
                self.s_cur = max(
                    self.s_cur,
                    min(self.s_total, self.cum_len[self._local_turn_rotate_seg_idx] + 1e-3),
                )
            done_kind = self._local_turn_rotate_kind or "turn"
            self._clear_local_turn_rotate_state()
            return {
                "mode": "settle",
                "reason": done_kind,
                "desired_heading": desired_heading,
            }

        w_cmd = max(
            -self.path_tracking_in_place_yaw_rate_max,
            min(self.path_tracking_in_place_yaw_rate_max, self.rotate_kp * yaw_err),
        )
        return {
            "mode": "rotate",
            "kind": self._local_turn_rotate_kind or "turn",
            "desired_heading": desired_heading,
            "yaw_err": yaw_err,
            "cmd": [0.0, w_cmd],
        }

    def _local_stop_turn_go_control(self, pose_x, pose_y, yaw, v_cap):
        if not self._local_stop_turn_go_active():
            self._clear_local_turn_rotate_state()
            return None

        rotate_step = self._step_local_turn_rotate(yaw)
        if rotate_step is not None:
            return rotate_step

        seg_idx = self._segment_index_at_s(self.s_cur)
        seg_heading, seg_t_hat = self._segment_heading_tangent(seg_idx)
        heading_err = angdiff(seg_heading, yaw)
        if abs(heading_err) > self.local_stop_turn_align_rad:
            self._local_turn_rotate_kind = "align"
            self._local_turn_rotate_seg_idx = seg_idx
            self._local_turn_rotate_heading = seg_heading
            self._local_turn_rotate_advance_after = False
            return self._step_local_turn_rotate(yaw)

        seg_end_s = self.cum_len[seg_idx + 1]
        seg_end_xy = (
            float(self.path_pts[seg_idx + 1][0]),
            float(self.path_pts[seg_idx + 1][1]),
        )
        segment_remaining = max(0.0, seg_end_s - self.s_cur)
        point_remaining = math.hypot(
            seg_end_xy[0] - float(pose_x),
            seg_end_xy[1] - float(pose_y),
        )
        corner_arrival = min(segment_remaining, point_remaining)

        if seg_idx + 1 < len(self.seg_lens):
            next_heading, _next_t_hat = self._segment_heading_tangent(seg_idx + 1)
            turn_angle = abs(angdiff(next_heading, seg_heading))
            if (
                turn_angle >= self.local_stop_turn_corner_trigger_rad
                and corner_arrival <= self.local_stop_turn_corner_arrival_m
            ):
                self._local_turn_rotate_kind = "corner"
                self._local_turn_rotate_seg_idx = seg_idx + 1
                self._local_turn_rotate_heading = next_heading
                self._local_turn_rotate_advance_after = True
                return self._step_local_turn_rotate(yaw)

        return {
            "mode": "drive",
            "seg_idx": seg_idx,
            "target_xy": seg_end_xy,
            "t_hat": seg_t_hat,
            "remaining_dist": max(segment_remaining, point_remaining),
            "v_cap": min(v_cap, self.local_stop_turn_speed_cap),
        }

    def _smooth_tracking_polyline(self, points):
        if len(points) < 3 or self.tracking_path_smoothing_passes <= 0:
            return list(points)

        smoothed = [(float(px), float(py)) for px, py in points]
        for _ in range(self.tracking_path_smoothing_passes):
            if len(smoothed) < 3:
                break
            next_pts = [smoothed[0]]
            for i in range(1, len(smoothed) - 1):
                x_prev, y_prev = smoothed[i - 1]
                x_cur, y_cur = smoothed[i]
                x_next, y_next = smoothed[i + 1]
                next_pts.append(
                    (
                        0.25 * x_prev + 0.50 * x_cur + 0.25 * x_next,
                        0.25 * y_prev + 0.50 * y_cur + 0.25 * y_next,
                    )
                )
            next_pts.append(smoothed[-1])
            smoothed = next_pts
        return smoothed

    def _sync_progress_to_current_pose(self):
        if len(self.path_pts) < 2:
            self.s_cur = 0.0
            return
        px, py = self._tracking_anchor_xy()
        s_proj, _lat_err, _idx, _t = self._project_to_path(px, py)
        self.s_cur = max(0.0, min(self.s_total, s_proj))

    def _empty_tracking_reference_path(self):
        msg = Path()
        if self.path_msg is not None:
            msg.header = self.path_msg.header
        else:
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "map"
        msg.header.stamp = rospy.Time.now()
        return msg

    def _tracking_anchor_xy(self):
        pose = self.current_pose.pose.pose.position
        yaw = self.get_yaw_from_quaternion(self.current_pose.pose.pose.orientation)
        front_offset = self.robot_half_length_m + self.footprint_padding_m
        return (
            float(pose.x) + math.cos(yaw) * front_offset,
            float(pose.y) + math.sin(yaw) * front_offset,
        )

    def _build_tracking_reference_path_msg(self):
        if len(self.path_pts) < 2:
            return self._empty_tracking_reference_path()

        msg = Path()
        msg.header = self.path_msg.header if self.path_msg is not None else self._empty_tracking_reference_path().header
        msg.header.stamp = rospy.Time.now()

        try:
            px, py = self._tracking_anchor_xy()
            s_min = max(0.0, self.s_cur - self.tracking_projection_back_window_m)
            s_max = min(self.s_total, self.s_cur + self.tracking_projection_forward_window_m)
            s_proj, _lat_err, _idx, _t = self._project_to_path(
                px,
                py,
                s_min=s_min,
                s_max=s_max,
            )
            start_s = max(0.0, min(self.s_total, max(self.s_cur, s_proj)))
        except Exception:
            start_s = max(0.0, min(self.s_total, self.s_cur))

        start_x, start_y, _ = self._interp_xy_tangent_at_s(start_s)
        ref_points = [(float(start_x), float(start_y))]
        for idx, pt in enumerate(self.path_pts[1:], start=1):
            if self.cum_len[idx] > start_s + 1e-3:
                ref_points.append((float(pt[0]), float(pt[1])))
        if len(ref_points) == 1:
            ref_points.append((float(self.path_pts[-1][0]), float(self.path_pts[-1][1])))

        deduped = []
        prev = None
        for pt in ref_points:
            if prev is None or math.hypot(pt[0] - prev[0], pt[1] - prev[1]) > 1e-4:
                deduped.append(pt)
                prev = pt

        for x, y in deduped:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        return msg

    def _publish_tracking_reference_path(self):
        if rospy.is_shutdown():
            return
        try:
            if len(self.path_pts) < 2:
                self.tracking_reference_pub.publish(self._empty_tracking_reference_path())
            else:
                self.tracking_reference_pub.publish(self._build_tracking_reference_path_msg())
        except rospy.ROSException:
            pass

    def _activate_path(self, path_msg, sig, source):
        path_changed = sig != self.path_sig
        goal_changed = False
        preserve_goal_reached = self._should_preserve_goal_reached_on_path_refresh(
            path_msg
        )
        if (
            path_changed
            and source == self.active_path_source
            and path_msg is not None
            and self.path_msg is not None
            and self.path_tracking_reset_goal_delta_m > 0.0
            and path_msg.poses
            and self.path_msg.poses
        ):
            new_goal = path_msg.poses[-1].pose.position
            old_goal = self.path_msg.poses[-1].pose.position
            goal_changed = (
                math.hypot(float(new_goal.x) - float(old_goal.x), float(new_goal.y) - float(old_goal.y))
                > self.path_tracking_reset_goal_delta_m
            )
        rolling_local_refresh = (
            source == "local"
            and self.active_path_source == "local"
            and path_msg is not None
            and self.path_msg is not None
            and not goal_changed
        )
        reset_tracking = (
            source != self.active_path_source
            or (sig != self.path_sig and not rolling_local_refresh)
            or path_msg is None
            or self.path_msg is None
            or goal_changed
        )
        self.path_sig = sig
        self.path_msg = path_msg
        self.active_path_source = source
        if reset_tracking:
            self._path_tracking_prev_w = 0.0
            self._path_tracking_prev_desired_yaw = None
            self._path_tracking_filtered_lat_err = 0.0
            self._last_tracking_debug = {}
            self._clear_local_turn_rotate_state()
            self._rot_mode = False
            self._rot_yaw_target = None
            if preserve_goal_reached:
                self.reach_goal_flag = True
                self.prev_goal_flag = True
            else:
                self.reach_goal_flag = False
                self.prev_goal_flag = False
        if path_msg is None or len(path_msg.poses) < 2:
            self._clear_local_turn_rotate_state()
            self.path_pts = []
            self.seg_lens = []
            self.cum_len = [0.0]
            self.s_total = 0.0
            self.s_cur = 0.0
            self._publish_tracking_reference_path()
            return
        self._rebuild_path_geometry()
        self._sync_progress_to_current_pose()
        self._publish_tracking_reference_path()

    def path_callback_global(self, path_msg):
        self.global_path_msg = path_msg
        self.global_path_sig = self._path_signature(path_msg)
        self.global_goal_xy = self._goal_xy_from_path_msg(path_msg)

    def path_callback_local(self, path_msg):
        now = rospy.Time.now()
        self.local_path_msg = path_msg
        self.local_path_sig = self._path_signature(path_msg, include_start=True)
        self.local_path_stamp = now
        source_stamp = path_msg.header.stamp if path_msg is not None else rospy.Time(0)
        if source_stamp.to_sec() <= 0.0:
            source_stamp = now
        self.local_path_source_stamp = source_stamp

    def _local_path_is_fresh(self, now=None):
        if self.local_path_msg is None or len(self.local_path_msg.poses) < 2:
            return False
        if self.local_path_stamp.to_sec() <= 0.0:
            return False
        if now is None:
            now = rospy.Time.now()
        receipt_age_s = (now - self.local_path_stamp).to_sec()
        if receipt_age_s > self.local_path_timeout_s:
            return False
        if self.local_path_source_stamp.to_sec() <= 0.0:
            return True
        source_age_s = (now - self.local_path_source_stamp).to_sec()
        if not self.enforce_local_path_source_stamp:
            return True
        return source_age_s <= self.local_path_source_timeout_s

    def _stamp_age_s(self, stamp, now=None):
        if stamp is None or stamp.to_sec() <= 0.0:
            return -1.0
        if now is None:
            now = rospy.Time.now()
        return max(0.0, (now - stamp).to_sec())

    def _refresh_active_path(self):
        now = rospy.Time.now()
        if self.follow_global_path_only:
            if self.global_path_msg is not None and len(self.global_path_msg.poses) >= 2:
                if self._incoming_path_requires_activation(self.global_path_msg, self.global_path_sig, "global"):
                    self._activate_path(self.global_path_msg, self.global_path_sig, "global")
                return
            if self.path_msg is not None or self.active_path_source != "none":
                self._activate_path(None, None, "none")
            return

        if self.current_path_mode == "hold":
            # Keep the last active path latched while the replanner requests a
            # temporary hold.  Clearing the active path here makes the
            # tracking-reference path disappear and forces the controller into
            # the generic stop_no_path branch even though the stop reason is
            # really "hold".
            return

        use_local = self._local_path_is_fresh(now=now)

        # In the default navigation architecture, the controller always tracks
        # a short rolling local path.  The fixed global path is the guide; the
        # local path is the actual control reference and should be preferred
        # whenever it is fresh.
        if use_local:
            if self._incoming_path_requires_activation(self.local_path_msg, self.local_path_sig, "local"):
                self._activate_path(self.local_path_msg, self.local_path_sig, "local")
            return

        if self.current_path_mode in ("follow_local", "follow_avoidance", "rejoin_global"):
            # Once the replanner stops providing a fresh local control path,
            # do not keep chasing the last latched segment or fall back to the
            # long global guide.  That stale reference is what can trigger
            # rotate-only / local corner-rotate behavior with no valid current
            # path on screen.
            if self.path_msg is not None or self.active_path_source != "none":
                self._activate_path(None, None, "none")
            return
        else:
            if self.global_path_msg is not None and len(self.global_path_msg.poses) >= 2:
                if self._incoming_path_requires_activation(self.global_path_msg, self.global_path_sig, "global"):
                    self._activate_path(self.global_path_msg, self.global_path_sig, "global")
                return

        if self.path_msg is not None:
            self._activate_path(None, None, "none")

    def pose_callback(self, msg):
        self.current_pose = msg
        self._update_emergency_stop_state()

    def server_to_robot_callback(self, msg):
        self.server_cmd_drive_mode = msg.Cmd_drive_mode
        if msg.Cmd_use_vel_control:
            if (self.max_speed != msg.Cmd_linear_velocity or
                self.max_yaw_rate != msg.Cmd_angular_velocity):
                self.max_speed = msg.Cmd_linear_velocity
                self.max_yaw_rate = msg.Cmd_angular_velocity
                rospy.loginfo("Updated limits: max_speed=%.3f, max_yaw_rate=%.1fdeg/s",
                              self.max_speed, math.degrees(self.max_yaw_rate))

    # ------------------------------- pure pursuit --------------------------------
    def _project_to_path(self, x, y, s_min=None, s_max=None):
        """Return (s_proj, lateral_err, seg_idx, t) where s is arc-length along path."""
        if len(self.path_pts) < 2:
            return 0.0, 0.0, 0, 0.0
        best_d2 = 1e18
        best_i = 0
        best_t = 0.0
        best_px = self.path_pts[0][0]
        best_py = self.path_pts[0][1]
        for i in range(len(self.path_pts) - 1):
            seg_s0 = self.cum_len[i]
            seg_s1 = self.cum_len[i + 1]
            seg_len = self.seg_lens[i]
            if s_min is not None and seg_s1 < s_min:
                continue
            if s_max is not None and seg_s0 > s_max:
                continue
            x0, y0 = self.path_pts[i]
            x1, y1 = self.path_pts[i + 1]
            vx, vy = x1 - x0, y1 - y0
            denom = vx * vx + vy * vy
            if denom < 1e-12:
                t_lo = 0.0
                t_hi = 0.0
                px, py = x0, y0
            else:
                t_lo = 0.0
                t_hi = 1.0
                if seg_len > 1e-9:
                    if s_min is not None:
                        t_lo = max(t_lo, (s_min - seg_s0) / seg_len)
                    if s_max is not None:
                        t_hi = min(t_hi, (s_max - seg_s0) / seg_len)
                if t_lo > t_hi:
                    continue
                t = ((x - x0) * vx + (y - y0) * vy) / denom
                t = max(t_lo, min(t_hi, t))
                px, py = x0 + t * vx, y0 + t * vy
            d2 = (x - px) ** 2 + (y - py) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
                best_t = t
                best_px = px
                best_py = py
        if best_d2 >= 1e17 and (s_min is not None or s_max is not None):
            return self._project_to_path(x, y)
        # arc-length at projection
        s_at_i = self.cum_len[best_i]
        s_proj = s_at_i + best_t * self.seg_lens[best_i]
        x0, y0 = self.path_pts[best_i]
        x1, y1 = self.path_pts[best_i + 1]
        seg_dx = x1 - x0
        seg_dy = y1 - y0
        seg_len = math.hypot(seg_dx, seg_dy)
        if seg_len < 1e-9:
            lat_err = 0.0
        else:
            nx = -seg_dy / seg_len
            ny = seg_dx / seg_len
            lat_err = (x - best_px) * nx + (y - best_py) * ny
        return s_proj, lat_err, best_i, best_t

    def _interp_xy_tangent_at_s(self, s):
        if s <= 0.0 or len(self.path_pts) < 2:
            x, y = self.path_pts[0]
            dx = self.path_pts[1][0] - self.path_pts[0][0]
            dy = self.path_pts[1][1] - self.path_pts[0][1]
            L = math.hypot(dx, dy) + 1e-9
            return x, y, (dx / L, dy / L)
        if s >= self.s_total:
            x, y = self.path_pts[-1]
            dx = self.path_pts[-1][0] - self.path_pts[-2][0]
            dy = self.path_pts[-1][1] - self.path_pts[-2][1]
            L = math.hypot(dx, dy) + 1e-9
            return x, y, (dx / L, dy / L)
        # find segment
        i = 0
        while i < len(self.seg_lens) and self.cum_len[i + 1] < s:
            i += 1
        ds = s - self.cum_len[i]
        L = self.seg_lens[i]
        t = 0.0 if L < 1e-9 else (ds / L)
        x0, y0 = self.path_pts[i]
        x1, y1 = self.path_pts[i + 1]
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        dx = (x1 - x0)
        dy = (y1 - y0)
        L = math.hypot(dx, dy) + 1e-9
        return x, y, (dx / L, dy / L)

    def _interp_xy_smoothed_tangent_at_s(self, s):
        x, y, t_hat = self._interp_xy_tangent_at_s(s)
        if len(self.path_pts) < 3:
            return x, y, t_hat

        window = self.path_tracking_tangent_window_m
        if self.active_path_source == "local":
            window = min(window, self.local_tracking_tangent_window_m)
            if self.local_tracking_target_step_cap_m > 1e-3:
                window = min(window, 1.5 * self.local_tracking_target_step_cap_m)
        if window <= 1e-3:
            return x, y, t_hat

        s_back = max(0.0, min(self.s_total, s - 0.35 * window))
        s_fwd = max(0.0, min(self.s_total, s + 0.65 * window))
        if (s_fwd - s_back) < 1e-3:
            s_back = max(0.0, min(self.s_total, s - window))
            s_fwd = max(0.0, min(self.s_total, s))
        if (s_fwd - s_back) < 1e-3:
            return x, y, t_hat

        x0, y0, _ = self._interp_xy_tangent_at_s(s_back)
        x1, y1, _ = self._interp_xy_tangent_at_s(s_fwd)
        dx = x1 - x0
        dy = y1 - y0
        L = math.hypot(dx, dy)
        if L < 1e-6:
            return x, y, t_hat
        return x, y, (dx / L, dy / L)

    def _update_progress_and_target(self, pose_x, pose_y, yaw):
        if not self.path_pts:
            return None, None, None, None, False, None, None

        tracking_x, tracking_y = self._tracking_anchor_xy()

        obstacle_response_active = (
            str(self.behavior_reason).strip().lower() != "clear"
            or self._avoidance_mode_active()
            or (
                math.isfinite(self.front_obstacle_clearance)
                and self.front_obstacle_clearance <= self.obstacle_response_clearance_m
            )
        )

        s_min = max(0.0, self.s_cur - self.tracking_projection_back_window_m)
        s_max = min(self.s_total, self.s_cur + self.tracking_projection_forward_window_m)
        s_proj, lat_err, idx, t = self._project_to_path(
            tracking_x,
            tracking_y,
            s_min=s_min,
            s_max=s_max,
        )

        # enforce monotonic progress with tiny back jitter allowed
        if s_proj + self.back_jitter_m >= self.s_cur:
            self.s_cur = max(self.s_cur, s_proj)

        base_s = max(self.s_cur, s_proj)
        preview_lookahead_min = self.lookahead_distance
        if (
            obstacle_response_active
            and (not self.follow_global_path_only)
            and self.active_path_source == "local"
        ):
            preview_lookahead_min = max(
                self.lookahead_distance,
                self.lookahead_distance * self.obstacle_response_lookahead_scale,
            )
        if self.active_path_source == "local":
            preview_lookahead_min = min(
                preview_lookahead_min,
                max(0.25, self.local_tracking_preview_m),
            )
        preview_lookahead = preview_lookahead_min
        preview_lookahead += self.lookahead_speed_gain * max(0.0, abs(self.last_cmd.linear.x))
        preview_lookahead += self.lookahead_error_gain * abs(lat_err)
        preview_lookahead = max(
            preview_lookahead_min,
            min(self.lookahead_max_distance, preview_lookahead),
        )
        active_goal_xy = (
            float(self.path_pts[-1][0]),
            float(self.path_pts[-1][1]),
        )
        dist_to_goal = self._distance_to_navigation_goal(
            tracking_x,
            tracking_y,
            active_goal_xy,
        )
        goal_align_active = (
            min(max(0.0, self.s_total - base_s), dist_to_goal)
            <= self.path_tracking_goal_align_window_m
        )

        if goal_align_active:
            s_target = self.s_total
            target_policy = "goal_align"
        elif (
            self.follow_global_path_only
            or self.active_path_source == "global"
            or (
                self.active_path_source == "local"
                and self.current_path_mode == "follow_local"
                and not obstacle_response_active
            )
        ):
            if abs(lat_err) > self.snap_lat_err:
                s_target = min(
                    self.s_total,
                    base_s + max(preview_lookahead, self.snap_target_ahead_m),
                )
            else:
                s_target = min(self.s_total, base_s + preview_lookahead)
            target_policy = (
                "local_clear_preview"
                if self.active_path_source == "local"
                else "global_preview"
            )
        else:
            # Follow the currently selected path segment more directly instead
            # of skipping far ahead on a heavily smoothed global path.  This
            # keeps obstacle-avoidance detours from being cut across by the
            # controller when the planner intentionally routes around a tight
            # obstacle.
            target_policy = (
                "local_segment"
                if self.active_path_source == "local"
                else "global_segment"
            )
            target_seg_idx = idx
            if t >= 0.98 and target_seg_idx + 1 < len(self.seg_lens):
                target_seg_idx += 1
            segment_end_s = self.cum_len[target_seg_idx + 1]
            segment_step_scale = 0.65 if self.active_path_source == "global" else 0.95
            segment_target_step_cap = max(0.05, self.path_tracking_target_step_m)
            if self.active_path_source == "local":
                segment_step_scale = min(
                    segment_step_scale,
                    self.local_tracking_segment_step_scale,
                )
                segment_target_step_cap = min(
                    segment_target_step_cap,
                    self.local_tracking_target_step_cap_m,
                )
            segment_target_step = min(
                segment_target_step_cap,
                max(0.05, segment_step_scale * self.seg_lens[target_seg_idx]),
            )
            s_target = min(
                self.s_total,
                min(segment_end_s, max(base_s, s_proj + segment_target_step)),
            )

        tx, ty, t_hat = self._interp_xy_smoothed_tangent_at_s(s_target)

        # Goal reach must use the actual distance to the path endpoint.  The
        # projection progress can jump near the end of a short/snapped path,
        # making arc_rem small while the robot is still physically far away.
        arc_rem = max(0.0, self.s_total - self.s_cur)
        at_goal = (dist_to_goal <= self.goal_thresh_m) and (abs(lat_err) <= self.lat_goal_slop)
        self._last_target_debug = {
            "mode": self.current_path_mode,
            "source": self.active_path_source,
            "policy": target_policy,
            "path_len": len(self.path_pts),
            "s_proj": s_proj,
            "s_cur": self.s_cur,
            "s_target": s_target,
            "s_total": self.s_total,
            "preview": preview_lookahead,
            "lat_err": lat_err,
            "goal_align": goal_align_active,
            "obstacle_response": obstacle_response_active,
        }

        return (s_proj, lat_err, (tx, ty), t_hat, at_goal, dist_to_goal, arc_rem)

    def _compute_emergency_bypass_state(
        self,
        pose_x,
        pose_y,
        yaw,
        s_proj,
        lat_err,
        target_xy,
        remaining_dist,
    ):
        # Let a planned detour keep moving only when the path clearly bends
        # around the obstacle and the obstacle is not yet in the hard-stop zone.
        self._emergency_bypass_debug = {}
        if not self.emergency_bypass_enabled:
            return False
        if not self.path_tracking_only or len(self.path_pts) < 2:
            return False
        if self.active_path_source == "none":
            return False
        if self._raw_immediate_contact_blocked:
            return False
        if not math.isfinite(self.front_obstacle_clearance):
            return False
        active_avoidance = self._avoidance_mode_active()
        min_bypass_clearance = (
            (
                self.active_avoidance_bypass_clearance_m
                if active_avoidance
                else self.avoidance_hard_stop_distance
            )
            + self.emergency_bypass_clearance_margin_m
        )
        if self.front_obstacle_clearance <= min_bypass_clearance:
            return False
        if remaining_dist <= max(self.goal_thresh_m, self.emergency_bypass_goal_window_m):
            return False

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        def _to_local(world_x, world_y):
            dx = float(world_x) - float(pose_x)
            dy = float(world_y) - float(pose_y)
            return (
                cos_yaw * dx + sin_yaw * dy,
                -sin_yaw * dx + cos_yaw * dy,
            )

        target_local_x, target_local_y = _to_local(target_xy[0], target_xy[1])
        base_s = max(self.s_cur, s_proj)
        preview_distance = self.emergency_bypass_preview_m * (1.35 if active_avoidance else 1.0)
        preview_s = min(self.s_total, base_s + preview_distance)
        preview_x, preview_y, preview_t_hat = self._interp_xy_smoothed_tangent_at_s(preview_s)
        preview_local_x, preview_local_y = _to_local(preview_x, preview_y)
        preview_path_yaw = math.atan2(preview_t_hat[1], preview_t_hat[0])
        preview_yaw_err = angdiff(preview_path_yaw, yaw)
        preview_bearing = math.atan2(
            preview_local_y,
            preview_local_x if preview_local_x > 1e-6 else 1e-6,
        )

        self._emergency_bypass_debug = {
            "preview_local_x": preview_local_x,
            "preview_local_y": preview_local_y,
            "preview_bearing_deg": math.degrees(preview_bearing),
            "preview_yaw_err_deg": math.degrees(preview_yaw_err),
            "target_local_x": target_local_x,
            "target_local_y": target_local_y,
        }

        if preview_local_x <= 0.05:
            return False

        lateral_detour = max(abs(target_local_y), abs(preview_local_y))
        lateral_threshold = self.emergency_bypass_target_lateral_m
        bearing_threshold = self.emergency_bypass_bearing_rad
        yaw_err_threshold = self.emergency_bypass_yaw_err_rad
        if active_avoidance:
            lateral_threshold = max(0.03, lateral_threshold * 0.75)
            bearing_threshold = max(math.radians(2.0), bearing_threshold * 0.75)
            yaw_err_threshold = max(math.radians(3.0), yaw_err_threshold * 0.75)
        return (
            lateral_detour >= lateral_threshold
            or abs(preview_bearing) >= bearing_threshold
            or abs(preview_yaw_err) >= yaw_err_threshold
        )

    def _near_goal_completion_allowed(self, dist_to_goal, arc_rem, lat_err):
        if self.enable_emergency_stop and self.emergency_blocked:
            return False
        if str(self.behavior_reason).strip().lower() != "clear":
            return False
        completion_window = max(self.goal_thresh_m, self.path_tracking_stop_distance_m)
        if dist_to_goal > completion_window:
            return False
        if arc_rem > completion_window:
            return False
        if abs(lat_err) > self.lat_goal_slop:
            return False
        return True

    def _near_goal_stuck_completion_allowed(self, now_sec):
        if self._no_valid_traj_since_sec <= 0.0:
            return False
        if self.enable_emergency_stop and self.emergency_blocked:
            return False
        if str(self.behavior_reason).strip().lower() != "clear":
            return False
        if (now_sec - self._no_valid_traj_since_sec) < self.goal_completion_stuck_hold_s:
            return False
        if self._no_valid_traj_dist_m > self.goal_completion_stuck_distance_m:
            return False
        if self._no_valid_traj_arc_m > self.goal_completion_stuck_arc_m:
            return False
        if abs(self._no_valid_traj_lat_m) > self.lat_goal_slop:
            return False
        return True

    def _clear_no_valid_traj_state(self):
        self._no_valid_traj_since_sec = 0.0
        self._no_valid_traj_dist_m = float("inf")
        self._no_valid_traj_arc_m = float("inf")
        self._no_valid_traj_lat_m = float("inf")

    # ------------------------------- dwa core ------------------------------------
    def dwa_control(self, x, goal_xy, t_hat, lat_err):
        dw = self.calc_dynamic_window(x)
        u, trajectory = self.calc_control_and_trajectory(x, dw, goal_xy, t_hat)
        return u, trajectory

    def path_tracking_control(self, x, goal_xy, t_hat, lat_err, v_cap, remaining_dist):
        obstacle_response_active = (
            str(self.behavior_reason).strip().lower() != "clear"
            or self._emergency_bypass_active
            or self._avoidance_mode_active()
            or (
                math.isfinite(self.front_obstacle_clearance)
                and self.front_obstacle_clearance <= self.obstacle_response_clearance_m
            )
        )
        path_yaw_raw = math.atan2(t_hat[1], t_hat[0])
        if abs(lat_err) <= self.path_tracking_cte_deadband_m:
            lat_err_ctrl = 0.0
        else:
            lat_err_ctrl = math.copysign(
                abs(lat_err) - self.path_tracking_cte_deadband_m,
                lat_err,
            )
        self._path_tracking_filtered_lat_err = (
            self.path_tracking_cte_filter_gain * lat_err_ctrl
            + (1.0 - self.path_tracking_cte_filter_gain) * self._path_tracking_filtered_lat_err
        )
        target_dx = float(goal_xy[0]) - float(x[0])
        target_dy = float(goal_xy[1]) - float(x[1])
        target_dist = math.hypot(target_dx, target_dy)
        target_point_yaw = path_yaw_raw if target_dist <= 1e-6 else math.atan2(target_dy, target_dx)
        goal_bearing_err = angdiff(target_point_yaw, path_yaw_raw)
        goal_bearing_err = max(
            -self.path_tracking_goal_bearing_cap,
            min(self.path_tracking_goal_bearing_cap, goal_bearing_err),
        )
        goal_align_ratio = 0.0
        if self.path_tracking_goal_align_window_m <= self.goal_thresh_m + 1e-6:
            if remaining_dist <= self.path_tracking_goal_align_window_m:
                goal_align_ratio = 1.0
        else:
            goal_align_ratio = 1.0 - (
                (remaining_dist - self.goal_thresh_m)
                / max(
                    1e-6,
                    self.path_tracking_goal_align_window_m - self.goal_thresh_m,
                )
            )
            goal_align_ratio = max(0.0, min(1.0, goal_align_ratio))
        goal_bearing_gain = self.path_tracking_goal_bearing_gain
        if self.active_path_source == "local":
            goal_bearing_gain *= 0.20
        elif obstacle_response_active:
            goal_bearing_gain *= 1.20
        goal_heading_weight = 0.0
        if remaining_dist <= max(self.lookahead_distance, 0.8):
            goal_heading_weight = min(0.28 if obstacle_response_active else 0.20, goal_bearing_gain)
        goal_heading_weight = max(goal_heading_weight, goal_align_ratio)
        cte_correction = math.atan2(
            self.path_tracking_cte_gain * self._path_tracking_filtered_lat_err,
            self.path_tracking_cte_soft_mps + max(0.0, abs(x[3])),
        )
        cte_correction = max(
            -self.path_tracking_cte_yaw_cap,
            min(self.path_tracking_cte_yaw_cap, cte_correction),
        )
        if goal_align_ratio > 0.0:
            cte_correction *= (
                1.0
                - (1.0 - self.path_tracking_goal_cte_scale_min) * goal_align_ratio
            )
        # Keep the robot aligned to the chosen active-path segment.
        # Only add a very small point-bearing bias near the final goal so the
        # robot does not peel away from the selected route during normal travel.
        desired_yaw_raw = path_yaw_raw + goal_heading_weight * goal_bearing_err - cte_correction
        heading_filter_gain = self.path_tracking_heading_filter_gain
        if self.active_path_source == "local":
            heading_filter_gain = self.local_tracking_heading_filter_gain
            if (
                self._path_tracking_prev_desired_yaw is not None
                and self.local_tracking_heading_filter_reset > 1e-6
                and abs(
                    angdiff(
                        desired_yaw_raw,
                        self._path_tracking_prev_desired_yaw,
                    )
                )
                >= self.local_tracking_heading_filter_reset
            ):
                # Local paths are short rolling control references.  When a
                # refreshed local segment points a new way, stale filtered
                # heading causes the robot to kick right/left before settling.
                self._path_tracking_prev_desired_yaw = None
        if self._path_tracking_prev_desired_yaw is None:
            desired_yaw = desired_yaw_raw
        else:
            desired_yaw = wrap_angle(
                self._path_tracking_prev_desired_yaw +
                heading_filter_gain *
                angdiff(desired_yaw_raw, self._path_tracking_prev_desired_yaw)
            )
        self._path_tracking_prev_desired_yaw = desired_yaw
        yaw_err = angdiff(desired_yaw, x[2])
        self._last_tracking_debug = {
            "path_yaw_deg": math.degrees(path_yaw_raw),
            "desired_yaw_deg": math.degrees(desired_yaw),
            "robot_yaw_deg": math.degrees(x[2]),
            "yaw_err_deg": math.degrees(yaw_err),
            "goal_bearing_err_deg": math.degrees(goal_bearing_err),
            "cte_correction_deg": math.degrees(cte_correction),
            "filtered_lat_err": self._path_tracking_filtered_lat_err,
            "gate": "precheck",
            "grid_enforce": "unknown",
            "tangent_window_m": (
                min(
                    self.path_tracking_tangent_window_m,
                    self.local_tracking_tangent_window_m,
                    1.5 * self.local_tracking_target_step_cap_m,
                )
                if self.active_path_source == "local"
                else self.path_tracking_tangent_window_m
            ),
        }
        v_limit = min(v_cap, self.path_tracking_speed_cap)
        if self._emergency_bypass_active:
            v_limit = min(v_limit, self.emergency_bypass_speed_limit_mps)
        if self.active_path_source == "local" and obstacle_response_active:
            v_limit = min(v_limit, self.local_tracking_obstacle_speed_cap_mps)
        abs_err = abs(yaw_err)
        need_progress = remaining_dist > self.goal_thresh_m
        tracking_kp = self.path_tracking_kp * (
            self.obstacle_response_tracking_kp_scale if obstacle_response_active else 1.0
        )
        yaw_rate_limit = self.path_tracking_yaw_rate_max * (
            self.obstacle_response_yaw_rate_scale if obstacle_response_active else 1.0
        )
        if self.active_path_source == "local":
            tracking_kp *= 0.88
            yaw_rate_limit *= 0.82
        if abs_err >= self.path_tracking_stop_yaw:
            w_target = tracking_kp * yaw_err
            allow_large_yaw_crawl = (
                need_progress and self.path_tracking_large_yaw_crawl_speed > 0.0
            )
            if (
                self.active_path_source == "local"
                and abs_err >= self.local_tracking_stop_turn_only_yaw
            ):
                allow_large_yaw_crawl = False
            if allow_large_yaw_crawl:
                v_cmd = min(v_limit, self.path_tracking_large_yaw_crawl_speed)
                w_limit = yaw_rate_limit
            else:
                v_cmd = 0.0
                w_limit = self.path_tracking_in_place_yaw_rate_max
        else:
            if self.path_tracking_slowdown_yaw > 1e-6:
                slow_ratio_floor = 0.18 if obstacle_response_active else 0.25
                slow_ratio = max(slow_ratio_floor, 1.0 - abs_err / self.path_tracking_slowdown_yaw)
            else:
                slow_ratio = 1.0
            v_cmd = min(v_limit, max(0.0, v_limit * slow_ratio))
            if need_progress and v_cmd > 0.0 and self.path_tracking_crawl_speed > 0.0:
                v_cmd = max(v_cmd, min(v_limit, self.path_tracking_crawl_speed))
            w_target = tracking_kp * yaw_err
            w_limit = yaw_rate_limit

        if goal_align_ratio > 0.0:
            w_limit = min(
                w_limit,
                self.path_tracking_goal_yaw_rate_max
                + (1.0 - goal_align_ratio)
                * max(0.0, w_limit - self.path_tracking_goal_yaw_rate_max),
            )
        w_target = max(
            -w_limit,
            min(w_limit, w_target),
        )
        w_filtered = (
            self.path_tracking_steer_filter_gain * w_target +
            (1.0 - self.path_tracking_steer_filter_gain) * self._path_tracking_prev_w
        )
        max_w_step = max(1e-3, self.path_tracking_yaw_accel_max * self.dt)
        w_delta = max(
            -max_w_step,
            min(max_w_step, w_filtered - self._path_tracking_prev_w),
        )
        w_cmd = self._path_tracking_prev_w + w_delta
        w_cmd = max(
            -w_limit,
            min(w_limit, w_cmd),
        )
        self._path_tracking_prev_w = w_cmd

        if (
            need_progress
            and (not obstacle_response_active)
            and self.cruise_min_speed > 0.0
            and remaining_dist > self.cruise_distance_m
            and abs(self._path_tracking_filtered_lat_err) < self.cruise_lat_err_m
            and abs_err < self.cruise_max_heading_err
            and abs(w_cmd) < self.cruise_max_yaw_rate
            and v_cmd > 0.0
        ):
            v_cmd = min(v_limit, max(v_cmd, self.cruise_min_speed))

        traj = self.predict_trajectory(x, v_cmd, w_cmd)
        enforce_track_drivable = bool(
            self.use_drivable_grid and self.path_tracking_enforce_drivable_grid
        )
        self._last_tracking_debug["grid_enforce"] = (
            "on" if enforce_track_drivable else "off"
        )
        self._last_tracking_debug["gate"] = "ok"
        if not self._trajectory_in_drivable_area(
            traj,
            ignore_start_distance_m=self.path_tracking_drivable_ignore_start_distance_m,
            check_drivable_grid=enforce_track_drivable,
        ):
            self._last_tracking_debug["gate"] = (
                "primary_grid_block" if enforce_track_drivable else "primary_risk_block"
            )
            recovery_v = min(
                v_limit,
                max(
                    self.path_tracking_crawl_speed,
                    self.path_tracking_large_yaw_crawl_speed,
                ),
            )
            recovery_traj = self.predict_trajectory(x, recovery_v, w_cmd)
            if (
                need_progress
                and recovery_v > 1e-4
                and self._trajectory_in_drivable_area(
                    recovery_traj,
                    ignore_start_distance_m=self.path_tracking_recovery_ignore_start_distance_m,
                    check_drivable_grid=enforce_track_drivable,
                )
            ):
                v_cmd = recovery_v
                traj = recovery_traj
                self._last_tracking_debug["gate"] = "recovery_drivable"
            elif (
                need_progress
                and recovery_v > 1e-4
                and (not enforce_track_drivable)
                and self._trajectory_is_risk_only_safe(recovery_traj)
            ):
                v_cmd = recovery_v
                traj = recovery_traj
                self._last_tracking_debug["gate"] = "recovery_risk_only"
            else:
                v_cmd = 0.0
                traj = self.predict_trajectory(x, v_cmd, w_cmd)
                self._last_tracking_debug["gate"] = (
                    "blocked_grid" if enforce_track_drivable else "blocked_risk"
                )
        return [v_cmd, w_cmd], traj

    def moving(self, x, u):
        x[2] = self.get_yaw_from_quaternion(self.current_pose.pose.pose.orientation)
        x[0] = self.current_pose.pose.pose.position.x
        x[1] = self.current_pose.pose.pose.position.y
        x[3] = u[0]
        x[4] = u[1]
        return x

    def motion(self, x, u):
        x[2] += u[1] * self.dt
        x[0] += u[0] * math.cos(x[2]) * self.dt
        x[1] += u[0] * math.sin(x[2]) * self.dt
        x[3] = u[0]
        x[4] = u[1]
        return x

    def calc_dynamic_window(self, x):
        Vs = [self.min_speed, self.max_speed, -self.max_yaw_rate, self.max_yaw_rate]
        Vd = [x[3] - self.max_accel * self.dt,
              x[3] + self.max_accel * self.dt,
              x[4] - self.max_delta_yaw_rate * self.dt,
              x[4] + self.max_delta_yaw_rate * self.dt]
        return [max(Vs[0], Vd[0]),
                min(Vs[1], Vd[1]),
                max(Vs[2], Vd[2]),
                min(Vs[3], Vd[3])]

    def predict_trajectory(self, x_init, v, y):
        x = np.array(x_init)
        trajectory = np.array(x)
        t = 0.0
        while t <= self.predict_time:
            x = self.motion(x, [v, y])
            trajectory = np.vstack((trajectory, x))
            t += self.dt
        return trajectory

    def _is_xy_drivable_grid_ok(self, x, y):
        if self.use_drivable_grid:
            if self.grid_data is None or self.grid_width <= 0 or self.grid_height <= 0:
                return True
            else:
                gx = int(math.floor((x - self.grid_origin_x) / self.grid_resolution))
                gy = int(math.floor((y - self.grid_origin_y) / self.grid_resolution))
                if gx < 0 or gy < 0 or gx >= self.grid_width or gy >= self.grid_height:
                    return False
                idx = gy * self.grid_width + gx
                occ = self.grid_data[idx]
                if occ < 0:
                    return (not self.grid_unknown_is_occupied)
                else:
                    return (occ == 0)
        return True

    def _is_xy_risk_ok(self, x, y):
        if not (
            self.use_dynamic_risk_grid
            and self.risk_grid_data is not None
            and self.risk_grid_width > 0
            and self.risk_grid_height > 0
            and self.risk_grid_resolution is not None
            and self.risk_grid_resolution > 0.0
        ):
            return True
        rgx = int(math.floor((x - self.risk_grid_origin_x) / self.risk_grid_resolution))
        rgy = int(math.floor((y - self.risk_grid_origin_y) / self.risk_grid_resolution))
        if rgx < 0 or rgy < 0 or rgx >= self.risk_grid_width or rgy >= self.risk_grid_height:
            return (not self.risk_unknown_is_occupied)
        ridx = rgy * self.risk_grid_width + rgx
        rocc = int(self.risk_grid_data[ridx])
        if rocc < 0:
            return (not self.risk_unknown_is_occupied)
        if rocc >= self.risk_occupied_threshold:
            return False
        return True

    def _is_xy_drivable(self, x, y):
        return self._is_xy_drivable_grid_ok(x, y) and self._is_xy_risk_ok(x, y)

    def _trajectory_is_risk_only_safe(self, traj):
        if not self.use_dynamic_risk_grid:
            return True
        res_candidates = []
        if self.risk_grid_resolution is not None and self.risk_grid_resolution > 0.0:
            res_candidates.append(float(self.risk_grid_resolution))
        if self.grid_resolution is not None and self.grid_resolution > 0.0:
            res_candidates.append(float(self.grid_resolution))
        sample_step = min(res_candidates) if res_candidates else 0.1
        offsets = self._footprint_sample_offsets(sample_step)
        for row in traj[1::self.traj_check_step]:
            yaw = float(row[2])
            c = math.cos(yaw)
            s = math.sin(yaw)
            for ox, oy in offsets:
                wx = float(row[0]) + c * float(ox) - s * float(oy)
                wy = float(row[1]) + s * float(ox) + c * float(oy)
                if not self._is_xy_risk_ok(wx, wy):
                    return False
        return True

    def _trajectory_in_drivable_area(
        self,
        traj,
        ignore_start_distance_m=0.0,
        check_drivable_grid=None,
        check_risk_grid=None,
    ):
        if check_drivable_grid is None:
            check_drivable_grid = bool(self.use_drivable_grid)
        if check_risk_grid is None:
            check_risk_grid = bool(self.use_dynamic_risk_grid)
        if not check_drivable_grid and not check_risk_grid:
            return True
        res_candidates = []
        if (
            check_drivable_grid
            and self.grid_resolution is not None
            and self.grid_resolution > 0.0
        ):
            res_candidates.append(float(self.grid_resolution))
        if (
            check_risk_grid
            and self.risk_grid_resolution is not None
            and self.risk_grid_resolution > 0.0
        ):
            res_candidates.append(float(self.risk_grid_resolution))
        sample_step = min(res_candidates) if res_candidates else 0.1
        offsets = self._footprint_sample_offsets(sample_step)
        inward_recovery_ok = self._initial_inward_recovery_window_ok(
            traj,
            offsets,
            ignore_start_distance_m,
            check_drivable_grid=check_drivable_grid,
        )
        if not inward_recovery_ok:
            return False
        traveled_m = 0.0
        prev_row = traj[0]
        for row in traj[1::self.traj_check_step]:
            traveled_m += math.hypot(float(row[0]) - float(prev_row[0]), float(row[1]) - float(prev_row[1]))
            prev_row = row
            yaw = float(row[2])
            c = math.cos(yaw)
            s = math.sin(yaw)
            for ox, oy in offsets:
                wx = float(row[0]) + c * float(ox) - s * float(oy)
                wy = float(row[1]) + s * float(ox) + c * float(oy)
                if traveled_m < ignore_start_distance_m:
                    # Near the robot, only permit a short inward-recovery window:
                    # we still reject any newly outward-growing footprint, but allow
                    # existing edge overlap to shrink back inside the drivable mask.
                    if check_drivable_grid and not self._is_xy_drivable_grid_ok_with_tolerance(wx, wy):
                        continue
                    if check_risk_grid and (not self._is_xy_risk_ok(wx, wy)):
                        return False
                    continue
                if (
                    check_drivable_grid
                    and (not self._is_xy_drivable_grid_ok_with_tolerance(wx, wy))
                ):
                    return False
                if check_risk_grid and (not self._is_xy_risk_ok(wx, wy)):
                    return False
        return True

    def _obstacle_cost_for_trajectory(self, traj, x_now):
        if not self.use_pointcloud_obstacle_cost:
            return 0.0, False
        obs = self.obstacle_local_points
        if obs.shape[0] == 0:
            return 0.0, False

        min_dist = float("inf")
        collision_padding = self.footprint_padding_m + self.safety_margin + self.obstacle_collision_radius
        start_idx = min(len(traj) - 1, 1 + self.obstacle_ignore_traj_steps)
        rows = traj[start_idx::self.traj_check_step]
        if len(rows) == 0:
            rows = traj[1::self.traj_check_step]

        for row in rows:
            dx = float(row[0]) - float(x_now[0])
            dy = float(row[1]) - float(x_now[1])
            heading = float(row[2]) - float(x_now[2])
            c = math.cos(heading)
            s = math.sin(heading)
            rel = obs - np.array([dx, dy], dtype=np.float32)
            footprint_frame = np.empty_like(rel)
            footprint_frame[:, 0] = c * rel[:, 0] + s * rel[:, 1]
            footprint_frame[:, 1] = -s * rel[:, 0] + c * rel[:, 1]
            clearances = self._rect_clearance_local(footprint_frame, padding=collision_padding)
            if clearances.size == 0:
                continue
            row_min = float(np.min(clearances))
            if row_min < min_dist:
                min_dist = row_min
            if row_min <= 0.0:
                return float("inf"), True

        if not math.isfinite(min_dist) or min_dist >= self.obstacle_influence_distance:
            return 0.0, False

        clearance = max(0.05, min_dist)
        return self.obstacle_cost_gain / clearance, False

    def calc_control_and_trajectory(self, x, dw, goal_xy, t_hat):
        x_init = x[:]
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])
        gx, gy = goal_xy
        t_hat = np.array(t_hat)
        found_valid = False
        stats = {
            "sampled": 0,
            "skip_spin": 0,
            "skip_grid": 0,
            "collision": 0,
            "valid": 0,
        }

        for v in np.arange(dw[0], dw[1], self.v_resolution):
            for y in np.arange(dw[2], dw[3], self.yaw_rate_resolution):
                stats["sampled"] += 1

                # 거의 제자리 회전(v ≈ 0, yawrate는 큰 경우) 후보는 아예 무시
                if abs(v) < 0.03 and abs(y) > math.radians(10.0):
                    stats["skip_spin"] += 1
                    continue

                traj = self.predict_trajectory(x_init, v, y)
                if not self._trajectory_in_drivable_area(traj):
                    stats["skip_grid"] += 1
                    continue

                # 1) 타깃 점(target_xy) 기준 heading 오차
                dx_t = gx - traj[-1, 0]
                dy_t = gy - traj[-1, 1]
                error_angle = math.atan2(dy_t, dx_t)
                cost_angle = angdiff(error_angle, traj[-1, 2])
                to_goal_cost = self.to_goal_cost_gain * abs(cost_angle)

                # 2) 타깃 점까지 거리 (경로 위로 붙도록)
                dist_target = math.hypot(dx_t, dy_t)
                target_cost = self.target_cost_gain * dist_target

                # 3) 속도 cost (빨리 가는 걸 약간 선호)
                speed_cost = self.speed_cost_gain * (self.max_speed - traj[-1, 3])

                # 4) progress / lateral
                move_vec = np.array([traj[-1, 0] - x[0], traj[-1, 1] - x[1]])
                progress = float(np.dot(move_vec, t_hat))    # path tangent 방향으로 얼마나 갔나

                # 뒤로 가는 건 강하게 penalty
                progress_penalty = self.progress_cost_gain * max(0.0, -progress)
                # 앞으로 가는 건 약하게 reward (빙글빙글 방지)
                progress_reward  = self.progress_forward_gain * max(0.0, progress)

                # lateral (경로 라인에서의 옆으로 벗어남)
                normal = np.array([-t_hat[1], t_hat[0]])     # path normal
                end_from_target = np.array([traj[-1, 0] - gx, traj[-1, 1] - gy])
                lat_pred = abs(float(np.dot(end_from_target, normal)))
                lateral_cost = self.lateral_cost_gain * (lat_pred ** 2)

                # 5) obstacle cost
                obstacle_cost, collision = self._obstacle_cost_for_trajectory(traj, x)
                if collision:
                    stats["collision"] += 1
                    continue
                stats["valid"] += 1

                # 최종 cost
                final_cost = (
                    to_goal_cost +
                    target_cost +
                    speed_cost +
                    progress_penalty +
                    lateral_cost +
                    obstacle_cost -
                    progress_reward
                )

                if final_cost <= min_cost:
                    found_valid = True
                    min_cost = final_cost
                    best_u = [v, y]
                    best_trajectory = traj

                    if (abs(best_u[0]) < self.robot_stuck_flag_cons and
                        abs(x[3]) < self.robot_stuck_flag_cons):
                        best_u[1] = -self.max_delta_yaw_rate

        self._last_eval_stats = stats
        if not found_valid:
            return [0.0, 0.0], np.array([x])
        return best_u, best_trajectory

    # --------------------------------- helpers -----------------------------------
    def visualize_target_point(self, target_point):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.0
        marker.color.a = 1.0
        marker.color.g = 1.0
        marker.pose.position.x = target_point[0]
        marker.pose.position.y = target_point[1]
        self.target_pub.publish(marker)

    def visualize_current_point(self, s_proj):
        # project point on path for viz
        x, y, _t = self._interp_xy_tangent_at_s(max(0.0, min(self.s_total, s_proj)))
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.0
        marker.color.a = 1.0
        marker.color.b = 1.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        self.current_pub.publish(marker)

    def visualize_predicted_trajectory(self, predicted_trajectory):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.type = marker.LINE_STRIP
        marker.action = marker.ADD
        marker.scale.x = 0.2
        marker.color.a = 1.0
        marker.color.b = 1.0
        for point in predicted_trajectory:
            p = Point()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = 1.0
            marker.points.append(p)
        self.trajectory_pub.publish(marker)

    def publish_traj_info(self):
        msg = Float32MultiArray()
        cur_idx = 0
        # estimate an index for compatibility (segment start)
        if self.path_pts and len(self.cum_len) > 1:
            while cur_idx + 1 < len(self.cum_len) and self.cum_len[cur_idx + 1] < self.s_cur:
                cur_idx += 1
        msg.data = [self.s_total, self.s_cur, cur_idx, float(self.reach_goal_flag)]
        self.traj_info_pub.publish(msg)

    def get_yaw_from_quaternion(self, q):
        e = transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        return e[2]

    # ----------------------------------- main ------------------------------------
    def _slew_scalar(self, current, target, accel_limit, decel_limit, dt):
        if dt <= 0.0:
            return target
        moving_toward_zero = (current * target < 0.0) or (abs(target) < abs(current))
        limit = decel_limit if moving_toward_zero else accel_limit
        max_step = max(0.0, limit) * dt
        if target > current:
            return min(target, current + max_step)
        return max(target, current - max_step)

    def _smooth_drive_command(self, target_cmd):
        if not self.cmd_smoothing_enabled:
            return target_cmd
        if self._local_stop_turn_go_active():
            self._last_cmd_smooth_stamp = rospy.Time.now()
            return target_cmd
        now = rospy.Time.now()
        if self._last_cmd_smooth_stamp == rospy.Time(0):
            self._last_cmd_smooth_stamp = now
            return target_cmd
        dt = max(0.0, (now - self._last_cmd_smooth_stamp).to_sec())
        self._last_cmd_smooth_stamp = now
        if dt <= 1e-6 or dt > 0.5:
            return target_cmd

        smoothed = Twist()
        smoothed.linear.x = self._slew_scalar(
            self.last_cmd.linear.x,
            target_cmd.linear.x,
            self.cmd_linear_accel_max,
            self.cmd_linear_decel_max,
            dt,
        )
        smoothed.angular.z = self._slew_scalar(
            self.last_cmd.angular.z,
            target_cmd.angular.z,
            self.cmd_angular_accel_max,
            self.cmd_angular_decel_max,
            dt,
        )
        if abs(target_cmd.linear.x) <= self.cmd_smoothing_zero_snap and abs(smoothed.linear.x) <= self.cmd_smoothing_zero_snap:
            smoothed.linear.x = 0.0
        if abs(target_cmd.angular.z) <= math.radians(0.5) and abs(smoothed.angular.z) <= math.radians(0.5):
            smoothed.angular.z = 0.0
        return smoothed

    def publish_drive(self, u):
        cmd = Twist()
        hard_stop_active = self._hard_stop_active()
        self._publish_emergency_stop_state()
        allow_reverse_recovery_motion = (
            self._reverse_recovery_active
            and (not self.behavior_stop)
            and len(u) >= 1
            and float(u[0]) < -1e-4
        )
        if hard_stop_active and not allow_reverse_recovery_motion:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self._last_cmd_smooth_stamp = rospy.Time.now()
        else:
            cmd.linear.x = u[0]
            cmd.angular.z = u[1]
            cmd = self._smooth_drive_command(cmd)
        self.last_cmd = cmd
        self.cmd_vel_pub.publish(cmd)

    def run(self):
        rospy.loginfo(
            "DWA node started | pose=%s global=%s local=%s path_mode=%s global_only=%s behavior=%s cmd=%s estop_topic=%s drivable=%s risk=%s local_avoidance=%s emergency_enabled=%s emergency_stop=%.2fm hard_stop=%.2fm overlay_stop=%s locked_only=%s overlay_topic=%s near_raw=%s raw_fallback=%s near_topic=%s near_frame=%s near_rate=%.1fHz near_roi=x[%.2f,%.2f] y=+/-%0.2f z[%.2f,%.2f] min_pts=%d reverse_recovery=%s hold=%.2fs dist=%.2fm speed=%.2fmps rear=%.2fm/%.2fm/%d self_filter=%s self_mask=%.2fx%.2fm footprint=%.2fm x %.2fm cmd_publish=%.1fHz path_tracking_only=%s crawl=%.2f/%.2f heading_filter=%.2f cmd_smooth=%s lin=%.2f/%.2f ang=%.0f/%.0fdeg",
            self.pose_topic,
            self.global_path_topic,
            self.local_path_topic,
            self.path_mode_topic,
            "on" if self.follow_global_path_only else "off",
            self.behavior_cmd_topic,
            self.cmd_vel_topic,
            self.emergency_stop_topic,
            "on" if self.use_drivable_grid else "off",
            "on" if self.use_dynamic_risk_grid else "off",
            "off" if self.follow_global_path_only else "on",
            "on" if self.enable_emergency_stop else "off",
            self.emergency_stop_distance,
            self.avoidance_hard_stop_distance,
            "on" if self.use_global_obstacle_overlay_boxes_for_stop else "off",
            "on" if self.global_obstacle_overlay_stop_locked_only else "off",
            self.global_obstacle_overlay_boxes_topic if self.global_obstacle_overlay_boxes_topic else "-",
            "on" if self.near_field_raw_stop_enabled else "off",
            "on" if self.enable_raw_cloud_fallback_stop else "off",
            self.near_field_raw_stop_topic if self.near_field_raw_stop_topic else "-",
            self.near_field_raw_stop_base_frame if self.near_field_raw_stop_base_frame else "-",
            self.near_field_raw_stop_max_update_hz,
            self.near_field_raw_stop_min_x_m,
            self.near_field_raw_stop_max_x_m,
            self.near_field_raw_stop_half_width_m,
            self.near_field_raw_stop_min_z_m,
            self.near_field_raw_stop_max_z_m,
            self.near_field_raw_stop_min_points,
            "on" if self.enable_reverse_recovery else "off",
            self.reverse_recovery_hold_trigger_s,
            self.reverse_recovery_distance_m,
            self.reverse_recovery_speed_mps,
            self.reverse_recovery_rear_check_distance_m,
            self.reverse_recovery_rear_half_width_m,
            self.reverse_recovery_rear_min_points,
            "on" if self.enable_self_filter else "off",
            self.self_filter_radius_x,
            self.self_filter_radius_y,
            self.robot_length_m,
            self.robot_width_m,
            self.cmd_publish_hz,
            "on" if self.path_tracking_only else "off",
            self.path_tracking_crawl_speed,
            self.path_tracking_large_yaw_crawl_speed,
            self.path_tracking_heading_filter_gain,
            "on" if self.cmd_smoothing_enabled else "off",
            self.cmd_linear_accel_max,
            self.cmd_linear_decel_max,
            math.degrees(self.cmd_angular_accel_max),
            math.degrees(self.cmd_angular_decel_max),
        )
        rospy.loginfo(
            "DWA launch profile | profile=%s real_mode=%s env=%s map=%s map_path=%s state=%s gates: enforce_drivable=%s pointcloud_static=%s use_drivable=%s unknown_occ=%s risk=%s local_avoidance=%s tracking: preview=%.2fm step_cap=%.2fm speed_cap=%.2fmps",
            self.launch_profile_label,
            self.launch_real_mode,
            self.launch_localization_environment,
            self.launch_map_profile_name,
            self.launch_localizer_map_relative_path,
            self.launch_runtime_drivable_state_file,
            "on" if self.path_tracking_enforce_drivable_grid else "off",
            "on" if self.pointcloud_static_blocking_enabled else "off",
            "on" if self.use_drivable_grid else "off",
            "occupied" if self.grid_unknown_is_occupied else "free",
            "on" if self.use_dynamic_risk_grid else "off",
            "off" if self.follow_global_path_only else "on",
            self.local_tracking_preview_m,
            self.local_tracking_target_step_cap_m,
            self.path_tracking_speed_cap,
        )
        x = [self.current_pose.pose.pose.position.x,
             self.current_pose.pose.pose.position.y,
             self.get_yaw_from_quaternion(self.current_pose.pose.pose.orientation),
             0.0, 0.0]
        rate = rospy.Rate(1.0 / self.dt)

        while not rospy.is_shutdown():
            self._refresh_active_path()
            self._update_emergency_stop_state()
            self._emergency_bypass_active = False
            self._emergency_bypass_debug = {}

            # behavior-layer hard stop
            if self.behavior_stop:
                self._rot_mode = False
                self._clear_local_turn_rotate_state()
                self._log_nav_reason(
                    "stop_behavior",
                    "reason=%s speed_limit=%.2f" % (self.behavior_reason, self.behavior_speed_limit),
                    warn=True,
                )
                self.publish_drive([0.0, 0.0])
                rate.sleep()
                continue

            # current pose snapshot
            yaw = self.get_yaw_from_quaternion(self.current_pose.pose.pose.orientation)
            px = self.current_pose.pose.pose.position.x
            py = self.current_pose.pose.pose.position.y

            if self._handle_reverse_recovery(px, py, yaw):
                rate.sleep()
                continue

            if not self.path_pts:
                if self.follow_global_path_only:
                    self._log_nav_reason(
                        "stop_no_path",
                        "active=%s global_pts=%d" % (
                            self.active_path_source,
                            len(self.global_path_msg.poses) if self.global_path_msg else 0,
                        ),
                        warn=True,
                    )
                else:
                    local_age = (rospy.Time.now() - self.local_path_stamp).to_sec() if self.local_path_stamp.to_sec() > 0.0 else -1.0
                    local_source_age = (
                        (rospy.Time.now() - self.local_path_source_stamp).to_sec()
                        if self.local_path_source_stamp.to_sec() > 0.0
                        else -1.0
                    )
                    self._log_nav_reason(
                        "stop_no_path",
                        "active=%s local_age=%.2fs source_age=%.2fs global_pts=%d local_pts=%d" % (
                            self.active_path_source,
                            local_age,
                            local_source_age,
                            len(self.global_path_msg.poses) if self.global_path_msg else 0,
                            len(self.local_path_msg.poses) if self.local_path_msg else 0,
                        ),
                        warn=True,
                    )
                self._rot_mode = False
                self._rot_yaw_target = None
                self._clear_local_turn_rotate_state()
                self.publish_drive([0.0, 0.0])
                rate.sleep()
                continue

            if self.current_path_mode == "hold":
                hold_s = 0.0
                if self._hold_mode_enter_sec > 0.0:
                    hold_s = max(
                        0.0, rospy.Time.now().to_sec() - self._hold_mode_enter_sec
                    )
                self._rot_mode = False
                self._log_nav_reason(
                    "stop_hold",
                    "src=%s active=%s hold=%.2fs path_pts=%d local_pts=%d global_pts=%d"
                    % (
                        self.active_path_source,
                        self.current_path_mode,
                        hold_s,
                        len(self.path_pts),
                        len(self.local_path_msg.poses)
                        if self.local_path_msg is not None
                        else 0,
                        len(self.global_path_msg.poses)
                        if self.global_path_msg is not None
                        else 0,
                    ),
                    warn=True,
                )
                self._clear_local_turn_rotate_state()
                self._publish_tracking_reference_path()
                self.publish_drive([0.0, 0.0])
                rate.sleep()
                continue

            # progress / target compute
            s_proj, lat_err, target_xy, t_hat, at_goal, dist_to_goal, arc_rem = \
                self._update_progress_and_target(px, py, yaw)
            if s_proj is None:
                self._log_nav_reason("stop_no_target", "failed to compute target from current path", warn=True)
                self._clear_local_turn_rotate_state()
                self.publish_drive([0.0, 0.0])
                rate.sleep()
                continue

            avoidance_can_continue = self._compute_emergency_bypass_state(
                px,
                py,
                yaw,
                s_proj,
                lat_err,
                target_xy,
                dist_to_goal,
            )
            self._emergency_bypass_active = bool(avoidance_can_continue)
            if self.enable_emergency_stop and self.emergency_blocked and not avoidance_can_continue:
                self._rot_mode = False
                self._log_nav_reason(
                    "stop_emergency",
                    "front obstacle stop active clr=%.2f raw=%.2f overlay=%.2f close=%d intrusion=%d overlay_boxes=%d source=%s" % (
                        self.front_obstacle_clearance if math.isfinite(self.front_obstacle_clearance) else float("inf"),
                        self._raw_front_obstacle_clearance if math.isfinite(self._raw_front_obstacle_clearance) else float("inf"),
                        self.overlay_box_front_clearance if math.isfinite(self.overlay_box_front_clearance) else float("inf"),
                        self._last_emergency_close_points,
                        self._last_emergency_intrusion_points,
                        self._overlay_box_match_count,
                        self._last_emergency_source,
                    ),
                    warn=True,
                )
                self._clear_local_turn_rotate_state()
                self.publish_drive([0.0, 0.0])
                rate.sleep()
                continue

            self._publish_tracking_reference_path()
            self.visualize_current_point(s_proj)
            self.visualize_target_point(target_xy)
            self.publish_traj_info()

            # goal handling
            self.prev_goal_flag = self.reach_goal_flag
            self.reach_goal_flag = at_goal
            if (not self.reach_goal_flag) and self._near_goal_completion_allowed(
                dist_to_goal,
                arc_rem,
                lat_err,
            ):
                self.reach_goal_flag = True
            if (not self.reach_goal_flag) and self._near_goal_stuck_completion_allowed(
                rospy.Time.now().to_sec()
            ):
                self.reach_goal_flag = True
            if self.reach_goal_flag:
                self._clear_no_valid_traj_state()
                self._log_nav_reason(
                    "goal_reached",
                    "dist=%.2f arc=%.2f lat=%.2f" % (dist_to_goal, arc_rem, lat_err),
                )
                self._clear_local_turn_rotate_state()
                self.publish_drive([0.0, 0.0])
                if not self.prev_goal_flag:
                    rospy.loginfo("Goal reached!")
                rate.sleep()
                continue

            # final-approach speed cap
            final_window = self.final_approach_window_m
            if dist_to_goal <= final_window:
                v_cap = max(
                    self.final_speed_min,
                    min(self.max_speed, self.final_speed_k * max(dist_to_goal, 0.0)),
                )
            else:
                v_cap = self.max_speed
            v_cap = min(v_cap, max(0.0, self.behavior_speed_limit))
            tracking_remaining_dist = dist_to_goal

            local_control = self._local_stop_turn_go_control(px, py, yaw, v_cap)
            if local_control is not None:
                if local_control["mode"] == "rotate":
                    self._rot_mode = False
                    self._log_nav_reason(
                        "local_turn_rotate",
                        "kind=%s err=%.1fdeg target=%.1fdeg" % (
                            local_control["kind"],
                            math.degrees(local_control["yaw_err"]),
                            math.degrees(local_control["desired_heading"]),
                        ),
                    )
                    x = self.moving(x, local_control["cmd"])
                    self.publish_drive(local_control["cmd"])
                    rate.sleep()
                    continue
                if local_control["mode"] == "settle":
                    self._log_nav_reason(
                        "local_turn_settle",
                        "reason=%s target=%.1fdeg" % (
                            local_control["reason"],
                            math.degrees(local_control["desired_heading"]),
                        ),
                    )
                    self.publish_drive([0.0, 0.0])
                    rate.sleep()
                    continue
                target_xy = local_control["target_xy"]
                t_hat = local_control["t_hat"]
                tracking_remaining_dist = local_control["remaining_dist"]
                v_cap = min(v_cap, local_control["v_cap"])
                self.visualize_target_point(target_xy)

            # rotate-only gating with path tangent
            desired = math.atan2(t_hat[1], t_hat[0])
            err = abs(angdiff(desired, yaw))
            # if we're roughly aligned with path tangent (forward progress), avoid entering rotate-only
            heading_vec = np.array([math.cos(yaw), math.sin(yaw)])
            dot_forward = float(np.dot(heading_vec, np.array(t_hat)))
            rotate_only_allowed = self._rotate_only_allowed(
                dist_to_goal,
                obstacle_response_active=self._emergency_bypass_active or str(self.behavior_reason).strip().lower() != "clear",
            )
            rotate_cooldown_active = rospy.Time.now() < self._rot_cooldown_until
            rotate_target_changed = (
                self._rot_last_timeout_target is None or
                abs(angdiff(desired, self._rot_last_timeout_target)) > self._ROT_RETARGET
            )
            if (
                (not self._rot_mode)
                and rotate_only_allowed
                and (err > self._ROT_HIGH)
                and (dot_forward < 0.2)
                and (not rotate_cooldown_active or rotate_target_changed)
            ):
                self.rotate_only_enter(yaw, desired)
            if self._rot_mode:
                if not rotate_only_allowed:
                    self._rot_mode = False
                    self._rot_last_timeout_target = self._rot_yaw_target
                    self._rot_cooldown_until = rospy.Time.now() + rospy.Duration(
                        self.rotate_reentry_cooldown_s
                    )
                    self._log_nav_reason(
                        "rotate_cancel",
                        "cancel rotate-only: active_path=%s behavior=%s bypass=%s" % (
                            self.active_path_source,
                            self.behavior_reason,
                            "on" if self._emergency_bypass_active else "off",
                        ),
                    )
                else:
                    u_rot, done, exit_reason = self.rotate_only_step(yaw)
                    if not done:
                        self._log_nav_reason(
                            "rotate_only",
                            "target=%.1fdeg" % math.degrees(self._rot_yaw_target),
                        )
                        x = self.moving(x, u_rot)
                        self.publish_drive(u_rot)
                        rate.sleep()
                        continue
                    if exit_reason is not None:
                        self._log_nav_reason(
                            "rotate_cooldown",
                            "skip re-entry for %.1fs after %s" % (
                                self.rotate_reentry_cooldown_s,
                                exit_reason,
                            ),
                        )

            if self.path_tracking_only:
                u, predicted = self.path_tracking_control(
                    x,
                    target_xy,
                    t_hat,
                    lat_err,
                    v_cap,
                    tracking_remaining_dist,
                )
            else:
                u, predicted = self.dwa_control(x, target_xy, t_hat, lat_err)
            self.visualize_predicted_trajectory(predicted)
            x = self.moving(x, u)

            # low-speed clamp with cap in direction of chosen v sign
            u_cmd = list(u)
            obstacle_response_active = (
                str(self.behavior_reason).strip().lower() != "clear"
                or self._emergency_bypass_active
            )

            if u_cmd[0] >= 0.0:
                u_cmd[0] = min(v_cap, max(0.0, u_cmd[0]))
            else:
                # min_speed = 0.0 이라서 여기까지는 거의 안 옴
                u_cmd[0] = -min(v_cap, max(0.0, -u_cmd[0]))

            # 너무 느리게 기어가면 노이즈만 생기니, 아주 작으면 그냥 0으로
            if (
                (not obstacle_response_active)
                and
                u_cmd[0] > 0.0
                and dist_to_goal > self.min_forward_cmd_distance
                and u_cmd[0] < self.min_forward_cmd
                and (self.path_tracking_only or u_cmd[0] > self.forward_motion_deadband)
            ):
                u_cmd[0] = min(v_cap, self.min_forward_cmd)

            if (
                (not self.path_tracking_only)
                and
                dist_to_goal > self.cruise_distance_m
                and abs(lat_err) < self.cruise_lat_err_m
                and abs(u_cmd[1]) < self.cruise_max_yaw_rate
                and u_cmd[0] > 0.0
            ):
                u_cmd[0] = min(v_cap, max(u_cmd[0], self.cruise_min_speed))

            if abs(u_cmd[0]) < self.forward_motion_deadband:
                if abs(u[0]) < self.forward_motion_deadband and abs(u[1]) < math.radians(1.0):
                    now_sec = rospy.Time.now().to_sec()
                    if self._no_valid_traj_since_sec <= 0.0:
                        self._no_valid_traj_since_sec = now_sec
                    self._no_valid_traj_dist_m = float(dist_to_goal)
                    self._no_valid_traj_arc_m = float(arc_rem)
                    self._no_valid_traj_lat_m = float(lat_err)
                    st = self._last_eval_stats
                    log_now = rospy.Time.now()
                    pose_age_s = self._stamp_age_s(self.current_pose.header.stamp, log_now)
                    local_age_s = self._stamp_age_s(self.local_path_stamp, log_now)
                    local_source_age_s = self._stamp_age_s(
                        self.local_path_source_stamp, log_now
                    )
                    tdbg = self._last_target_debug
                    self._log_nav_reason(
                        "stop_no_valid_traj",
                        "src=%s mode=%s pts=%d policy=%s gate=%s grid=%s dist=%.2f arc=%.2f lat=%.2f s=%.2f->%.2f/%.2f sampled=%d valid=%d skip_grid=%d collision=%d pose_age=%.2fs local_age=%.2fs source_age=%.2fs" % (
                            self.active_path_source,
                            self.current_path_mode,
                            int(tdbg.get("path_len", len(self.path_pts))),
                            str(tdbg.get("policy", "-")),
                            str(self._last_tracking_debug.get("gate", "-")),
                            str(self._last_tracking_debug.get("grid_enforce", "-")),
                            dist_to_goal,
                            arc_rem,
                            lat_err,
                            float(tdbg.get("s_proj", self.s_cur)),
                            float(tdbg.get("s_target", self.s_cur)),
                            float(tdbg.get("s_total", self.s_total)),
                            st.get("sampled", 0),
                            st.get("valid", 0),
                            st.get("skip_grid", 0),
                            st.get("collision", 0),
                            pose_age_s,
                            local_age_s,
                            local_source_age_s,
                        ),
                        warn=True,
                    )
                elif abs(u[0]) > 0.0:
                    self._clear_no_valid_traj_state()
                    self._log_nav_reason(
                        "stop_deadband",
                        "raw_v=%.3f dist=%.2f arc=%.2f v_cap=%.2f" % (
                            u[0],
                            dist_to_goal,
                            arc_rem,
                            v_cap,
                        ),
                    )
                u_cmd[0] = 0.0
            else:
                self._clear_no_valid_traj_state()
                dbg = self._last_tracking_debug
                tdbg = self._last_target_debug
                log_now = rospy.Time.now()
                pose_age_s = self._stamp_age_s(self.current_pose.header.stamp, log_now)
                local_age_s = self._stamp_age_s(self.local_path_stamp, log_now)
                local_source_age_s = self._stamp_age_s(
                    self.local_path_source_stamp, log_now
                )
                self._log_nav_reason(
                    "tracking",
                    "src=%s mode=%s pts=%d policy=%s gate=%s grid=%s cmd_v=%.3f cmd_w=%.3f dist=%.2f arc=%.2f lat=%.2f s=%.2f->%.2f/%.2f yaw=%.1f path=%.1f des=%.1f err=%.1f cte=%.1f bypass=%s p_y=%.2f clr=%.2f pose_age=%.2fs local_age=%.2fs source_age=%.2fs" % (
                        self.active_path_source,
                        self.current_path_mode,
                        int(tdbg.get("path_len", len(self.path_pts))),
                        str(tdbg.get("policy", "-")),
                        str(dbg.get("gate", "-")),
                        str(dbg.get("grid_enforce", "-")),
                        u_cmd[0],
                        u_cmd[1],
                        dist_to_goal,
                        arc_rem,
                        lat_err,
                        float(tdbg.get("s_proj", self.s_cur)),
                        float(tdbg.get("s_target", self.s_cur)),
                        float(tdbg.get("s_total", self.s_total)),
                        dbg.get("robot_yaw_deg", 0.0),
                        dbg.get("path_yaw_deg", 0.0),
                        dbg.get("desired_yaw_deg", 0.0),
                        dbg.get("yaw_err_deg", 0.0),
                        dbg.get("cte_correction_deg", 0.0),
                        "on" if self._emergency_bypass_active else "off",
                        self._emergency_bypass_debug.get("preview_local_y", 0.0),
                        self.front_obstacle_clearance if math.isfinite(self.front_obstacle_clearance) else float("inf"),
                        pose_age_s,
                        local_age_s,
                        local_source_age_s,
                    ),
                )

            self.publish_drive(u_cmd)
            rate.sleep()

# -------------------------------- entry point ------------------------------------
def main():
    rospy.init_node('dwa_node', anonymous=True)
    dwa = DWAControl()
    dwa.run()

if __name__ == "__main__":
    main()
