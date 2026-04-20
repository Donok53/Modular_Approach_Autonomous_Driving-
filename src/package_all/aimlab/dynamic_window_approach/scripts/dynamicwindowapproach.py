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
import numpy as np
import rospy
import tf2_ros
import tf.transformations as transformations

from geometry_msgs.msg import Twist, Point, PoseStamped
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool, Float32MultiArray, Header

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
        self.avoidance_path_topic = rospy.get_param("~avoidance_path_topic", "/planning/avoidance_path")
        self.active_path_topic = rospy.get_param("~active_path_topic", "/planning/active_path")
        self.follow_global_path_only = bool(
            rospy.get_param("~follow_global_path_only", False)
        )
        self.local_path_timeout_s = float(rospy.get_param("~local_path_timeout_s", 4.0))
        self.avoidance_path_timeout_s = float(
            rospy.get_param("~avoidance_path_timeout_s", max(0.5, self.local_path_timeout_s))
        )
        self.use_muxed_active_path = (
            bool(rospy.get_param("~use_muxed_active_path", True))
            and not self.follow_global_path_only
        )
        self.active_path_timeout_s = float(
            rospy.get_param(
                "~active_path_timeout_s",
                max(0.5, self.avoidance_path_timeout_s),
            )
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
        self.near_field_raw_stop_downsample = max(
            1, int(rospy.get_param("~near_field_raw_stop_downsample", 1))
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
        self._last_emergency_source = "none"
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
        self.rotate_only_deg = rospy.get_param("~rotate_only_deg", 80.0)
        self.rotate_exit_deg = rospy.get_param("~rotate_exit_deg", 30.0)
        self.rotate_kp = rospy.get_param("~rotate_kp", 2.0)
        self.rotate_w_max_deg = rospy.get_param("~rotate_w_max_deg", 120.0)
        self.rotate_ok_count = rospy.get_param("~rotate_ok_count", 3)
        self.rotate_max_spin_deg = rospy.get_param("~rotate_max_spin_deg", 420.0)
        self.rotate_max_time_s = rospy.get_param("~rotate_max_time_s", 6.0)
        self.rotate_reentry_cooldown_s = max(
            0.0, float(rospy.get_param("~rotate_reentry_cooldown_s", 2.5))
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
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 0.55)
        self.back_jitter_m = rospy.get_param("~back_jitter_m", 0.3)
        self.goal_thresh_m = rospy.get_param("~goal_thresh_m", 0.25)
        self.final_approach_window_m = rospy.get_param("~final_approach_window_m", 2.5)
        self.final_speed_k = rospy.get_param("~final_speed_k", 0.75)
        self.final_speed_min = rospy.get_param("~final_speed_min", 0.22)
        self.lat_goal_slop = rospy.get_param("~lat_goal_slop", 0.6)
        self.near_goal_no_rotate_m = rospy.get_param("~near_goal_no_rotate_m", 1.0)
        self.forward_motion_deadband = rospy.get_param("~forward_motion_deadband", 0.02)
        self.min_forward_cmd = rospy.get_param("~min_forward_cmd", 0.20)
        self.min_forward_cmd_distance = rospy.get_param("~min_forward_cmd_distance", 0.8)
        self.cruise_min_speed = rospy.get_param("~cruise_min_speed", 0.0)
        self.cruise_distance_m = rospy.get_param("~cruise_distance_m", 2.5)
        self.cruise_lat_err_m = rospy.get_param("~cruise_lat_err_m", 0.25)
        self.cruise_max_yaw_rate = math.radians(rospy.get_param("~cruise_max_yaw_rate_deg", 45.0))
        self.current_point_search_radius_m = 5.0  # legacy (kept for /traj_info)
        # 경로에서 이 정도 이상 벗어나면 일단 경로로 붙는 스냅 단계
        self.snap_lat_err = rospy.get_param("~snap_lat_err", 0.25)
        self.snap_target_ahead_m = max(
            0.0, float(rospy.get_param("~snap_target_ahead_m", 0.40))
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
        self.path_tracking_only = bool(rospy.get_param("~path_tracking_only", True))
        self.path_tracking_kp = float(rospy.get_param("~path_tracking_kp", 1.2))
        self.path_tracking_yaw_rate_max = math.radians(
            rospy.get_param("~path_tracking_yaw_rate_max_deg", 55.0)
        )
        self.path_tracking_in_place_yaw_rate_max = math.radians(
            rospy.get_param(
                "~path_tracking_in_place_yaw_rate_max_deg",
                rospy.get_param("~path_tracking_yaw_rate_max_deg", 55.0),
            )
        )
        self.path_tracking_yaw_accel_max = math.radians(
            rospy.get_param("~path_tracking_yaw_accel_max_deg", 180.0)
        )
        self.path_tracking_speed_cap = max(
            0.05, float(rospy.get_param("~path_tracking_speed_cap", 0.25))
        )
        self.path_tracking_steer_filter_gain = min(
            1.0, max(0.05, float(rospy.get_param("~path_tracking_steer_filter_gain", 0.20)))
        )
        self.path_tracking_slowdown_yaw = math.radians(
            rospy.get_param("~path_tracking_slowdown_yaw_deg", 35.0)
        )
        self.path_tracking_stop_yaw = math.radians(
            rospy.get_param("~path_tracking_stop_yaw_deg", 65.0)
        )
        self.path_tracking_cte_gain = float(rospy.get_param("~path_tracking_cte_gain", 1.6))
        self.path_tracking_cte_soft_mps = max(
            0.05, float(rospy.get_param("~path_tracking_cte_soft_mps", 0.25))
        )
        self.path_tracking_cte_deadband_m = max(
            0.0, float(rospy.get_param("~path_tracking_cte_deadband_m", 0.03))
        )
        self.path_tracking_cte_filter_gain = min(
            1.0,
            max(0.05, float(rospy.get_param("~path_tracking_cte_filter_gain", 0.25))),
        )
        self.path_tracking_cte_yaw_cap = math.radians(
            rospy.get_param("~path_tracking_cte_yaw_cap_deg", 35.0)
        )
        self.path_tracking_heading_filter_gain = min(
            1.0,
            max(0.05, float(rospy.get_param("~path_tracking_heading_filter_gain", 0.18))),
        )
        self.path_tracking_goal_bearing_gain = min(
            1.0,
            max(0.0, float(rospy.get_param("~path_tracking_goal_bearing_gain", 0.45))),
        )
        self.path_tracking_goal_bearing_cap = math.radians(
            rospy.get_param("~path_tracking_goal_bearing_cap_deg", 35.0)
        )
        self.path_tracking_target_step_m = max(
            0.05, float(rospy.get_param("~path_tracking_target_step_m", 0.18))
        )
        self.path_tracking_tangent_window_m = max(
            0.0,
            float(
                rospy.get_param(
                    "~path_tracking_tangent_window_m",
                    max(0.25, min(0.60, self.lookahead_distance)),
                )
            ),
        )
        self.path_tracking_crawl_speed = max(
            0.0, float(rospy.get_param("~path_tracking_crawl_speed", 0.10))
        )
        self.path_tracking_large_yaw_crawl_speed = max(
            0.0,
            float(
                rospy.get_param(
                    "~path_tracking_large_yaw_crawl_speed",
                    self.path_tracking_crawl_speed,
                )
            ),
        )
        self.path_tracking_stop_distance_m = max(
            0.0, float(rospy.get_param("~path_tracking_stop_distance_m", 0.80))
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
                        min(25.0, math.degrees(self.path_tracking_yaw_rate_max)),
                    )
                ),
            ),
        )
        self.path_tracking_drivable_ignore_start_distance_m = max(
            0.0,
            float(
                rospy.get_param(
                    "~path_tracking_drivable_ignore_start_distance_m",
                    0.45,
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
        self._path_tracking_prev_w = 0.0
        self._path_tracking_prev_desired_yaw = None
        self._path_tracking_filtered_lat_err = 0.0

        # Internal path buffers
        self.global_path_msg = None
        self.global_path_sig = None
        self.local_path_msg = None
        self.local_path_sig = None
        self.local_path_stamp = rospy.Time(0)
        self.avoidance_path_msg = None
        self.avoidance_path_sig = None
        self.avoidance_path_stamp = rospy.Time(0)
        self.active_path_msg = None
        self.active_path_sig = None
        self.active_path_stamp = rospy.Time(0)
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
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.emergency_stop_topic = rospy.get_param("~emergency_stop_topic", "/planning/emergency_stop")
        self.cmd_publish_hz = max(5.0, float(rospy.get_param("~cmd_publish_hz", 20.0)))
        self.last_cmd = Twist()

        # ===== ROS I/O =====
        self.sub_path_global = rospy.Subscriber(self.global_path_topic, Path, self.path_callback_global, queue_size=5)
        self.sub_path_local = None
        self.sub_path_avoidance = None
        if not self.follow_global_path_only:
            self.sub_path_local = rospy.Subscriber(
                self.local_path_topic, Path, self.path_callback_local, queue_size=5
            )
            self.sub_path_avoidance = rospy.Subscriber(
                self.avoidance_path_topic,
                Path,
                self.path_callback_avoidance,
                queue_size=5,
            )
        self.sub_path_active = None
        if self.use_muxed_active_path:
            self.sub_path_active = rospy.Subscriber(
                self.active_path_topic,
                Path,
                self.path_callback_active,
                queue_size=5,
            )
        self.sub_pose = rospy.Subscriber(self.pose_topic, Odometry, self.pose_callback)
        self.sub_server_cmd = rospy.Subscriber("server_to_robot_topic", server_to_robot, self.server_to_robot_callback)
        self.sub_behavior = rospy.Subscriber(
            self.behavior_cmd_topic,
            BehaviorCommand,
            self.behavior_cmd_callback,
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
        self.sub_cloud = rospy.Subscriber(self.cloud_topic, PointCloud2, self.cloud_callback, queue_size=1)
        self.sub_near_field_raw_stop = None
        if self.near_field_raw_stop_enabled and self.near_field_raw_stop_topic:
            self.sub_near_field_raw_stop = rospy.Subscriber(
                self.near_field_raw_stop_topic,
                PointCloud2,
                self.near_field_raw_stop_callback,
                queue_size=1,
                buff_size=2**24,
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
        if self._publish_emergency_stop_state():
            self.last_cmd = Twist()
        self.cmd_vel_pub.publish(self.last_cmd)
        self._refresh_near_field_raw_stop_marker_if_waiting()

    def _hard_stop_active(self):
        return bool(self.emergency_blocked or self.behavior_stop)

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
        near_raw_blocked = self._near_field_raw_stop_blocked and near_raw_fresh

        raw_fallback_blocked = self._raw_immediate_contact_blocked or (
            self._raw_emergency_band_blocked
            and self._raw_front_obstacle_clearance <= self.avoidance_hard_stop_distance
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

    def _near_field_header(self, msg, frame_id=None):
        header = Header()
        header.stamp = msg.header.stamp if msg.header.stamp.to_sec() > 0.0 else rospy.Time.now()
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

        markers = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        box = Marker()
        box.header = header
        box.ns = "near_field_raw_stop"
        box.id = 0
        box.type = Marker.CUBE
        box.action = Marker.ADD
        box.pose.position.x = 0.5 * (
            self.near_field_raw_stop_min_x_m + self.near_field_raw_stop_max_x_m
        )
        box.pose.position.y = 0.0
        box.pose.position.z = 0.5 * (
            self.near_field_raw_stop_min_z_m + self.near_field_raw_stop_max_z_m
        )
        box.pose.orientation.w = 1.0
        box.scale.x = self.near_field_raw_stop_max_x_m - self.near_field_raw_stop_min_x_m
        box.scale.y = 2.0 * self.near_field_raw_stop_half_width_m
        box.scale.z = self.near_field_raw_stop_max_z_m - self.near_field_raw_stop_min_z_m
        box.color.a = 0.22 if self._near_field_raw_stop_blocked else 0.12
        box.color.r = 1.0 if self._near_field_raw_stop_blocked else 0.1
        box.color.g = 0.05 if self._near_field_raw_stop_blocked else 0.9
        box.color.b = 0.05
        box.lifetime = rospy.Duration(max(0.2, self.near_field_raw_stop_timeout_s * 2.0))
        markers.markers.append(box)

        text = Marker()
        text.header = header
        text.ns = "near_field_raw_stop"
        text.id = 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = self.near_field_raw_stop_max_x_m
        text.pose.position.y = 0.0
        text.pose.position.z = self.near_field_raw_stop_max_z_m + 0.25
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
            self._near_field_raw_stop_tf_status,
        )
        text.lifetime = box.lifetime
        markers.markers.append(text)

        self.near_field_raw_stop_marker_pub.publish(markers)

    def near_field_raw_stop_callback(self, msg):
        try:
            transform_mat, transform_ok, debug_frame = self._near_field_raw_stop_transform_matrix(msg)
            header = self._near_field_header(msg, debug_frame)
            if not transform_ok:
                self._near_field_raw_stop_last_stamp = rospy.Time.now()
                self._near_field_raw_stop_last_count = 0
                self._near_field_raw_stop_last_cells = 0
                self._near_field_raw_stop_last_min_x = float("inf")
                self._publish_near_field_raw_stop_debug(header, [])
                self._update_emergency_stop_state()
                return

            hit_points = []
            occupied_cells = set()
            min_x = float("inf")
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
                x, y, z = self._transform_point_xyz(transform_mat, float(x), float(y), float(z))
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
                if x < min_x:
                    min_x = x

            hit_count = len(hit_points)
            cell_count = len(occupied_cells)
            detected = (
                hit_count >= self.near_field_raw_stop_min_points
                and cell_count >= self.near_field_raw_stop_min_cells
            )
            footprint_front = self.robot_half_length_m + self.footprint_padding_m
            min_front_clearance = (
                min_x - footprint_front if math.isfinite(min_x) else float("inf")
            )
            stop_detected = (
                detected and min_front_clearance <= self.emergency_stop_distance
            )
            if stop_detected:
                self._near_field_raw_stop_on += 1
                self._near_field_raw_stop_off = 0
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
            self._publish_near_field_raw_stop_debug(header, hit_points)
            self._update_emergency_stop_state()

            if self.near_field_raw_stop_log_period_s > 0.0:
                now = rospy.Time.now()
                if (
                    now - self._near_field_raw_stop_last_log
                ).to_sec() >= self.near_field_raw_stop_log_period_s:
                    self._near_field_raw_stop_last_log = now
                    rospy.loginfo(
                        "near_field_raw_stop: %s pts=%d cells=%d min_x=%.2f clearance=%.2f stop_dist=%.2f roi=[x %.2f..%.2f y +/-%.2f z %.2f..%.2f] topic=%s frame=%s tf=%s",
                        "STOP" if self._near_field_raw_stop_blocked else "clear",
                        hit_count,
                        cell_count,
                        min_x if math.isfinite(min_x) else float("inf"),
                        min_front_clearance if math.isfinite(min_front_clearance) else float("inf"),
                        self.emergency_stop_distance,
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

    # ------------------------------- obstacle stop -------------------------------
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
        return max_free_gap_m + 1e-6 < self.emergency_passable_width_m

    def cloud_callback(self, msg):
        try:
            obs = []
            close_points = []
            intrusion_points = []
            min_front_clearance = float("inf")
            influence_sq = self.obstacle_influence_distance * self.obstacle_influence_distance
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

    # ------------------------------- rotate-only --------------------------------
    def rotate_only_enter(self, cur_yaw, desired_yaw):
        self._rot_mode = True
        self._rot_yaw_target = desired_yaw
        self._rot_prev_yaw = cur_yaw
        self._rot_accum = 0.0
        self._rot_ok = 0
        self._rot_start_time = rospy.Time.now()
        self._rot_cooldown_until = rospy.Time(0)
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
            if exit_reason == "aligned":
                self._rot_last_timeout_target = None
                self._rot_cooldown_until = rospy.Time(0)
            else:
                self._rot_last_timeout_target = self._rot_yaw_target
                self._rot_cooldown_until = now + rospy.Duration(self.rotate_reentry_cooldown_s)
            rospy.loginfo("Rotate-only EXIT: err=%.1f°, accum=%.1f°, t=%.1fs",
                          math.degrees(err), math.degrees(self._rot_accum), time_in)
            return None, True, exit_reason
        return u, False, None

    # ------------------------------ path handling --------------------------------
    def _path_signature(self, path_msg):
        if not path_msg or not path_msg.poses:
            return None
        n = len(path_msg.poses)
        p0 = path_msg.poses[min(1, n - 1)].pose.position
        p_mid = path_msg.poses[n // 2].pose.position
        p1 = path_msg.poses[-1].pose.position
        return (n,
                round(p0.x, 3), round(p0.y, 3),
                round(p_mid.x, 3), round(p_mid.y, 3),
                round(p1.x, 3), round(p1.y, 3))

    def _rebuild_path_geometry(self):
        self.path_pts = []
        for ps in self.path_msg.poses:
            p = ps.pose.position
            self.path_pts.append((p.x, p.y))
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
        self.reach_goal_flag = False
        self.prev_goal_flag = False

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
        px = float(self.current_pose.pose.pose.position.x)
        py = float(self.current_pose.pose.pose.position.y)
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

    def _build_tracking_reference_path_msg(self):
        if len(self.path_pts) < 2:
            return self._empty_tracking_reference_path()

        msg = Path()
        msg.header = self.path_msg.header if self.path_msg is not None else self._empty_tracking_reference_path().header
        msg.header.stamp = rospy.Time.now()

        try:
            px = float(self.current_pose.pose.pose.position.x)
            py = float(self.current_pose.pose.pose.position.y)
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
        reset_tracking = (
            source != self.active_path_source
            or path_msg is None
            or self.path_msg is None
        )
        self.path_sig = sig
        self.path_msg = path_msg
        self.active_path_source = source
        if reset_tracking:
            self._path_tracking_prev_w = 0.0
            self._path_tracking_prev_desired_yaw = None
            self._path_tracking_filtered_lat_err = 0.0
            self._rot_mode = False
            self._rot_yaw_target = None
        if path_msg is None or len(path_msg.poses) < 2:
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

    def path_callback_local(self, path_msg):
        self.local_path_msg = path_msg
        self.local_path_sig = self._path_signature(path_msg)
        self.local_path_stamp = rospy.Time.now()

    def path_callback_avoidance(self, path_msg):
        self.avoidance_path_msg = path_msg
        self.avoidance_path_sig = self._path_signature(path_msg)
        self.avoidance_path_stamp = rospy.Time.now()

    def path_callback_active(self, path_msg):
        self.active_path_msg = path_msg
        self.active_path_sig = self._path_signature(path_msg)
        self.active_path_stamp = rospy.Time.now()

    def _refresh_active_path(self):
        now = rospy.Time.now()
        if self.follow_global_path_only:
            if self.global_path_msg is not None and len(self.global_path_msg.poses) >= 2:
                if self.active_path_source != "global" or self.path_sig != self.global_path_sig:
                    self._activate_path(self.global_path_msg, self.global_path_sig, "global")
                return
            if self.path_msg is not None or self.active_path_source != "none":
                self._activate_path(None, None, "none")
            return

        if self.use_muxed_active_path:
            active_fresh = (
                self.active_path_stamp.to_sec() > 0.0
                and (now - self.active_path_stamp).to_sec() <= self.active_path_timeout_s
            )
            if active_fresh:
                if self.active_path_msg is not None and len(self.active_path_msg.poses) >= 2:
                    if self.active_path_source != "active" or self.path_sig != self.active_path_sig:
                        self._activate_path(self.active_path_msg, self.active_path_sig, "active")
                elif self.path_msg is not None or self.active_path_source != "none":
                    self._activate_path(None, None, "none")
                return

            if self.path_msg is not None or self.active_path_source != "none":
                self._activate_path(None, None, "none")
            return

        use_avoidance = (
            self.avoidance_path_msg is not None
            and len(self.avoidance_path_msg.poses) >= 2
            and (now - self.avoidance_path_stamp).to_sec() <= self.avoidance_path_timeout_s
        )
        if use_avoidance:
            if self.active_path_source != "avoidance" or self.path_sig != self.avoidance_path_sig:
                self._activate_path(self.avoidance_path_msg, self.avoidance_path_sig, "avoidance")
            return

        use_local = (
            self.local_path_msg is not None
            and len(self.local_path_msg.poses) >= 2
            and (now - self.local_path_stamp).to_sec() <= self.local_path_timeout_s
        )
        if use_local:
            if self.active_path_source != "local" or self.path_sig != self.local_path_sig:
                self._activate_path(self.local_path_msg, self.local_path_sig, "local")
            return

        if self.global_path_msg is not None and len(self.global_path_msg.poses) >= 2:
            if self.active_path_source != "global" or self.path_sig != self.global_path_sig:
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
        if len(self.path_pts) < 3 or self.path_tracking_tangent_window_m <= 1e-3:
            return x, y, t_hat

        window = self.path_tracking_tangent_window_m
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

        s_min = max(0.0, self.s_cur - self.tracking_projection_back_window_m)
        s_max = min(self.s_total, self.s_cur + self.tracking_projection_forward_window_m)
        s_proj, lat_err, idx, t = self._project_to_path(
            pose_x,
            pose_y,
            s_min=s_min,
            s_max=s_max,
        )

        # enforce monotonic progress with tiny back jitter allowed
        if s_proj + self.back_jitter_m >= self.s_cur:
            self.s_cur = max(self.s_cur, s_proj)

        base_s = max(self.s_cur, s_proj)
        gx, gy = self.path_pts[-1]
        dist_to_goal = math.hypot(gx - pose_x, gy - pose_y)
        goal_align_active = (
            min(max(0.0, self.s_total - base_s), dist_to_goal)
            <= self.path_tracking_goal_align_window_m
        )

        if goal_align_active:
            s_target = self.s_total
        else:
            # Follow the currently selected active-path segment directly.
            # Keep the target only a short distance ahead on the same segment so
            # the robot does not cut diagonally across corners or skip ahead to a
            # visually different shortcut.
            target_seg_idx = idx
            if t >= 0.98 and target_seg_idx + 1 < len(self.seg_lens):
                target_seg_idx += 1
            segment_end_s = self.cum_len[target_seg_idx + 1]
            segment_target_step = min(
                max(0.05, self.path_tracking_target_step_m),
                max(0.05, 0.8 * self.seg_lens[target_seg_idx]),
            )
            s_target = min(
                self.s_total,
                min(segment_end_s, max(base_s, s_proj + segment_target_step)),
            )

        tx, ty, t_hat = self._interp_xy_smoothed_tangent_at_s(s_target)

        # goal metrics
        arc_rem = max(0.0, self.s_total - self.s_cur)
        at_goal = (min(arc_rem, dist_to_goal) <= self.goal_thresh_m) and (abs(lat_err) <= self.lat_goal_slop)

        return (s_proj, lat_err, (tx, ty), t_hat, at_goal, dist_to_goal, arc_rem)

    # ------------------------------- dwa core ------------------------------------
    def dwa_control(self, x, goal_xy, t_hat, lat_err):
        dw = self.calc_dynamic_window(x)
        u, trajectory = self.calc_control_and_trajectory(x, dw, goal_xy, t_hat)
        return u, trajectory

    def path_tracking_control(self, x, goal_xy, t_hat, lat_err, v_cap, remaining_dist):
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
        goal_heading_weight = 0.0
        if remaining_dist <= max(self.lookahead_distance, 0.8):
            goal_heading_weight = min(0.20, self.path_tracking_goal_bearing_gain)
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
        if self._path_tracking_prev_desired_yaw is None:
            desired_yaw = desired_yaw_raw
        else:
            desired_yaw = wrap_angle(
                self._path_tracking_prev_desired_yaw +
                self.path_tracking_heading_filter_gain *
                angdiff(desired_yaw_raw, self._path_tracking_prev_desired_yaw)
            )
        self._path_tracking_prev_desired_yaw = desired_yaw
        yaw_err = angdiff(desired_yaw, x[2])
        v_limit = min(v_cap, self.path_tracking_speed_cap)
        abs_err = abs(yaw_err)
        need_progress = remaining_dist > self.goal_thresh_m
        if abs_err >= self.path_tracking_stop_yaw:
            w_target = self.path_tracking_kp * yaw_err
            if need_progress and self.path_tracking_large_yaw_crawl_speed > 0.0:
                v_cmd = min(v_limit, self.path_tracking_large_yaw_crawl_speed)
                w_limit = self.path_tracking_yaw_rate_max
            else:
                v_cmd = 0.0
                w_limit = self.path_tracking_in_place_yaw_rate_max
        else:
            if self.path_tracking_slowdown_yaw > 1e-6:
                slow_ratio = max(0.25, 1.0 - abs_err / self.path_tracking_slowdown_yaw)
            else:
                slow_ratio = 1.0
            v_cmd = min(v_limit, max(0.0, v_limit * slow_ratio))
            if need_progress and v_cmd > 0.0 and self.path_tracking_crawl_speed > 0.0:
                v_cmd = max(v_cmd, min(v_limit, self.path_tracking_crawl_speed))
            w_target = self.path_tracking_kp * yaw_err
            w_limit = self.path_tracking_yaw_rate_max

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

        traj = self.predict_trajectory(x, v_cmd, w_cmd)
        if not self._trajectory_in_drivable_area(
            traj,
            ignore_start_distance_m=self.path_tracking_drivable_ignore_start_distance_m,
        ):
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
                )
            ):
                v_cmd = recovery_v
                traj = recovery_traj
            elif (
                need_progress
                and recovery_v > 1e-4
                and self._trajectory_is_risk_only_safe(recovery_traj)
            ):
                v_cmd = recovery_v
                traj = recovery_traj
            else:
                v_cmd = 0.0
                traj = self.predict_trajectory(x, v_cmd, w_cmd)
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

    def _trajectory_in_drivable_area(self, traj, ignore_start_distance_m=0.0):
        if not self.use_drivable_grid and not self.use_dynamic_risk_grid:
            return True
        res_candidates = []
        if self.grid_resolution is not None and self.grid_resolution > 0.0:
            res_candidates.append(float(self.grid_resolution))
        if self.risk_grid_resolution is not None and self.risk_grid_resolution > 0.0:
            res_candidates.append(float(self.risk_grid_resolution))
        sample_step = min(res_candidates) if res_candidates else 0.1
        offsets = self._footprint_sample_offsets(sample_step)
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
                    if not self._is_xy_risk_ok(wx, wy):
                        return False
                    continue
                if not self._is_xy_drivable(wx, wy):
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
    def publish_drive(self, u):
        cmd = Twist()
        if self._publish_emergency_stop_state():
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            cmd.linear.x = u[0]
            cmd.angular.z = u[1]
        self.last_cmd = cmd
        self.cmd_vel_pub.publish(cmd)

    def run(self):
        rospy.loginfo(
            "DWA node started | pose=%s global=%s local=%s avoidance=%s active=%s global_only=%s active_mux=%s behavior=%s cmd=%s estop_topic=%s drivable=%s risk=%s local_avoidance=%s emergency_stop=%.2fm hard_stop=%.2fm overlay_stop=%s locked_only=%s overlay_topic=%s near_raw=%s near_topic=%s near_frame=%s near_roi=x[%.2f,%.2f] y=+/-%0.2f z[%.2f,%.2f] min_pts=%d self_filter=%s self_mask=%.2fx%.2fm footprint=%.2fm x %.2fm cmd_publish=%.1fHz path_tracking_only=%s crawl=%.2f/%.2f heading_filter=%.2f",
            self.pose_topic,
            self.global_path_topic,
            self.local_path_topic,
            self.avoidance_path_topic,
            self.active_path_topic,
            "on" if self.follow_global_path_only else "off",
            "on" if self.use_muxed_active_path else "off",
            self.behavior_cmd_topic,
            self.cmd_vel_topic,
            self.emergency_stop_topic,
            "on" if self.use_drivable_grid else "off",
            "on" if self.use_dynamic_risk_grid else "off",
            "off" if self.follow_global_path_only else "on",
            self.emergency_stop_distance,
            self.avoidance_hard_stop_distance,
            "on" if self.use_global_obstacle_overlay_boxes_for_stop else "off",
            "on" if self.global_obstacle_overlay_stop_locked_only else "off",
            self.global_obstacle_overlay_boxes_topic if self.global_obstacle_overlay_boxes_topic else "-",
            "on" if self.near_field_raw_stop_enabled else "off",
            self.near_field_raw_stop_topic if self.near_field_raw_stop_topic else "-",
            self.near_field_raw_stop_base_frame if self.near_field_raw_stop_base_frame else "-",
            self.near_field_raw_stop_min_x_m,
            self.near_field_raw_stop_max_x_m,
            self.near_field_raw_stop_half_width_m,
            self.near_field_raw_stop_min_z_m,
            self.near_field_raw_stop_max_z_m,
            self.near_field_raw_stop_min_points,
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
        )
        x = [self.current_pose.pose.pose.position.x,
             self.current_pose.pose.pose.position.y,
             self.get_yaw_from_quaternion(self.current_pose.pose.pose.orientation),
             0.0, 0.0]
        rate = rospy.Rate(1.0 / self.dt)

        while not rospy.is_shutdown():
            self._refresh_active_path()
            self._update_emergency_stop_state()

            # emergency stop only (avoidance can proceed unless obstacle is critically close)
            avoidance_can_continue = (
                (not self.follow_global_path_only)
                and self.active_path_source == "avoidance"
                and self.front_obstacle_clearance > self.avoidance_hard_stop_distance
            )
            if self.emergency_blocked and not avoidance_can_continue:
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
                self.publish_drive([0.0, 0.0])
                rate.sleep()
                continue

            # behavior-layer hard stop
            if self.behavior_stop:
                self._rot_mode = False
                self._log_nav_reason(
                    "stop_behavior",
                    "reason=%s speed_limit=%.2f" % (self.behavior_reason, self.behavior_speed_limit),
                    warn=True,
                )
                self.publish_drive([0.0, 0.0])
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
                    self._log_nav_reason(
                        "stop_no_path",
                        "active=%s local_age=%.2fs global_pts=%d local_pts=%d" % (
                            self.active_path_source,
                            local_age,
                            len(self.global_path_msg.poses) if self.global_path_msg else 0,
                            len(self.local_path_msg.poses) if self.local_path_msg else 0,
                        ),
                        warn=True,
                    )
                self.publish_drive([0.0, 0.0])
                rate.sleep()
                continue

            # current pose snapshot
            yaw = self.get_yaw_from_quaternion(self.current_pose.pose.pose.orientation)
            px = self.current_pose.pose.pose.position.x
            py = self.current_pose.pose.pose.position.y

            # progress / target compute
            s_proj, lat_err, target_xy, t_hat, at_goal, dist_to_goal, arc_rem = \
                self._update_progress_and_target(px, py, yaw)
            if s_proj is None:
                self._log_nav_reason("stop_no_target", "failed to compute target from current path", warn=True)
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
            if self.reach_goal_flag:
                self._log_nav_reason(
                    "goal_reached",
                    "dist=%.2f arc=%.2f lat=%.2f" % (dist_to_goal, arc_rem, lat_err),
                )
                self.publish_drive([0.0, 0.0])
                if not self.prev_goal_flag:
                    rospy.loginfo("Goal reached!")
                rate.sleep()
                continue

            # rotate-only gating with path tangent
            desired = math.atan2(t_hat[1], t_hat[0])
            err = abs(angdiff(desired, yaw))
            # if we're roughly aligned with path tangent (forward progress), avoid entering rotate-only
            heading_vec = np.array([math.cos(yaw), math.sin(yaw)])
            dot_forward = float(np.dot(heading_vec, np.array(t_hat)))
            rotate_cooldown_active = rospy.Time.now() < self._rot_cooldown_until
            rotate_target_changed = (
                self._rot_last_timeout_target is None or
                abs(angdiff(desired, self._rot_last_timeout_target)) > self._ROT_RETARGET
            )
            if (
                (not self._rot_mode)
                and (err > self._ROT_HIGH)
                and (dot_forward < 0.2)
                and (min(arc_rem, dist_to_goal) > self.near_goal_no_rotate_m)
                and (not rotate_cooldown_active or rotate_target_changed)
            ):
                self.rotate_only_enter(yaw, desired)
            if self._rot_mode:
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
                if exit_reason in ("timeout", "spin_limit"):
                    self._log_nav_reason(
                        "rotate_cooldown",
                        "skip re-entry for %.1fs after %s" % (
                            self.rotate_reentry_cooldown_s,
                            exit_reason,
                        ),
                    )

            # final-approach speed cap
            final_window = self.final_approach_window_m
            if min(arc_rem, dist_to_goal) <= final_window:
                v_cap = max(self.final_speed_min,
                            min(self.max_speed, self.final_speed_k * max(dist_to_goal, 0.0)))
            else:
                v_cap = self.max_speed
            v_cap = min(v_cap, max(0.0, self.behavior_speed_limit))

            if self.path_tracking_only:
                u, predicted = self.path_tracking_control(
                    x,
                    target_xy,
                    t_hat,
                    lat_err,
                    v_cap,
                    min(arc_rem, dist_to_goal),
                )
            else:
                u, predicted = self.dwa_control(x, target_xy, t_hat, lat_err)
            self.visualize_predicted_trajectory(predicted)
            x = self.moving(x, u)

            # low-speed clamp with cap in direction of chosen v sign
            u_cmd = list(u)

            if u_cmd[0] >= 0.0:
                u_cmd[0] = min(v_cap, max(0.0, u_cmd[0]))
            else:
                # min_speed = 0.0 이라서 여기까지는 거의 안 옴
                u_cmd[0] = -min(v_cap, max(0.0, -u_cmd[0]))

            # 너무 느리게 기어가면 노이즈만 생기니, 아주 작으면 그냥 0으로
            if (
                u_cmd[0] > 0.0
                and min(arc_rem, dist_to_goal) > self.min_forward_cmd_distance
                and u_cmd[0] < self.min_forward_cmd
                and (self.path_tracking_only or u_cmd[0] > self.forward_motion_deadband)
            ):
                u_cmd[0] = min(v_cap, self.min_forward_cmd)

            if (
                (not self.path_tracking_only)
                and
                min(arc_rem, dist_to_goal) > self.cruise_distance_m
                and abs(lat_err) < self.cruise_lat_err_m
                and abs(u_cmd[1]) < self.cruise_max_yaw_rate
                and u_cmd[0] > 0.0
            ):
                u_cmd[0] = min(v_cap, max(u_cmd[0], self.cruise_min_speed))

            if abs(u_cmd[0]) < self.forward_motion_deadband:
                if abs(u[0]) < self.forward_motion_deadband and abs(u[1]) < math.radians(1.0):
                    st = self._last_eval_stats
                    self._log_nav_reason(
                        "stop_no_valid_traj",
                        "dist=%.2f arc=%.2f lat=%.2f sampled=%d valid=%d skip_grid=%d collision=%d" % (
                            dist_to_goal,
                            arc_rem,
                            lat_err,
                            st.get("sampled", 0),
                            st.get("valid", 0),
                            st.get("skip_grid", 0),
                            st.get("collision", 0),
                        ),
                        warn=True,
                    )
                elif abs(u[0]) > 0.0:
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
                self._log_nav_reason(
                    "tracking",
                    "cmd_v=%.3f cmd_w=%.3f dist=%.2f arc=%.2f lat=%.2f" % (
                        u_cmd[0],
                        u_cmd[1],
                        dist_to_goal,
                        arc_rem,
                        lat_err,
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
