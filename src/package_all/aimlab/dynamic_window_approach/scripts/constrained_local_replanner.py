#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import heapq
import math
import traceback
import zlib
from array import array
from collections import deque

import rospy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from dynamic_window_approach.msg import ExplainabilityEvent, TrackedObjectArray


class ConstrainedLocalReplanner:
    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic", "/lio_localizer/odometry/optimization")
        self.global_path_topic = rospy.get_param("~global_path_topic", "/astar/path")
        self.drivable_grid_topic = rospy.get_param("~drivable_grid_topic", "/lio_sam/drivable_area/grid")
        self.dynamic_risk_grid_topic = rospy.get_param("~dynamic_risk_grid_topic", "/planning/dynamic_risk_grid")
        self.local_path_topic = rospy.get_param("~local_path_topic", "/planning/local_path")
        self.avoidance_path_topic = rospy.get_param("~avoidance_path_topic", "/planning/avoidance_path")
        self.path_mode_topic = rospy.get_param("~path_mode_topic", "/planning/path_mode")
        self.path_history_topic = rospy.get_param("~path_history_topic", "/planning/path_history")
        self.travel_history_topic = rospy.get_param("~travel_history_topic", "/planning/travel_history")
        self.travel_history_path_topic = rospy.get_param(
            "~travel_history_path_topic", "/planning/travel_history_path"
        )
        nominal_path_reference_mode = str(
            rospy.get_param("~nominal_path_reference_mode", "local")
        ).strip().lower()
        if nominal_path_reference_mode not in ("local", "global"):
            rospy.logwarn(
                "constrained_local_replanner: unsupported nominal_path_reference_mode=%s, falling back to local",
                nominal_path_reference_mode,
            )
            nominal_path_reference_mode = "local"
        if nominal_path_reference_mode == "global":
            rospy.logwarn(
                "constrained_local_replanner: nominal_path_reference_mode=global is deprecated; "
                "using always-on local nominal tracking instead"
            )
            nominal_path_reference_mode = "local"
        self.nominal_path_reference_mode = nominal_path_reference_mode
        self.pointcloud_topic = rospy.get_param("~pointcloud_topic", "/ouster/points")
        self.obstacle_pointcloud_topic = rospy.get_param(
            "~obstacle_pointcloud_topic", self.pointcloud_topic
        )
        self.use_raw_near_obstacle_hits = bool(
            rospy.get_param("~use_raw_near_obstacle_hits", True)
        )
        self.raw_near_obstacle_hits_topic = str(
            rospy.get_param(
                "~raw_near_obstacle_hits_topic",
                "/planning/near_field_raw_overlay_hits",
            )
        ).strip()
        self.tracked_objects_topic = rospy.get_param(
            "~tracked_objects_topic", "/perception/tracked_objects"
        )
        self.use_direct_goal = bool(rospy.get_param("~use_direct_goal", False))
        self.direct_goal_topic = rospy.get_param("~direct_goal_topic", "/move_base_simple/goal")
        self.direct_goal_timeout_s = max(
            0.0, float(rospy.get_param("~direct_goal_timeout_s", 0.0))
        )
        self.direct_goal_refresh_distance_m = max(
            0.0, float(rospy.get_param("~direct_goal_refresh_distance_m", 0.05))
        )
        self.direct_goal_refresh_yaw_deg = max(
            0.0, float(rospy.get_param("~direct_goal_refresh_yaw_deg", 5.0))
        )
        self.goal_tolerance_m = max(0.05, float(rospy.get_param("~goal_tolerance_m", 0.35)))
        self.snap_search_radius_cells = max(1, int(rospy.get_param("~snap_search_radius_cells", 30)))
        self.freeze_path_on_first_plan = bool(rospy.get_param("~freeze_path_on_first_plan", True))
        self.smooth_path_line_of_sight = bool(rospy.get_param("~smooth_path_line_of_sight", True))
        self.enable_avoidance_path = bool(rospy.get_param("~enable_avoidance_path", True))
        self.allow_best_effort_path = bool(rospy.get_param("~allow_best_effort_path", True))
        self.smooth_avoidance_line_of_sight = bool(
            rospy.get_param("~smooth_avoidance_line_of_sight", True)
        )
        self.max_los_segment_m = max(
            0.3, float(rospy.get_param("~max_los_segment_m", 1.0))
        )
        self.best_effort_improve_margin_cells = max(
            0.0, float(rospy.get_param("~best_effort_improve_margin_cells", 2.0))
        )
        self.best_effort_update_period_s = max(
            0.1, float(rospy.get_param("~best_effort_update_period_s", 1.5))
        )
        self.best_effort_max_goal_gap_m = max(
            0.0, float(rospy.get_param("~best_effort_max_goal_gap_m", 0.60))
        )

        self.lookahead_m = max(2.0, float(rospy.get_param("~lookahead_m", 10.0)))
        self.nominal_start_heading_penalty_m = max(
            0.0, float(rospy.get_param("~nominal_start_heading_penalty_m", 1.25))
        )
        self.nominal_start_heading_search_ahead_m = max(
            0.0,
            float(rospy.get_param("~nominal_start_heading_search_ahead_m", 5.0)),
        )
        self.nominal_start_min_heading_dot = max(
            -1.0,
            min(1.0, float(rospy.get_param("~nominal_start_min_heading_dot", 0.05))),
        )
        self.nominal_start_backtrack_m = max(
            0.0, float(rospy.get_param("~nominal_start_backtrack_m", 1.0))
        )
        self.window_margin_m = max(1.0, float(rospy.get_param("~window_margin_m", 12.0)))
        legacy_robot_radius = max(0.05, float(rospy.get_param("~robot_radius_m", 0.45)))
        self.robot_width_m = max(
            0.05, float(rospy.get_param("~robot_width_m", 0.58))
        )
        self.robot_length_m = max(
            0.05, float(rospy.get_param("~robot_length_m", 0.612))
        )
        self.robot_radius = 0.5 * math.hypot(self.robot_length_m, self.robot_width_m)
        self.footprint_padding_m = max(0.0, float(rospy.get_param("~footprint_padding_m", 0.0)))
        self.robot_half_length = 0.5 * self.robot_length_m + self.footprint_padding_m
        self.robot_half_width = 0.5 * self.robot_width_m + self.footprint_padding_m
        self.footprint_clearance_radius_m = self.robot_radius + self.footprint_padding_m
        self.trim_published_path_to_robot_front = bool(
            rospy.get_param("~trim_published_path_to_robot_front", True)
        )
        self.path_start_front_offset_m = max(
            0.0,
            float(
                rospy.get_param(
                    "~path_start_front_offset_m",
                    0.5 * self.robot_length_m + self.footprint_padding_m,
                )
            ),
        )
        self.default_path_blocking_radius_m = (
            0.5 * self.robot_width_m + self.footprint_padding_m + 0.16
        )
        self.path_blocking_radius_m = max(
            0.05,
            float(
                rospy.get_param(
                    "~path_blocking_radius_m",
                    self.default_path_blocking_radius_m,
                )
            ),
        )
        self.near_goal_relaxed_path_blocking_radius_m = max(
            0.05,
            float(
                rospy.get_param(
                    "~near_goal_relaxed_path_blocking_radius_m",
                    0.5 * self.robot_width_m + self.footprint_padding_m + 0.02,
                )
            ),
        )
        self.grid_only_relaxed_path_blocking_enabled = bool(
            rospy.get_param("~grid_only_relaxed_path_blocking_enabled", True)
        )
        self.allow_grid_only_nominal_fallback = bool(
            rospy.get_param("~allow_grid_only_nominal_fallback", True)
        )
        self.grid_only_nominal_fallback_max_cells = max(
            0, int(rospy.get_param("~grid_only_nominal_fallback_max_cells", 0))
        )
        self.grid_only_avoidance_search_enabled = bool(
            rospy.get_param("~grid_only_avoidance_search_enabled", False)
        )
        self.grid_only_relaxed_path_blocking_radius_m = max(
            0.05,
            float(
                rospy.get_param(
                    "~grid_only_relaxed_path_blocking_radius_m",
                    0.5 * self.robot_width_m + self.footprint_padding_m + 0.02,
                )
            ),
        )
        self.relaxed_snap_path_blocking_radius_m = max(
            0.05,
            float(
                rospy.get_param(
                    "~relaxed_snap_path_blocking_radius_m",
                    0.5 * self.robot_width_m + self.footprint_padding_m + 0.02,
                )
            ),
        )
        self.risk_threshold = int(rospy.get_param("~risk_occupied_threshold", 45))
        self.max_expand = max(100, int(rospy.get_param("~max_expand", 25000)))
        self.replan_hz = max(1.0, float(rospy.get_param("~replan_hz", 6.0)))
        self.replan_period_s = 1.0 / self.replan_hz
        self.local_path_keepalive_enabled = bool(
            rospy.get_param("~local_path_keepalive_enabled", True)
        )
        self.local_path_keepalive_hz = max(
            0.5, float(rospy.get_param("~local_path_keepalive_hz", 5.0))
        )
        self.local_path_keepalive_max_age_s = max(
            0.0, float(rospy.get_param("~local_path_keepalive_max_age_s", 2.5))
        )
        self.simplify_stride = max(1, int(rospy.get_param("~simplify_stride", 1)))
        self.published_path_spacing_m = max(
            0.05, float(rospy.get_param("~published_path_spacing_m", 0.25))
        )
        self.path_history_max_paths = max(
            1, int(rospy.get_param("~path_history_max_paths", 12))
        )
        self.used_local_path_commit_distance_m = max(
            0.05,
            float(rospy.get_param("~used_local_path_commit_distance_m", 0.20)),
        )
        self.travel_history_max_points = max(
            2, int(rospy.get_param("~travel_history_max_points", 400))
        )
        self.travel_history_spacing_m = max(
            0.02, float(rospy.get_param("~travel_history_spacing_m", 0.05))
        )
        self.recognized_obstacles_marker_topic = str(
            rospy.get_param(
                "~recognized_obstacles_marker_topic",
                "/planning/recognized_obstacles",
            )
        ).strip()
        self.recognized_obstacles_marker_max_points = max(
            1, int(rospy.get_param("~recognized_obstacles_marker_max_points", 450))
        )
        self.recognized_obstacles_marker_scale_m = max(
            0.02, float(rospy.get_param("~recognized_obstacles_marker_scale_m", 0.09))
        )
        self.recognized_obstacles_marker_lifetime_s = max(
            0.0, float(rospy.get_param("~recognized_obstacles_marker_lifetime_s", 0.8))
        )
        self.blocking_obstacles_marker_topic = str(
            rospy.get_param(
                "~blocking_obstacles_marker_topic",
                "/planning/blocking_obstacles",
            )
        ).strip()
        self.blocking_obstacles_marker_max_points = max(
            1, int(rospy.get_param("~blocking_obstacles_marker_max_points", 180))
        )
        self.blocking_obstacles_marker_scale_m = max(
            0.02, float(rospy.get_param("~blocking_obstacles_marker_scale_m", 0.12))
        )
        self.blocking_obstacles_marker_lifetime_s = max(
            0.0, float(rospy.get_param("~blocking_obstacles_marker_lifetime_s", 0.8))
        )
        self.obstacle_min_z = float(rospy.get_param("~obstacle_min_z", -0.15))
        self.obstacle_max_z = float(rospy.get_param("~obstacle_max_z", 2.2))
        self.enable_slope_compensation = bool(
            rospy.get_param("~enable_slope_compensation", True)
        )
        self.slope_compensation_max_abs_rad = math.radians(
            max(0.0, float(rospy.get_param("~slope_compensation_max_abs_deg", 25.0)))
        )
        self.enable_ground_band_rejection = bool(
            rospy.get_param("~enable_ground_band_rejection", True)
        )
        self.lidar_height_m = max(
            0.0, float(rospy.get_param("~lidar_height_m", 0.525))
        )
        self.ground_reject_min_m = float(
            rospy.get_param("~ground_reject_min_m", -0.20)
        )
        self.ground_reject_max_m = float(
            rospy.get_param("~ground_reject_max_m", 0.04)
        )
        self.obstacle_max_range_m = max(1.0, float(rospy.get_param("~obstacle_max_range_m", 12.0)))
        self.obstacle_downsample = max(1, int(rospy.get_param("~obstacle_downsample", 6)))
        self.pointcloud_cluster_resolution_m = max(
            0.05, float(rospy.get_param("~pointcloud_cluster_resolution_m", 0.15))
        )
        self.pointcloud_min_cluster_points = max(
            1, int(rospy.get_param("~pointcloud_min_cluster_points", 4))
        )
        self.pointcloud_visibility_hold_s = max(
            0.0, float(rospy.get_param("~pointcloud_visibility_hold_s", 0.45))
        )
        self.raw_near_obstacle_hold_s = max(
            0.0,
            float(
                rospy.get_param(
                    "~raw_near_obstacle_hold_s",
                    self.pointcloud_visibility_hold_s,
                )
            ),
        )
        self.static_obstacle_memory_enabled = bool(
            rospy.get_param("~static_obstacle_memory_enabled", True)
        )
        self.static_obstacle_memory_ttl_s = max(
            0.0, float(rospy.get_param("~static_obstacle_memory_ttl_s", 1.2))
        )
        self.static_obstacle_memory_merge_radius_m = max(
            0.05, float(rospy.get_param("~static_obstacle_memory_merge_radius_m", 0.22))
        )
        self.static_obstacle_memory_max_range_m = max(
            0.5, float(rospy.get_param("~static_obstacle_memory_max_range_m", 3.0))
        )
        self.static_obstacle_memory_max_support = max(
            self.pointcloud_min_cluster_points,
            int(rospy.get_param("~static_obstacle_memory_max_support", 8)),
        )
        self.static_obstacle_memory_max_z_m = float(
            rospy.get_param("~static_obstacle_memory_max_z_m", 0.75)
        )
        self.static_obstacle_memory_max_points = max(
            0, int(rospy.get_param("~static_obstacle_memory_max_points", 40))
        )
        self.static_obstacle_memory_persistence_frames = max(
            1, int(rospy.get_param("~static_obstacle_memory_persistence_frames", 3))
        )
        self.static_obstacle_memory_lock_ttl_s = max(
            self.static_obstacle_memory_ttl_s,
            float(rospy.get_param("~static_obstacle_memory_lock_ttl_s", 8.0)),
        )
        self.static_obstacle_memory_locked_keep_range_m = max(
            self.static_obstacle_memory_max_range_m,
            float(
                rospy.get_param(
                    "~static_obstacle_memory_locked_keep_range_m", 4.5
                )
            ),
        )
        self.static_obstacle_memory_blind_zone_radius_m = max(
            0.0,
            float(
                rospy.get_param(
                    "~static_obstacle_memory_blind_zone_radius_m", 1.40
                )
            ),
        )
        self.static_obstacle_memory_blind_zone_hold_ttl_s = max(
            self.static_obstacle_memory_ttl_s,
            float(
                rospy.get_param(
                    "~static_obstacle_memory_blind_zone_hold_ttl_s", 5.0
                )
            ),
        )
        self.known_map_subtraction_enabled = bool(
            rospy.get_param("~known_map_subtraction_enabled", True)
        )
        self.known_map_subtraction_radius_m = max(
            0.0, float(rospy.get_param("~known_map_subtraction_radius_m", 0.30))
        )
        self.use_map_filtered_path_obstacle_trigger = bool(
            rospy.get_param("~use_map_filtered_path_obstacle_trigger", True)
        )
        self.map_filtered_path_trigger_margin_m = max(
            0.0,
            float(rospy.get_param("~map_filtered_path_trigger_margin_m", 0.10)),
        )
        self.map_filtered_path_trigger_min_points = max(
            1,
            int(rospy.get_param("~map_filtered_path_trigger_min_points", 3)),
        )
        self.local_blind_zone_guard_enabled = bool(
            rospy.get_param("~local_blind_zone_guard_enabled", True)
        )
        self.local_blind_zone_guard_radius_m = max(
            0.0, float(rospy.get_param("~local_blind_zone_guard_radius_m", 1.40))
        )
        self.local_blind_zone_guard_ttl_s = max(
            0.0, float(rospy.get_param("~local_blind_zone_guard_ttl_s", 0.45))
        )
        self.local_blind_zone_guard_lookahead_m = max(
            0.10, float(rospy.get_param("~local_blind_zone_guard_lookahead_m", 1.0))
        )
        self.local_blind_zone_guard_heading_deadband_deg = max(
            0.0,
            float(
                rospy.get_param("~local_blind_zone_guard_heading_deadband_deg", 12.0)
            ),
        )
        self.local_blind_zone_guard_side_margin_m = max(
            0.0, float(rospy.get_param("~local_blind_zone_guard_side_margin_m", 0.08))
        )
        self.local_blind_zone_guard_side_lateral_limit_m = max(
            self.robot_half_width + self.local_blind_zone_guard_side_margin_m,
            float(
                rospy.get_param(
                    "~local_blind_zone_guard_side_lateral_limit_m",
                    min(
                        self.local_blind_zone_guard_radius_m,
                        self.robot_half_width
                        + self.local_blind_zone_guard_side_margin_m
                        + 0.40,
                    ),
                )
            ),
        )
        self.tracked_object_virtual_obstacles_enabled = bool(
            rospy.get_param("~tracked_object_virtual_obstacles_enabled", False)
        )
        self.tracked_object_avoidance_enabled = bool(
            rospy.get_param("~tracked_object_avoidance_enabled", False)
        )
        self.tracked_object_virtual_max_range_m = max(
            0.5, float(rospy.get_param("~tracked_object_virtual_max_range_m", 4.0))
        )
        self.tracked_object_prediction_horizon_s = max(
            0.0, float(rospy.get_param("~tracked_object_prediction_horizon_s", 1.2))
        )
        self.tracked_object_prediction_step_s = max(
            0.1, float(rospy.get_param("~tracked_object_prediction_step_s", 0.3))
        )
        self.tracked_object_prediction_min_speed_mps = max(
            0.0, float(rospy.get_param("~tracked_object_prediction_min_speed_mps", 0.10))
        )
        self.tracked_object_virtual_margin_m = max(
            0.0, float(rospy.get_param("~tracked_object_virtual_margin_m", 0.20))
        )
        self.near_field_object_memory_enabled = bool(
            rospy.get_param("~near_field_object_memory_enabled", True)
        )
        self.near_field_object_memory_ttl_s = max(
            0.0, float(rospy.get_param("~near_field_object_memory_ttl_s", 1.5))
        )
        self.near_field_object_memory_max_range_m = max(
            0.5, float(rospy.get_param("~near_field_object_memory_max_range_m", 2.5))
        )
        self.near_field_object_memory_merge_radius_m = max(
            0.05, float(rospy.get_param("~near_field_object_memory_merge_radius_m", 0.30))
        )
        self.near_field_object_memory_max_points = max(
            0, int(rospy.get_param("~near_field_object_memory_max_points", 120))
        )
        self.enable_global_pointcloud_overlay = bool(
            rospy.get_param("~enable_global_pointcloud_overlay", False)
        )
        self.global_obstacle_overlay_topic = str(
            rospy.get_param("~global_obstacle_overlay_topic", "/planning/global_obstacle_overlay")
        ).strip()
        self.global_pointcloud_overlay_persistence_frames = max(
            1, int(rospy.get_param("~global_pointcloud_overlay_persistence_frames", 3))
        )
        self.global_pointcloud_overlay_ttl_s = max(
            0.0, float(rospy.get_param("~global_pointcloud_overlay_ttl_s", 2.0))
        )
        self.global_pointcloud_overlay_merge_radius_m = max(
            0.05, float(rospy.get_param("~global_pointcloud_overlay_merge_radius_m", 0.25))
        )
        self.global_pointcloud_overlay_max_range_m = max(
            0.5, float(rospy.get_param("~global_pointcloud_overlay_max_range_m", 8.0))
        )
        self.global_pointcloud_overlay_lookahead_m = max(
            0.5, float(rospy.get_param("~global_pointcloud_overlay_lookahead_m", 8.0))
        )
        self.global_pointcloud_overlay_corridor_margin_m = max(
            0.0, float(rospy.get_param("~global_pointcloud_overlay_corridor_margin_m", 1.0))
        )
        self.global_pointcloud_overlay_max_points = max(
            0, int(rospy.get_param("~global_pointcloud_overlay_max_points", 200))
        )
        self.global_pointcloud_overlay_blind_zone_radius_m = max(
            0.0,
            float(rospy.get_param("~global_pointcloud_overlay_blind_zone_radius_m", 1.40)),
        )
        self.global_pointcloud_overlay_blind_zone_hold_ttl_s = max(
            self.global_pointcloud_overlay_ttl_s,
            float(rospy.get_param("~global_pointcloud_overlay_blind_zone_hold_ttl_s", 6.0)),
        )
        self.use_pointcloud_static_blocking = bool(
            rospy.get_param("~use_pointcloud_static_blocking", True)
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
        self.pointcloud_static_block_margin_m = max(
            0.0, float(rospy.get_param("~pointcloud_static_block_margin_m", 0.05))
        )
        self.obstacle_block_margin_m = max(
            0.0, float(rospy.get_param("~obstacle_block_margin_m", 0.30))
        )
        self.use_pointcloud_avoidance_trigger = bool(
            rospy.get_param("~use_pointcloud_avoidance_trigger", False)
        )
        self.avoidance_trigger_margin_m = max(
            0.0, float(rospy.get_param("~avoidance_trigger_margin_m", 0.25))
        )
        self.avoidance_trigger_ahead_m = max(
            1.0, float(rospy.get_param("~avoidance_trigger_ahead_m", 8.0))
        )
        self.forward_path_obstacle_filter_enabled = bool(
            rospy.get_param("~forward_path_obstacle_filter_enabled", True)
        )
        self.forward_path_obstacle_rear_tolerance_m = max(
            0.0,
            float(rospy.get_param("~forward_path_obstacle_rear_tolerance_m", 0.05)),
        )
        self.risk_block_confirm_cells = max(
            1, int(rospy.get_param("~risk_block_confirm_cells", 2))
        )
        self.pointcloud_block_confirm_points = max(
            1, int(rospy.get_param("~pointcloud_block_confirm_points", 2))
        )
        self.avoidance_min_overlay_points = max(
            1, int(rospy.get_param("~avoidance_min_overlay_points", 3))
        )
        self.avoidance_min_cluster_count = max(
            1, int(rospy.get_param("~avoidance_min_cluster_count", 2))
        )
        self.avoidance_hold_s = max(0.0, float(rospy.get_param("~avoidance_hold_s", 1.5)))
        self.avoidance_clear_confirm_cycles = max(
            1, int(rospy.get_param("~avoidance_clear_confirm_cycles", 6))
        )
        self.avoidance_trigger_confirm_cycles = max(
            1, int(rospy.get_param("~avoidance_trigger_confirm_cycles", 2))
        )
        self.avoidance_trigger_confirm_max_gap_s = max(
            0.0, float(rospy.get_param("~avoidance_trigger_confirm_max_gap_s", 2.5))
        )
        self.blocked_clear_hold_s = max(
            0.0, float(rospy.get_param("~blocked_clear_hold_s", 0.35))
        )
        self.avoidance_reuse_on_failure_s = max(
            0.0, float(rospy.get_param("~avoidance_reuse_on_failure_s", 2.0))
        )
        self.avoidance_reuse_max_deviation_m = max(
            0.0, float(rospy.get_param("~avoidance_reuse_max_deviation_m", 0.8))
        )
        self.avoidance_fast_reuse_enabled = bool(
            rospy.get_param("~avoidance_fast_reuse_enabled", False)
        )
        self.avoidance_fast_reuse_window_s = max(
            0.0, float(rospy.get_param("~avoidance_fast_reuse_window_s", 0.0))
        )
        self.avoidance_keep_until_endpoint_distance_m = max(
            0.10,
            float(
                rospy.get_param(
                    "~avoidance_keep_until_endpoint_distance_m",
                    max(0.70, self.robot_length_m * 1.15),
                )
            ),
        )
        self.avoidance_clear_detour_hold_s = max(
            0.0, float(rospy.get_param("~avoidance_clear_detour_hold_s", 1.2))
        )
        self.allow_avoidance_reuse_on_no_solution = bool(
            rospy.get_param("~allow_avoidance_reuse_on_no_solution", False)
        )
        self.allow_nominal_local_fallback_on_no_solution = bool(
            rospy.get_param(
                "~allow_nominal_local_fallback_on_no_solution", True
            )
        )
        self.weak_grid_no_solution_fallback_enabled = bool(
            rospy.get_param("~weak_grid_no_solution_fallback_enabled", True)
        )
        self.weak_grid_no_solution_fallback_max_cells = max(
            0, int(rospy.get_param("~weak_grid_no_solution_fallback_max_cells", 3))
        )
        self.weak_grid_no_solution_fallback_max_memory_points = max(
            0,
            int(
                rospy.get_param(
                    "~weak_grid_no_solution_fallback_max_memory_points", 1
                )
            ),
        )
        self.avoidance_branch_backtrack_cells = max(
            0, int(rospy.get_param("~avoidance_branch_backtrack_cells", 2))
        )
        self.avoidance_rejoin_min_distance_m = max(
            0.3, float(rospy.get_param("~avoidance_rejoin_min_distance_m", 1.0))
        )
        self.avoidance_same_side_commitment_enabled = bool(
            rospy.get_param("~avoidance_same_side_commitment_enabled", True)
        )
        self.avoidance_memory_only_trigger_enabled = bool(
            rospy.get_param("~avoidance_memory_only_trigger_enabled", False)
        )
        self.avoidance_local_collapse_straights = bool(
            rospy.get_param("~avoidance_local_collapse_straights", True)
        )
        self.short_curved_avoidance_enabled = bool(
            rospy.get_param("~short_curved_avoidance_enabled", True)
        )
        self.short_curved_avoidance_preview_m = max(
            1.2,
            float(
                rospy.get_param(
                    "~short_curved_avoidance_preview_m",
                    max(
                        self.avoidance_rejoin_min_distance_m + self.robot_length_m * 2.0,
                        3.2,
                    ),
                )
            ),
        )
        self.short_curved_avoidance_tail_m = max(
            0.4,
            float(
                rospy.get_param(
                    "~short_curved_avoidance_tail_m",
                    max(self.robot_length_m * 1.6, 1.0),
                )
            ),
        )
        self.short_curved_avoidance_smooth_passes = max(
            1, int(rospy.get_param("~short_curved_avoidance_smooth_passes", 2))
        )
        self.short_curved_avoidance_max_curvature = max(
            0.1,
            float(rospy.get_param("~short_curved_avoidance_max_curvature", 1.35)),
        )
        self.short_curved_avoidance_max_heading_delta_rad = math.radians(
            max(
                5.0,
                float(
                    rospy.get_param(
                        "~short_curved_avoidance_max_heading_delta_deg", 55.0
                    )
                ),
            )
        )
        self.sidestep_avoidance_enabled = bool(
            rospy.get_param("~sidestep_avoidance_enabled", True)
        )
        self.sidestep_avoidance_min_offset_m = max(
            0.10,
            float(
                rospy.get_param(
                    "~sidestep_avoidance_min_offset_m",
                    self.robot_half_width + 0.18,
                )
            ),
        )
        self.sidestep_avoidance_max_offset_m = max(
            self.sidestep_avoidance_min_offset_m + 0.05,
            float(rospy.get_param("~sidestep_avoidance_max_offset_m", 1.15)),
        )
        self.sidestep_avoidance_preview_m = max(
            0.8, float(rospy.get_param("~sidestep_avoidance_preview_m", 2.0))
        )
        self.sidestep_avoidance_forward_margin_m = max(
            0.20,
            float(rospy.get_param("~sidestep_avoidance_forward_margin_m", 0.55)),
        )
        # In global-nominal mode, we need to detect and branch around obstacles
        # earlier than the short controller-facing lookahead, otherwise the
        # robot keeps following the fixed global path until emergency stop.
        self.avoidance_plan_horizon_m = max(
            self.lookahead_m,
            self.avoidance_trigger_ahead_m
            + max(self.avoidance_rejoin_min_distance_m, self.robot_length_m * 1.5),
        )
        self.rejoin_mode_hold_s = max(
            0.0, float(rospy.get_param("~rejoin_mode_hold_s", 1.0))
        )
        self.blocked_stop_before_avoidance_s = max(
            0.0, float(rospy.get_param("~blocked_stop_before_avoidance_s", 0.0))
        )
        self.near_goal_block_ignore_distance_m = max(
            0.0, float(rospy.get_param("~near_goal_block_ignore_distance_m", 1.0))
        )
        self.near_goal_tail_block_ignore_distance_m = max(
            0.0, float(rospy.get_param("~near_goal_tail_block_ignore_distance_m", 0.45))
        )
        self.near_goal_block_ignore_after_avoidance_s = max(
            0.0,
            float(rospy.get_param("~near_goal_block_ignore_after_avoidance_s", 1.0)),
        )
        self.near_goal_recent_avoidance_release_distance_m = max(
            0.0,
            float(rospy.get_param("~near_goal_recent_avoidance_release_distance_m", 0.30)),
        )
        self.self_filter_radius_x = max(
            0.0, float(rospy.get_param("~self_filter_radius_x", 0.5 * self.robot_length_m))
        )
        self.self_filter_radius_y = max(
            0.0, float(rospy.get_param("~self_filter_radius_y", 0.5 * self.robot_width_m))
        )
        self.debug_avoidance_logging = bool(rospy.get_param("~debug_avoidance_logging", True))
        self.debug_avoidance_log_period_s = max(
            0.1, float(rospy.get_param("~debug_avoidance_log_period_s", 1.0))
        )
        self.debug_timing_logging = bool(rospy.get_param("~debug_timing_logging", True))
        self.debug_timing_log_period_s = max(
            0.2, float(rospy.get_param("~debug_timing_log_period_s", 1.0))
        )
        self.debug_timing_overrun_s = max(
            0.02,
            float(
                rospy.get_param(
                    "~debug_timing_overrun_s",
                    max(0.12, 0.75 * self.replan_period_s),
                )
            ),
        )

        self.have_odom = False
        self.odom_stamp_sec = 0.0
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.odom_roll = 0.0
        self.odom_pitch = 0.0
        self.global_path = None
        self.drivable_grid = None
        self.risk_grid = None
        self.drivable_grid_signature = None
        self.risk_grid_signature = None
        self._inflated_blocked_cache = {}
        self._inflated_blocked_cache_max_entries = 6
        self.direct_goal = None
        self.direct_goal_stamp_sec = 0.0
        self.cached_direct_goal_cell = None
        self.frozen_direct_goal_cell = None
        self.frozen_direct_grid_path = None
        self.frozen_direct_start_xy = None
        self.frozen_direct_goal_xy = None
        self.last_published_goal_cell = None
        self.last_published_end_cell = None
        self.last_path_publish_sec = 0.0
        self.last_local_path_msg = None
        self.last_local_path_geometry_sec = 0.0
        self.last_local_path_publish_sec = 0.0
        self.current_obstacle_points_map = []
        self.current_obstacle_points_stamp_sec = 0.0
        self.last_nonempty_current_obstacle_points_map = []
        self.last_nonempty_current_obstacle_points_stamp_sec = 0.0
        self.raw_near_obstacle_points_map = []
        self.raw_near_obstacle_points_stamp_sec = 0.0
        self.last_nonempty_raw_near_obstacle_points_map = []
        self.last_nonempty_raw_near_obstacle_points_stamp_sec = 0.0
        self.obstacle_points_map = []
        self.obstacle_raw_point_count = 0
        self.obstacle_cluster_count = 0
        self.obstacle_memory_points = []
        self.obstacle_memory_count = 0
        self.obstacle_memory_locked_count = 0
        self.known_map_filtered_count = 0
        self.known_map_filtered_points_map = []
        self.next_obstacle_memory_id = 1
        self.current_tracked_object_points_map = []
        self.tracked_object_points_map = []
        self.tracked_object_memory_points = []
        self.tracked_object_memory_count = 0
        self.tracked_object_count = 0
        self.global_obstacle_overlay_memory = []
        self.global_obstacle_overlay_points_map = []
        self.global_obstacle_overlay_candidate_count = 0
        self.global_obstacle_overlay_confirmed_count = 0
        self.avoidance_active = False
        self.avoidance_clear_count = 0
        self.last_avoidance_publish_sec = 0.0
        self.last_avoidance_grid_path = None
        self.last_avoidance_world_path = None
        self.last_avoidance_solution_sec = 0.0
        self.last_avoidance_validation_sec = 0.0
        self.last_avoidance_active_sec = 0.0
        self.active_avoidance_obstacle_key = None
        self.pending_avoidance_trigger_key = None
        self.pending_avoidance_trigger_count = 0
        self.pending_avoidance_trigger_stamp_sec = 0.0
        self.local_blocked_since_sec = 0.0
        self.local_clear_since_sec = 0.0
        self.last_avoidance_trigger_reason = ""
        self.last_avoidance_direction = "none"
        self.global_path_signature = None
        self.global_nominal_progress_idx = 0
        # Default to following the replanner's nominal local segment.  The
        # global route should supply direction, while the local replanner owns
        # the short-horizon maneuvering path that the controller actually
        # tracks outdoors.
        self.current_path_mode = (
            "follow_global"
            if self.nominal_path_reference_mode == "global"
            else "follow_local"
        )
        self.rejoin_mode_until_sec = 0.0
        self._last_explain_key = None
        self._last_explain_time = 0.0
        self.path_history_entries = deque(maxlen=self.path_history_max_paths)
        self.path_history_next_id = 0
        self.last_history_signature = {
            "local": None,
            "avoidance": None,
            "used_local": None,
        }
        self.pending_used_local_points = None
        self.pending_used_local_frame_id = "map"
        self.pending_used_local_origin_xy = None
        self.pending_used_local_signature = None
        self.pending_used_local_committed = False
        self.travel_history_points = deque(maxlen=self.travel_history_max_points)
        self.debug_text_topic = str(
            rospy.get_param("~debug_text_topic", "/planning/local_replanner_debug_text")
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
        self.explainability_topic = rospy.get_param(
            "~explainability_topic", "/planning/explainability"
        )
        self._last_debug_text = ""
        self._last_debug_text_time = 0.0
        self._last_debug_screen_time = 0.0
        self._last_timer_start_sec = 0.0

        self.pub_local_path = rospy.Publisher(self.local_path_topic, Path, queue_size=2)
        self.pub_avoidance_path = rospy.Publisher(self.avoidance_path_topic, Path, queue_size=2)
        self.pub_path_mode = rospy.Publisher(self.path_mode_topic, String, queue_size=4, latch=True)
        self.pub_path_history = rospy.Publisher(
            self.path_history_topic, MarkerArray, queue_size=2, latch=True
        )
        self.pub_travel_history = None
        self.pub_travel_history_path = None
        if self.travel_history_topic:
            self.pub_travel_history = rospy.Publisher(
                self.travel_history_topic, Marker, queue_size=2, latch=True
            )
        if self.travel_history_path_topic:
            self.pub_travel_history_path = rospy.Publisher(
                self.travel_history_path_topic, Path, queue_size=2, latch=True
            )
        self.pub_explainability = rospy.Publisher(
            self.explainability_topic, ExplainabilityEvent, queue_size=20
        )
        self.pub_recognized_obstacles = None
        if self.recognized_obstacles_marker_topic:
            self.pub_recognized_obstacles = rospy.Publisher(
                self.recognized_obstacles_marker_topic,
                MarkerArray,
                queue_size=2,
            )
        self.pub_blocking_obstacles = None
        if self.blocking_obstacles_marker_topic:
            self.pub_blocking_obstacles = rospy.Publisher(
                self.blocking_obstacles_marker_topic,
                MarkerArray,
                queue_size=2,
            )
        self.pub_global_obstacle_overlay = None
        if self.enable_global_pointcloud_overlay and self.global_obstacle_overlay_topic:
            self.pub_global_obstacle_overlay = rospy.Publisher(
                self.global_obstacle_overlay_topic, OccupancyGrid, queue_size=1
            )
        self.pub_debug_text = None
        if self.debug_text_topic:
            self.pub_debug_text = rospy.Publisher(
                self.debug_text_topic, String, queue_size=20
            )
        self.sub_odom = rospy.Subscriber(
            self.odom_topic,
            Odometry,
            self.odom_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.sub_global = rospy.Subscriber(self.global_path_topic, Path, self.global_path_callback, queue_size=5)
        self.sub_drivable = rospy.Subscriber(self.drivable_grid_topic, OccupancyGrid, self.drivable_grid_callback, queue_size=3)
        self.sub_risk = rospy.Subscriber(self.dynamic_risk_grid_topic, OccupancyGrid, self.risk_grid_callback, queue_size=3)
        self.sub_tracked_objects = rospy.Subscriber(
            self.tracked_objects_topic,
            TrackedObjectArray,
            self.tracked_objects_callback,
            queue_size=3,
        )
        self.sub_cloud = rospy.Subscriber(
            self.obstacle_pointcloud_topic,
            PointCloud2,
            self.cloud_callback,
            queue_size=1,
        )
        self.sub_raw_near_obstacle_hits = None
        if self.use_raw_near_obstacle_hits and self.raw_near_obstacle_hits_topic:
            self.sub_raw_near_obstacle_hits = rospy.Subscriber(
                self.raw_near_obstacle_hits_topic,
                PointCloud2,
                self.raw_near_obstacle_hits_callback,
                queue_size=1,
            )
        self.sub_direct_goal = None
        if self.use_direct_goal:
            self.sub_direct_goal = rospy.Subscriber(self.direct_goal_topic, PoseStamped, self.direct_goal_callback, queue_size=2)

        self._clear_path_history()
        self._clear_travel_history()
        self._publish_path_mode(self.current_path_mode, force=True)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.replan_hz), self.on_timer)
        self.local_path_keepalive_timer = None
        if self.local_path_keepalive_enabled:
            self.local_path_keepalive_timer = rospy.Timer(
                rospy.Duration(1.0 / self.local_path_keepalive_hz),
                self.on_local_path_keepalive,
            )
        rospy.loginfo(
            "constrained_local_replanner started | profile=%s real_mode=%s env=%s map=%s map_path=%s state=%s global=%s drivable=%s risk=%s local=%s avoidance=%s nominal_ref=%s direct_goal=%s(%s) footprint=%.2fm x %.2fm freeze_first=%s avoid=%s pc_static=%s pc_trigger=%s grid_only_fallback=%s grid_only_search=%s nominal_no_solution_fallback=%s timing_log=%s/%.1fs",
            self.launch_profile_label,
            self.launch_real_mode,
            self.launch_localization_environment,
            self.launch_map_profile_name,
            self.launch_localizer_map_relative_path,
            self.launch_runtime_drivable_state_file,
            self.global_path_topic,
            self.drivable_grid_topic,
            self.dynamic_risk_grid_topic,
            self.local_path_topic,
            self.avoidance_path_topic,
            self.nominal_path_reference_mode,
            "on" if self.use_direct_goal else "off",
            self.direct_goal_topic,
            self.robot_length_m,
            self.robot_width_m,
            "on" if self.freeze_path_on_first_plan else "off",
            "on" if self.enable_avoidance_path else "off",
            "on" if self.use_pointcloud_static_blocking else "off",
            "on" if self.use_pointcloud_avoidance_trigger else "off",
            "on" if self.allow_grid_only_nominal_fallback else "off",
            "on" if self.grid_only_avoidance_search_enabled else "off",
            "on" if self.allow_nominal_local_fallback_on_no_solution else "off",
            "on" if self.debug_timing_logging else "off",
            self.debug_timing_log_period_s,
        )
        if self.enable_global_pointcloud_overlay and self.global_obstacle_overlay_topic:
            rospy.loginfo(
                "constrained_local_replanner global obstacle overlay | topic=%s persist=%d ttl=%.1fs blind_ttl=%.1fs blind_radius=%.2fm range=%.1fm lookahead=%.1fm corridor_margin=%.2fm",
                self.global_obstacle_overlay_topic,
                self.global_pointcloud_overlay_persistence_frames,
                self.global_pointcloud_overlay_ttl_s,
                self.global_pointcloud_overlay_blind_zone_hold_ttl_s,
                self.global_pointcloud_overlay_blind_zone_radius_m,
                self.global_pointcloud_overlay_max_range_m,
                self.global_pointcloud_overlay_lookahead_m,
                self.global_pointcloud_overlay_corridor_margin_m,
            )
        if self.use_raw_near_obstacle_hits and self.raw_near_obstacle_hits_topic:
            rospy.loginfo(
                "constrained_local_replanner raw-near obstacle hits | topic=%s hold=%.2fs",
                self.raw_near_obstacle_hits_topic,
                self.raw_near_obstacle_hold_s,
            )
        if self.local_blind_zone_guard_enabled:
            rospy.loginfo(
                "constrained_local_replanner local blind zone | radius=%.2fm ttl=%.2fs lookahead=%.2fm side_limit=%.2fm side_max=%.2fm",
                self.local_blind_zone_guard_radius_m,
                self.local_blind_zone_guard_ttl_s,
                self.local_blind_zone_guard_lookahead_m,
                self.robot_half_width + self.local_blind_zone_guard_side_margin_m,
                self.local_blind_zone_guard_side_lateral_limit_m,
            )
        if self.static_obstacle_memory_enabled:
            rospy.loginfo(
                "constrained_local_replanner static obstacle memory | ttl=%.1fs lock_ttl=%.1fs blind_ttl=%.1fs blind_radius=%.2fm persist=%d locked_keep=%.1fm range=%.1fm map_subtract=%s radius=%.2fm map_filtered_path_trigger=%s min=%d margin=%.2fm",
                self.static_obstacle_memory_ttl_s,
                self.static_obstacle_memory_lock_ttl_s,
                self.static_obstacle_memory_blind_zone_hold_ttl_s,
                self.static_obstacle_memory_blind_zone_radius_m,
                self.static_obstacle_memory_persistence_frames,
                self.static_obstacle_memory_locked_keep_range_m,
                self.static_obstacle_memory_max_range_m,
                "on" if self.known_map_subtraction_enabled else "off",
                self.known_map_subtraction_radius_m,
                "on" if self.use_map_filtered_path_obstacle_trigger else "off",
                self.map_filtered_path_trigger_min_points,
                self.map_filtered_path_trigger_margin_m,
            )
        if self.tracked_object_virtual_obstacles_enabled or self.near_field_object_memory_enabled:
            rospy.loginfo(
                "constrained_local_replanner tracked objects | topic=%s virtual=%s avoidance=%s range=%.1fm horizon=%.1fs step=%.2fs min_speed=%.2fmps margin=%.2fm memory=%s ttl=%.1fs near_range=%.1fm",
                self.tracked_objects_topic,
                "on" if self.tracked_object_virtual_obstacles_enabled else "off",
                "on" if self.tracked_object_avoidance_enabled else "off",
                self.tracked_object_virtual_max_range_m,
                self.tracked_object_prediction_horizon_s,
                self.tracked_object_prediction_step_s,
                self.tracked_object_prediction_min_speed_mps,
                self.tracked_object_virtual_margin_m,
                "on" if self.near_field_object_memory_enabled else "off",
                self.near_field_object_memory_ttl_s,
                self.near_field_object_memory_max_range_m,
            )
        if self.pub_recognized_obstacles is not None:
            rospy.loginfo(
                "constrained_local_replanner recognized obstacle markers | topic=%s max_points=%d scale=%.2fm lifetime=%.1fs",
                self.recognized_obstacles_marker_topic,
                self.recognized_obstacles_marker_max_points,
                self.recognized_obstacles_marker_scale_m,
                self.recognized_obstacles_marker_lifetime_s,
            )
        if self.pub_blocking_obstacles is not None:
            rospy.loginfo(
                "constrained_local_replanner blocking obstacle markers | topic=%s max_points=%d scale=%.2fm lifetime=%.1fs",
                self.blocking_obstacles_marker_topic,
                self.blocking_obstacles_marker_max_points,
                self.blocking_obstacles_marker_scale_m,
                self.blocking_obstacles_marker_lifetime_s,
            )

    def _publish_path_mode(self, mode, force=False):
        mode_str = str(mode or "hold").strip().lower()
        if not force and mode_str == self.current_path_mode:
            return
        self.current_path_mode = mode_str
        if rospy.is_shutdown():
            return
        try:
            self.pub_path_mode.publish(String(data=mode_str))
        except rospy.ROSException:
            return
        rospy.loginfo("constrained_local_replanner: path_mode=%s", mode_str)

    def _remember_local_path_msg(self, msg):
        self.last_local_path_msg = msg
        now_sec = rospy.Time.now().to_sec()
        self.last_local_path_geometry_sec = now_sec
        self.last_local_path_publish_sec = now_sec

    def _forget_local_path_msg(self):
        self.last_local_path_msg = None
        self.last_local_path_geometry_sec = 0.0
        self.last_local_path_publish_sec = 0.0

    def _publish_local_path_msg(self, msg):
        if msg is None or len(getattr(msg, "poses", [])) < 2:
            return
        self.pub_local_path.publish(msg)
        self._remember_local_path_msg(msg)

    def on_local_path_keepalive(self, _event):
        if (not self.local_path_keepalive_enabled) or rospy.is_shutdown():
            return
        src_msg = self.last_local_path_msg
        if src_msg is None or len(getattr(src_msg, "poses", [])) < 2:
            return
        if self.current_path_mode not in ("follow_local", "follow_avoidance", "rejoin_global"):
            return

        now = rospy.Time.now()
        now_sec = now.to_sec()
        if self.last_local_path_geometry_sec <= 0.0:
            return
        geometry_age_s = now_sec - self.last_local_path_geometry_sec
        if (
            self.local_path_keepalive_max_age_s > 0.0
            and geometry_age_s > self.local_path_keepalive_max_age_s
        ):
            return
        min_publish_period_s = 1.0 / max(0.5, self.local_path_keepalive_hz)
        if (
            self.last_local_path_publish_sec > 0.0
            and (now_sec - self.last_local_path_publish_sec) < 0.8 * min_publish_period_s
        ):
            return

        msg = Path()
        msg.header.stamp = now
        msg.header.frame_id = src_msg.header.frame_id
        geometry_stamp = src_msg.poses[0].header.stamp
        if geometry_stamp.to_sec() <= 0.0:
            geometry_stamp = src_msg.header.stamp
        for src_pose in src_msg.poses:
            pose = PoseStamped()
            pose.header.frame_id = src_pose.header.frame_id or src_msg.header.frame_id
            pose.header.stamp = geometry_stamp
            pose.pose = src_pose.pose
            msg.poses.append(pose)
        try:
            self.pub_local_path.publish(msg)
            self.last_local_path_publish_sec = now_sec
        except rospy.ROSException:
            return

    def _use_global_nominal_reference(self):
        return self.nominal_path_reference_mode == "global"

    def _publish_nominal_reference_path(self, world_points, frame_id, stamp):
        # Keep a rolling local nominal path alive at all times.  The fixed
        # global path remains the long-horizon guide, but the controller should
        # always track a short local segment that can bend early around
        # obstacles.
        self._publish_world_path(world_points, frame_id, stamp)
        self._publish_path_mode("follow_local")

    def _arm_rejoin_mode(self, now_sec):
        if self.rejoin_mode_hold_s <= 1e-6:
            self.rejoin_mode_until_sec = 0.0
            return False
        self.rejoin_mode_until_sec = max(0.0, float(now_sec)) + self.rejoin_mode_hold_s
        return True

    def _is_rejoin_mode_active(self, now_sec):
        return self.rejoin_mode_until_sec > 0.0 and float(now_sec) < self.rejoin_mode_until_sec

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
    def _sample_marker_points(points, max_points):
        pts = list(points)
        if max_points <= 0 or len(pts) <= max_points:
            return pts
        stride = max(1, int(math.ceil(float(len(pts)) / float(max_points))))
        sampled = pts[::stride]
        if sampled and sampled[-1] != pts[-1]:
            sampled.append(pts[-1])
        return sampled[:max_points]

    def _make_obstacle_points_marker(
        self,
        stamp,
        marker_id,
        namespace,
        points,
        scale_m,
        color_rgba,
        z_offset,
        *,
        max_points=None,
        lifetime_s=None,
    ):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = "map"
        marker.ns = namespace
        marker.id = int(marker_id)
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(scale_m)
        marker.scale.y = float(scale_m)
        marker.color.r = float(color_rgba[0])
        marker.color.g = float(color_rgba[1])
        marker.color.b = float(color_rgba[2])
        marker.color.a = float(color_rgba[3])
        marker.lifetime = rospy.Duration(
            self.recognized_obstacles_marker_lifetime_s
            if lifetime_s is None
            else max(0.0, float(lifetime_s))
        )
        for wx, wy in self._sample_marker_points(
            points,
            self.recognized_obstacles_marker_max_points
            if max_points is None
            else max(1, int(max_points)),
        ):
            pt = Point()
            pt.x = float(wx)
            pt.y = float(wy)
            pt.z = float(z_offset)
            marker.points.append(pt)
        return marker

    def _make_obstacle_sphere_marker(
        self,
        stamp,
        marker_id,
        namespace,
        wx,
        wy,
        diameter_m,
        color_rgba,
        z_offset,
        *,
        lifetime_s=None,
    ):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = "map"
        marker.ns = namespace
        marker.id = int(marker_id)
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = float(wx)
        marker.pose.position.y = float(wy)
        marker.pose.position.z = float(z_offset)
        marker.scale.x = float(diameter_m)
        marker.scale.y = float(diameter_m)
        marker.scale.z = float(diameter_m)
        marker.color.r = float(color_rgba[0])
        marker.color.g = float(color_rgba[1])
        marker.color.b = float(color_rgba[2])
        marker.color.a = float(color_rgba[3])
        marker.lifetime = rospy.Duration(
            self.recognized_obstacles_marker_lifetime_s
            if lifetime_s is None
            else max(0.0, float(lifetime_s))
        )
        return marker

    def _make_obstacle_cube_list_marker(
        self,
        stamp,
        marker_id,
        namespace,
        points,
        scale_xy_m,
        scale_z_m,
        color_rgba,
        z_offset,
        *,
        max_points=None,
        lifetime_s=None,
    ):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = "map"
        marker.ns = namespace
        marker.id = int(marker_id)
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(scale_xy_m)
        marker.scale.y = float(scale_xy_m)
        marker.scale.z = float(scale_z_m)
        marker.color.r = float(color_rgba[0])
        marker.color.g = float(color_rgba[1])
        marker.color.b = float(color_rgba[2])
        marker.color.a = float(color_rgba[3])
        marker.lifetime = rospy.Duration(
            self.recognized_obstacles_marker_lifetime_s
            if lifetime_s is None
            else max(0.0, float(lifetime_s))
        )
        for wx, wy in self._sample_marker_points(
            points,
            self.recognized_obstacles_marker_max_points
            if max_points is None
            else max(1, int(max_points)),
        ):
            pt = Point()
            pt.x = float(wx)
            pt.y = float(wy)
            pt.z = float(z_offset)
            marker.points.append(pt)
        return marker

    def _publish_recognized_obstacle_markers(self, stamp=None):
        if self.pub_recognized_obstacles is None or rospy.is_shutdown():
            return

        marker_stamp = stamp if hasattr(stamp, "to_sec") else rospy.Time.now()
        if marker_stamp.to_sec() <= 0.0:
            marker_stamp = rospy.Time.now()

        markers = MarkerArray()
        delete_all = Marker()
        delete_all.header.stamp = marker_stamp
        delete_all.header.frame_id = "map"
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        scale = self.recognized_obstacles_marker_scale_m
        current_static_points = self._effective_current_obstacle_points_map(
            marker_stamp.to_sec()
        )
        static_memory_points = self._memory_points_from_entries(
            self.obstacle_memory_points, confirmed_only=True
        )
        tracked_points = list(self.tracked_object_points_map)
        overlay_points = list(self.global_obstacle_overlay_points_map)

        if current_static_points:
            markers.markers.append(
                self._make_obstacle_points_marker(
                    marker_stamp,
                    10,
                    "static_current",
                    current_static_points,
                    scale,
                    (0.74, 0.28, 0.98, 0.95),
                    0.05,
                )
            )
        if static_memory_points:
            markers.markers.append(
                self._make_obstacle_points_marker(
                    marker_stamp,
                    20,
                    "static_memory",
                    static_memory_points,
                    scale * 0.9,
                    (0.60, 0.24, 0.88, 0.45),
                    0.08,
                )
            )
        if tracked_points:
            markers.markers.append(
                self._make_obstacle_points_marker(
                    marker_stamp,
                    30,
                    "tracked_dynamic",
                    tracked_points,
                    scale * 0.95,
                    (0.92, 0.10, 0.95, 0.90),
                    0.11,
                )
            )
        if overlay_points:
            markers.markers.append(
                self._make_obstacle_points_marker(
                    marker_stamp,
                    40,
                    "global_overlay",
                    overlay_points,
                    scale * 1.10,
                    (0.52, 0.18, 0.96, 0.85),
                    0.14,
                )
            )

        blind_zone_memory = self._get_active_local_blind_zone_memory(
            now_sec=marker_stamp.to_sec()
        )
        if blind_zone_memory is not None:
            markers.markers.append(
                self._make_obstacle_sphere_marker(
                    marker_stamp,
                    50,
                    "blind_zone_focus",
                    blind_zone_memory["world_x"],
                    blind_zone_memory["world_y"],
                    max(0.14, scale * 1.8),
                    (1.0, 0.15, 1.0, 0.95),
                    0.18,
                )
            )

        self.pub_recognized_obstacles.publish(markers)

    @staticmethod
    def _build_marker_delete_all(stamp):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = "map"
        marker.action = Marker.DELETEALL
        return marker

    @staticmethod
    def _memory_entry_position(entry):
        if isinstance(entry, dict):
            return float(entry.get("x", 0.0)), float(entry.get("y", 0.0))
        wx, wy, _seen_sec = entry
        return float(wx), float(wy)

    @staticmethod
    def _memory_entry_last_seen(entry):
        if isinstance(entry, dict):
            return float(entry.get("last_seen", 0.0))
        _wx, _wy, seen_sec = entry
        return float(seen_sec)

    @staticmethod
    def _memory_entry_hits(entry):
        if isinstance(entry, dict):
            return int(entry.get("hits", 1))
        return 1

    @staticmethod
    def _memory_entry_locked(entry):
        if isinstance(entry, dict):
            return bool(entry.get("locked", False))
        return False

    def _is_confirmed_static_obstacle_entry(self, entry):
        return self._memory_entry_locked(entry) or (
            self._memory_entry_hits(entry)
            >= self.static_obstacle_memory_persistence_frames
        )

    def _memory_points_from_entries(self, entries, confirmed_only=False):
        points = []
        for entry in entries:
            if confirmed_only and (not self._is_confirmed_static_obstacle_entry(entry)):
                continue
            wx, wy = self._memory_entry_position(entry)
            points.append((wx, wy))
        return points

    def _relevant_static_memory_entries(
        self,
        blocking_points_world=None,
        blocking_cells_world=None,
    ):
        source_points = []
        if blocking_points_world:
            source_points.extend(
                (float(wx), float(wy)) for wx, wy in list(blocking_points_world)
            )
        if blocking_cells_world:
            source_points.extend(
                (float(wx), float(wy)) for wx, wy in list(blocking_cells_world)
            )
        if not source_points:
            return []

        confirmed_entries = [
            entry
            for entry in self.obstacle_memory_points
            if self._is_confirmed_static_obstacle_entry(entry)
        ]
        if not confirmed_entries:
            return []

        match_radius_m = max(
            self.static_obstacle_memory_merge_radius_m * 2.5,
            self.pointcloud_cluster_resolution_m * 2.5,
            self.obstacle_block_margin_m + 0.20,
        )
        match_radius_sq = match_radius_m * match_radius_m
        matched = []
        for entry in confirmed_entries:
            wx, wy = self._memory_entry_position(entry)
            best_d2 = None
            for sx, sy in source_points:
                d2 = (float(wx) - float(sx)) * (float(wx) - float(sx)) + (
                    float(wy) - float(sy)
                ) * (float(wy) - float(sy))
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
            if best_d2 is None or best_d2 > match_radius_sq:
                continue
            matched.append((best_d2, entry))
        if not matched:
            return []

        matched.sort(
            key=lambda item: (
                item[0],
                -self._memory_entry_hits(item[1]),
                -self._memory_entry_last_seen(item[1]),
            )
        )
        primary_entry = matched[0][1]
        primary_x, primary_y = self._memory_entry_position(primary_entry)
        group_radius_m = max(
            self.static_obstacle_memory_merge_radius_m * 3.0,
            self.pointcloud_cluster_resolution_m * 3.0,
            0.40,
        )
        group_radius_sq = group_radius_m * group_radius_m
        grouped = []
        seen_ids = set()
        for _best_d2, entry in matched:
            wx, wy = self._memory_entry_position(entry)
            d2 = (float(wx) - float(primary_x)) * (float(wx) - float(primary_x)) + (
                float(wy) - float(primary_y)
            ) * (float(wy) - float(primary_y))
            if d2 > group_radius_sq:
                continue
            entry_id = int(entry.get("id", -1)) if isinstance(entry, dict) else -1
            if entry_id >= 0 and entry_id in seen_ids:
                continue
            if entry_id >= 0:
                seen_ids.add(entry_id)
            grouped.append(entry)
        grouped.sort(
            key=lambda entry: int(entry.get("id", -1)) if isinstance(entry, dict) else -1
        )
        return grouped

    def _stable_avoidance_obstacle_points(
        self,
        blocking_cells_world=None,
        blocking_points_world=None,
    ):
        matched_entries = self._relevant_static_memory_entries(
            blocking_points_world=blocking_points_world,
            blocking_cells_world=blocking_cells_world,
        )

        stable_points = []
        obstacle_key = None
        if matched_entries:
            stable_points.extend(
                self._memory_points_from_entries(matched_entries, confirmed_only=False)
            )
            raw_candidates = []
            if blocking_points_world:
                raw_candidates.extend(list(blocking_points_world))
            if blocking_cells_world:
                raw_candidates.extend(list(blocking_cells_world))
            attach_radius_m = max(
                self.static_obstacle_memory_merge_radius_m * 2.0,
                self.pointcloud_cluster_resolution_m * 2.0,
                0.30,
            )
            attach_radius_sq = attach_radius_m * attach_radius_m
            for wx, wy in raw_candidates:
                for ex, ey in stable_points:
                    d2 = (float(wx) - float(ex)) * (float(wx) - float(ex)) + (
                        float(wy) - float(ey)
                    ) * (float(wy) - float(ey))
                    if d2 <= attach_radius_sq:
                        stable_points.append((float(wx), float(wy)))
                        break
            entry_ids = [
                int(entry.get("id", -1))
                for entry in matched_entries
                if isinstance(entry, dict) and int(entry.get("id", -1)) >= 0
            ]
            if entry_ids:
                obstacle_key = ("static", tuple(sorted(entry_ids)))

        if not stable_points:
            if blocking_points_world:
                stable_points.extend(
                    (float(wx), float(wy)) for wx, wy in list(blocking_points_world)
                )
            if blocking_cells_world:
                stable_points.extend(
                    (float(wx), float(wy)) for wx, wy in list(blocking_cells_world)
                )
            if stable_points:
                sample = stable_points[: min(6, len(stable_points))]
                cx = sum(float(wx) for wx, _wy in sample) / float(len(sample))
                cy = sum(float(wy) for _wx, wy in sample) / float(len(sample))
                obstacle_key = ("raw", round(cx * 5.0) / 5.0, round(cy * 5.0) / 5.0)

        stable_points = self._dedupe_world_points(stable_points)
        return obstacle_key, stable_points, matched_entries

    def _clear_blocking_obstacle_markers(self, stamp=None):
        if self.pub_blocking_obstacles is None or rospy.is_shutdown():
            return

        marker_stamp = stamp if hasattr(stamp, "to_sec") else rospy.Time.now()
        if marker_stamp.to_sec() <= 0.0:
            marker_stamp = rospy.Time.now()

        markers = MarkerArray()
        markers.markers.append(self._build_marker_delete_all(marker_stamp))
        self.pub_blocking_obstacles.publish(markers)

    def _publish_blocking_obstacle_markers(
        self,
        stamp=None,
        *,
        blocking_points=None,
        blocked_cells=None,
        blind_zone_conflict=None,
        cell_scale_m=None,
    ):
        if self.pub_blocking_obstacles is None or rospy.is_shutdown():
            return

        marker_stamp = stamp if hasattr(stamp, "to_sec") else rospy.Time.now()
        if marker_stamp.to_sec() <= 0.0:
            marker_stamp = rospy.Time.now()

        markers = MarkerArray()
        markers.markers.append(self._build_marker_delete_all(marker_stamp))

        if blocking_points:
            markers.markers.append(
                self._make_obstacle_points_marker(
                    marker_stamp,
                    110,
                    "blocking_points",
                    blocking_points,
                    self.blocking_obstacles_marker_scale_m,
                    (0.88, 0.18, 0.96, 0.95),
                    0.07,
                    max_points=self.blocking_obstacles_marker_max_points,
                    lifetime_s=self.blocking_obstacles_marker_lifetime_s,
                )
            )
        if blocked_cells:
            cell_scale = (
                self.blocking_obstacles_marker_scale_m
                if cell_scale_m is None
                else max(0.02, float(cell_scale_m))
            )
            markers.markers.append(
                self._make_obstacle_cube_list_marker(
                    marker_stamp,
                    120,
                    "blocked_path_cells",
                    blocked_cells,
                    cell_scale,
                    max(0.04, cell_scale * 0.45),
                    (0.70, 0.14, 0.92, 0.55),
                    0.03,
                    max_points=self.blocking_obstacles_marker_max_points,
                    lifetime_s=self.blocking_obstacles_marker_lifetime_s,
                )
            )
        if blind_zone_conflict is not None:
            markers.markers.append(
                self._make_obstacle_sphere_marker(
                    marker_stamp,
                    130,
                    "blind_zone_focus",
                    blind_zone_conflict["world_x"],
                    blind_zone_conflict["world_y"],
                    max(0.16, self.blocking_obstacles_marker_scale_m * 1.9),
                    (1.0, 0.10, 1.0, 0.95),
                    0.18,
                    lifetime_s=self.blocking_obstacles_marker_lifetime_s,
                )
            )

        self.pub_blocking_obstacles.publish(markers)

    def _publish_debug_text(self, text, stamp=None, force=False):
        if rospy.is_shutdown():
            return
        stamp_sec = rospy.get_time() if stamp is None else float(stamp.to_sec())
        if (
            (not force)
            and text == self._last_debug_text
            and (stamp_sec - self._last_debug_text_time) < self.debug_text_period_s
        ):
            return
        self._last_debug_text = text
        self._last_debug_text_time = stamp_sec
        if self.pub_debug_text is not None:
            try:
                self.pub_debug_text.publish(String(data=text))
            except rospy.ROSException:
                pass
        if self.debug_screen_logging and (
            force or (stamp_sec - self._last_debug_screen_time) >= self.debug_screen_log_period_s
        ):
            self._last_debug_screen_time = stamp_sec
            rospy.loginfo("constrained_local_replanner debug | %s", text)

    def odom_callback(self, msg):
        stamp_sec = msg.header.stamp.to_sec()
        self.odom_stamp_sec = stamp_sec if stamp_sec > 0.0 else rospy.Time.now().to_sec()
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.odom_x = float(p.x)
        self.odom_y = float(p.y)
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        max_abs = self.slope_compensation_max_abs_rad
        if max_abs > 0.0:
            roll = max(-max_abs, min(max_abs, roll))
            pitch = max(-max_abs, min(max_abs, pitch))
        self.odom_roll = float(roll)
        self.odom_pitch = float(pitch)
        self.odom_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.have_odom = True
        self._record_travel_history_point(self.odom_x, self.odom_y)

    def global_path_callback(self, msg):
        pts = self._path_points(msg)
        sig = self._path_signature_from_points(pts)
        if sig != self.global_path_signature:
            self.global_nominal_progress_idx = 0
            self.last_avoidance_validation_sec = 0.0
            self.global_path_signature = sig
        self.global_path = msg

    def drivable_grid_callback(self, msg):
        next_sig = self._grid_signature(msg)
        if next_sig != self.drivable_grid_signature:
            self._inflated_blocked_cache.clear()
        self.drivable_grid_signature = next_sig
        self.drivable_grid = msg

    def risk_grid_callback(self, msg):
        next_sig = self._grid_signature(msg)
        if next_sig != self.risk_grid_signature:
            self._inflated_blocked_cache.clear()
        self.risk_grid_signature = next_sig
        self.risk_grid = msg

    @staticmethod
    def _grid_data_crc(data):
        try:
            if hasattr(data, "tobytes"):
                raw = data.tobytes()
            else:
                raw = array("b", data).tobytes()
            return zlib.crc32(raw) & 0xFFFFFFFF
        except Exception:
            crc = 0
            for value in data:
                crc = zlib.crc32(bytes((int(value) & 0xFF,)), crc)
            return crc & 0xFFFFFFFF

    def _grid_signature(self, msg):
        if msg is None:
            return None
        info = msg.info
        origin = info.origin.position
        return (
            int(info.width),
            int(info.height),
            round(float(info.resolution), 6),
            round(float(origin.x), 4),
            round(float(origin.y), 4),
            self._grid_data_crc(msg.data),
        )

    def _local_to_map(self, x, y):
        c = math.cos(self.odom_yaw)
        s = math.sin(self.odom_yaw)
        mx = self.odom_x + c * x - s * y
        my = self.odom_y + s * x + c * y
        return mx, my

    def _robot_front_anchor_xy(self):
        if not self.have_odom:
            return None
        return self._local_to_map(self.path_start_front_offset_m, 0.0)

    def _resolve_path_start_xy(self, start_xy):
        if not self.have_odom:
            return start_xy
        front_xy = self._robot_front_anchor_xy()
        if front_xy is None:
            return start_xy
        if start_xy is None:
            return front_xy
        sx = float(start_xy[0])
        sy = float(start_xy[1])
        center_snap_tol_m = max(0.05, 0.5 * self.path_start_front_offset_m)
        if math.hypot(sx - self.odom_x, sy - self.odom_y) <= center_snap_tol_m:
            return front_xy
        return (sx, sy)

    def _world_to_local(self, wx, wy):
        dx = float(wx) - self.odom_x
        dy = float(wy) - self.odom_y
        c = math.cos(self.odom_yaw)
        s = math.sin(self.odom_yaw)
        return c * dx + s * dy, (-s) * dx + c * dy

    @staticmethod
    def _normalized_frame_id(frame_id):
        return str(frame_id or "").strip().lstrip("/")

    def _known_map_frame_ids(self):
        frame_ids = {"map"}
        if self.drivable_grid is not None:
            frame_id = self._normalized_frame_id(self.drivable_grid.header.frame_id)
            if frame_id:
                frame_ids.add(frame_id)
        if self.global_path is not None:
            frame_id = self._normalized_frame_id(self.global_path.header.frame_id)
            if frame_id:
                frame_ids.add(frame_id)
        return frame_ids

    def _xy_message_point_to_map(self, msg, x, y):
        frame_id = self._normalized_frame_id(getattr(msg.header, "frame_id", ""))
        if frame_id and frame_id in self._known_map_frame_ids():
            return float(x), float(y)
        return self._local_to_map(x, y)

    def _pointcloud_cluster_cell(self, x, y):
        res = max(1e-3, self.pointcloud_cluster_resolution_m)
        return (int(math.floor(x / res)), int(math.floor(y / res)))

    def _leveled_z(self, x, y, z):
        if (not self.enable_slope_compensation) or (not self.have_odom):
            return z

        cr = math.cos(self.odom_roll)
        sr = math.sin(self.odom_roll)
        cp = math.cos(self.odom_pitch)
        sp = math.sin(self.odom_pitch)

        y1 = cr * y - sr * z
        z1 = sr * y + cr * z
        return (-sp * x) + (cp * z1)

    def _ground_relative_height(self, x, y, z):
        return self._leveled_z(x, y, z) + self.lidar_height_m

    def _dedupe_world_points(self, points_map):
        if not points_map:
            return []
        deduped = {}
        for wx, wy in points_map:
            deduped[self._pointcloud_cluster_cell(wx, wy)] = (wx, wy)
        return list(deduped.values())

    @staticmethod
    def _sample_axis_values(half_extent, spacing):
        if half_extent <= max(1e-6, 0.5 * spacing):
            return [0.0]
        values = {-half_extent, 0.0, half_extent}
        pos = -half_extent + spacing
        while pos < half_extent:
            values.add(round(pos, 4))
            pos += spacing
        return sorted(values)

    def _sample_oriented_box_points(self, center_x, center_y, half_x, half_y, yaw, spacing):
        c = math.cos(yaw)
        s = math.sin(yaw)
        sampled = []
        xs = self._sample_axis_values(max(0.0, half_x), spacing)
        ys = self._sample_axis_values(max(0.0, half_y), spacing)
        for lx in xs:
            for ly in ys:
                sampled.append((
                    center_x + c * lx - s * ly,
                    center_y + s * lx + c * ly,
                ))
        return sampled

    @staticmethod
    def _tracked_object_label(obj):
        return str(getattr(obj, "label", "") or "").strip().lower()

    @staticmethod
    def _yaw_from_quaternion(quat):
        return math.atan2(
            2.0 * (float(quat.w) * float(quat.z) + float(quat.x) * float(quat.y)),
            1.0 - 2.0 * (float(quat.y) * float(quat.y) + float(quat.z) * float(quat.z)),
        )

    def _build_tracked_object_virtual_points(self, obj):
        center_x = float(obj.pose.position.x)
        center_y = float(obj.pose.position.y)
        if (not math.isfinite(center_x)) or (not math.isfinite(center_y)):
            return []
        dx = center_x - self.odom_x
        dy = center_y - self.odom_y
        max_range_sq = self.tracked_object_virtual_max_range_m * self.tracked_object_virtual_max_range_m
        if (dx * dx + dy * dy) > max_range_sq:
            return []

        size_x = float(obj.size.x)
        size_y = float(obj.size.y)
        vx = float(obj.twist.linear.x)
        vy = float(obj.twist.linear.y)
        if (not math.isfinite(size_x)) or size_x <= 0.0:
            size_x = 0.25
        if (not math.isfinite(size_y)) or size_y <= 0.0:
            size_y = 0.25
        if not math.isfinite(vx):
            vx = 0.0
        if not math.isfinite(vy):
            vy = 0.0
        half_x = 0.5 * size_x + self.tracked_object_virtual_margin_m
        half_y = 0.5 * size_y + self.tracked_object_virtual_margin_m
        speed = math.hypot(vx, vy)
        yaw = (
            math.atan2(vy, vx)
            if speed > self.tracked_object_prediction_min_speed_mps
            else self._yaw_from_quaternion(obj.pose.orientation)
        )
        spacing = max(0.15, self.pointcloud_cluster_resolution_m)
        horizon_s = (
            max(0.0, self.tracked_object_prediction_horizon_s)
            if speed > self.tracked_object_prediction_min_speed_mps
            else 0.0
        )
        step_s = max(1e-3, self.tracked_object_prediction_step_s)
        num_steps = max(1, int(math.ceil(horizon_s / step_s)) + 1)
        points = []
        for step_idx in range(num_steps):
            dt = min(horizon_s, float(step_idx) * step_s)
            px = center_x + vx * dt
            py = center_y + vy * dt
            pdx = px - self.odom_x
            pdy = py - self.odom_y
            if (pdx * pdx + pdy * pdy) > max_range_sq:
                continue
            points.extend(
                self._sample_oriented_box_points(px, py, half_x, half_y, yaw, spacing)
            )
        return self._dedupe_world_points(points)

    def _prune_tracked_object_memory(self, now_sec):
        if (
            (not self.near_field_object_memory_enabled)
            or self.near_field_object_memory_ttl_s <= 0.0
            or self.near_field_object_memory_max_points <= 0
        ):
            self.tracked_object_memory_points = []
            self.tracked_object_memory_count = 0
            return []

        max_range_sq = self.near_field_object_memory_max_range_m * self.near_field_object_memory_max_range_m
        kept = []
        for wx, wy, seen_sec in self.tracked_object_memory_points:
            if (now_sec - seen_sec) > self.near_field_object_memory_ttl_s:
                continue
            dx = wx - self.odom_x
            dy = wy - self.odom_y
            if (dx * dx + dy * dy) > max_range_sq:
                continue
            kept.append((wx, wy, seen_sec))

        kept.sort(key=lambda item: item[2], reverse=True)
        if len(kept) > self.near_field_object_memory_max_points:
            kept = kept[: self.near_field_object_memory_max_points]

        self.tracked_object_memory_points = kept
        self.tracked_object_memory_count = len(kept)
        return [(wx, wy) for wx, wy, _ in kept]

    def _update_tracked_object_memory(self, candidates_map, now_sec):
        remembered_points = self._prune_tracked_object_memory(now_sec)
        if (not self.near_field_object_memory_enabled) or (not candidates_map):
            return remembered_points

        merge_radius_sq = (
            self.near_field_object_memory_merge_radius_m
            * self.near_field_object_memory_merge_radius_m
        )
        memory = list(self.tracked_object_memory_points)
        for wx, wy in candidates_map:
            best_idx = None
            best_d2 = merge_radius_sq
            for idx, (mx, my, _) in enumerate(memory):
                d2 = (wx - mx) * (wx - mx) + (wy - my) * (wy - my)
                if d2 <= best_d2:
                    best_d2 = d2
                    best_idx = idx
            if best_idx is None:
                memory.append((wx, wy, now_sec))
            else:
                memory[best_idx] = (wx, wy, now_sec)

        self.tracked_object_memory_points = memory
        return self._prune_tracked_object_memory(now_sec)

    def _merge_point_sets(self, primary_points, extra_points, merge_radius_m):
        if not extra_points:
            return list(primary_points)
        if not primary_points:
            return list(extra_points)

        merge_radius_sq = merge_radius_m * merge_radius_m
        merged = list(primary_points)
        for wx, wy in extra_points:
            duplicate = False
            for cx, cy in merged:
                d2 = (wx - cx) * (wx - cx) + (wy - cy) * (wy - cy)
                if d2 <= merge_radius_sq:
                    duplicate = True
                    break
            if not duplicate:
                merged.append((wx, wy))
        return merged

    def tracked_objects_callback(self, msg):
        if not self.have_odom:
            return

        stamp_sec = msg.header.stamp.to_sec()
        if stamp_sec <= 0.0:
            stamp_sec = rospy.Time.now().to_sec()

        if not self.tracked_object_virtual_obstacles_enabled:
            self.tracked_object_count = 0
            self.current_tracked_object_points_map = []
            self.tracked_object_points_map = []
            self.tracked_object_memory_points = []
            self.tracked_object_memory_count = 0
            self._publish_recognized_obstacle_markers(msg.header.stamp)
            return

        current_points = []
        memory_candidates = []
        near_field_range_sq = (
            self.near_field_object_memory_max_range_m
            * self.near_field_object_memory_max_range_m
        )
        tracked_count = 0
        for obj in msg.objects:
            object_points = self._build_tracked_object_virtual_points(obj)
            if not object_points:
                continue
            tracked_count += 1
            current_points.extend(object_points)
            dx = float(obj.pose.position.x) - self.odom_x
            dy = float(obj.pose.position.y) - self.odom_y
            if (dx * dx + dy * dy) <= near_field_range_sq:
                memory_candidates.extend(object_points)

        current_points = self._dedupe_world_points(current_points)
        remembered_points = self._update_tracked_object_memory(memory_candidates, stamp_sec)
        self.current_tracked_object_points_map = current_points
        self.tracked_object_points_map = self._merge_point_sets(
            current_points,
            remembered_points,
            self.near_field_object_memory_merge_radius_m,
        )
        self.tracked_object_count = tracked_count
        self._publish_recognized_obstacle_markers(msg.header.stamp)

    def _prune_obstacle_memory(self, now_sec):
        if (
            (not self.static_obstacle_memory_enabled)
            or self.static_obstacle_memory_ttl_s <= 0.0
            or self.static_obstacle_memory_max_points <= 0
        ):
            self.obstacle_memory_points = []
            self.obstacle_memory_count = 0
            self.obstacle_memory_locked_count = 0
            return []

        max_range_sq = self.static_obstacle_memory_max_range_m * self.static_obstacle_memory_max_range_m
        locked_keep_range_sq = (
            self.static_obstacle_memory_locked_keep_range_m
            * self.static_obstacle_memory_locked_keep_range_m
        )
        blind_zone_radius_sq = (
            self.static_obstacle_memory_blind_zone_radius_m
            * self.static_obstacle_memory_blind_zone_radius_m
        )
        kept = []
        for entry in self.obstacle_memory_points:
            wx, wy = self._memory_entry_position(entry)
            seen_sec = self._memory_entry_last_seen(entry)
            hits = self._memory_entry_hits(entry)
            locked = self._memory_entry_locked(entry)
            if self._is_known_map_static_obstacle(wx, wy):
                continue
            effective_ttl_s = self.static_obstacle_memory_ttl_s
            dx = wx - self.odom_x
            dy = wy - self.odom_y
            if (
                blind_zone_radius_sq > 0.0
                and (dx * dx + dy * dy) <= blind_zone_radius_sq
            ):
                effective_ttl_s = max(
                    effective_ttl_s,
                    self.static_obstacle_memory_blind_zone_hold_ttl_s,
                )
            if locked:
                effective_ttl_s = max(
                    effective_ttl_s, self.static_obstacle_memory_lock_ttl_s
                )
            if (now_sec - seen_sec) > effective_ttl_s:
                continue
            keep_range_sq = locked_keep_range_sq if locked else max_range_sq
            if (dx * dx + dy * dy) > keep_range_sq:
                continue
            normalized = dict(entry) if isinstance(entry, dict) else {}
            normalized["x"] = wx
            normalized["y"] = wy
            normalized["last_seen"] = seen_sec
            normalized["hits"] = hits
            normalized["locked"] = locked
            if "id" not in normalized:
                normalized["id"] = int(self.next_obstacle_memory_id)
                self.next_obstacle_memory_id += 1
            normalized["lock_time"] = float(normalized.get("lock_time", 0.0))
            kept.append(normalized)

        kept.sort(
            key=lambda item: (
                1 if bool(item.get("locked", False)) else 0,
                int(item.get("hits", 1)),
                float(item.get("last_seen", 0.0)),
            ),
            reverse=True,
        )
        if len(kept) > self.static_obstacle_memory_max_points:
            kept = kept[: self.static_obstacle_memory_max_points]

        self.obstacle_memory_points = kept
        self.obstacle_memory_count = len(kept)
        self.obstacle_memory_locked_count = sum(
            1 for item in kept if bool(item.get("locked", False))
        )
        return self._memory_points_from_entries(kept, confirmed_only=True)

    def _update_obstacle_memory(self, candidates_map, now_sec):
        remembered_points = self._prune_obstacle_memory(now_sec)
        if (not self.static_obstacle_memory_enabled) or (not candidates_map):
            return remembered_points

        merge_radius_sq = (
            self.static_obstacle_memory_merge_radius_m * self.static_obstacle_memory_merge_radius_m
        )
        memory = list(self.obstacle_memory_points)
        for wx, wy in candidates_map:
            best_idx = None
            best_d2 = merge_radius_sq
            for idx, entry in enumerate(memory):
                mx, my = self._memory_entry_position(entry)
                d2 = (wx - mx) * (wx - mx) + (wy - my) * (wy - my)
                if d2 <= best_d2:
                    best_d2 = d2
                    best_idx = idx
            if best_idx is None:
                memory.append(
                    {
                        "id": int(self.next_obstacle_memory_id),
                        "x": float(wx),
                        "y": float(wy),
                        "last_seen": float(now_sec),
                        "hits": 1,
                        "locked": False,
                        "lock_time": 0.0,
                    }
                )
                self.next_obstacle_memory_id += 1
            else:
                entry = dict(memory[best_idx])
                prev_hits = max(1, self._memory_entry_hits(entry))
                hits = min(
                    prev_hits + 1,
                    self.static_obstacle_memory_persistence_frames + 16,
                )
                locked = bool(entry.get("locked", False))
                entry["x"] = float(wx)
                entry["y"] = float(wy)
                entry["last_seen"] = float(now_sec)
                entry["hits"] = int(hits)
                if (not locked) and (
                    hits >= self.static_obstacle_memory_persistence_frames
                ):
                    locked = True
                    entry["lock_time"] = float(now_sec)
                    rospy.loginfo(
                        "constrained_local_replanner: locked static obstacle #%d at (%.2f, %.2f) after %d hits",
                        int(entry.get("id", -1)),
                        float(entry["x"]),
                        float(entry["y"]),
                        hits,
                    )
                entry["locked"] = locked
                memory[best_idx] = entry

        self.obstacle_memory_points = memory
        return self._prune_obstacle_memory(now_sec)

    def _merge_obstacle_memory_points(self, current_points_map, remembered_points_map):
        return self._merge_point_sets(
            current_points_map,
            remembered_points_map,
            self.static_obstacle_memory_merge_radius_m,
        )

    def _effective_raw_near_obstacle_points_map(self, now_sec=None):
        current_points = list(self.raw_near_obstacle_points_map)
        if current_points:
            return current_points
        if self.raw_near_obstacle_hold_s <= 0.0:
            return []
        if now_sec is None:
            now_sec = rospy.Time.now().to_sec()
        age_s = now_sec - self.last_nonempty_raw_near_obstacle_points_stamp_sec
        if (
            self.last_nonempty_raw_near_obstacle_points_stamp_sec > 0.0
            and age_s <= self.raw_near_obstacle_hold_s
        ):
            self._debug_avoidance_log(
                "constrained_local_replanner: reusing recent raw-near obstacle hits | age={:.2f}s pts={}".format(
                    max(0.0, age_s),
                    len(self.last_nonempty_raw_near_obstacle_points_map),
                )
            )
            return list(self.last_nonempty_raw_near_obstacle_points_map)
        return []

    def _effective_current_obstacle_points_map(self, now_sec=None):
        if now_sec is None:
            now_sec = rospy.Time.now().to_sec()
        current_points = list(self.current_obstacle_points_map)
        if (
            (not current_points)
            and self.pointcloud_visibility_hold_s > 0.0
        ):
            age_s = now_sec - self.last_nonempty_current_obstacle_points_stamp_sec
            if (
                self.last_nonempty_current_obstacle_points_stamp_sec > 0.0
                and age_s <= self.pointcloud_visibility_hold_s
            ):
                self._debug_avoidance_log(
                    "constrained_local_replanner: reusing recent current obstacle points during lidar dropout | age={:.2f}s pts={}".format(
                        max(0.0, age_s),
                        len(self.last_nonempty_current_obstacle_points_map),
                    )
                )
                current_points = list(self.last_nonempty_current_obstacle_points_map)

        raw_near_points = self._effective_raw_near_obstacle_points_map(now_sec=now_sec)
        if not raw_near_points:
            return current_points
        return self._merge_point_sets(
            current_points,
            raw_near_points,
            self.pointcloud_cluster_resolution_m,
        )

    def _refresh_merged_obstacle_points_map(self, now_sec=None, remembered_points=None):
        if now_sec is None:
            now_sec = rospy.Time.now().to_sec()
        if remembered_points is None:
            remembered_points = self._prune_obstacle_memory(now_sec)
        self.obstacle_points_map = self._merge_obstacle_memory_points(
            self._effective_current_obstacle_points_map(now_sec=now_sec),
            remembered_points,
        )

    def _combined_dynamic_obstacle_points(self, include_tracked=True):
        current_points = self._effective_current_obstacle_points_map()
        tracked_points = self.tracked_object_points_map if include_tracked else []
        if not tracked_points:
            return current_points
        merge_radius_m = max(
            self.pointcloud_cluster_resolution_m,
            self.near_field_object_memory_merge_radius_m,
        )
        return self._merge_point_sets(
            current_points,
            tracked_points,
            merge_radius_m,
        )

    def _avoidance_trigger_points_map(self, include_tracked=True):
        points = self._combined_dynamic_obstacle_points(include_tracked=include_tracked)
        if (
            self.use_map_filtered_path_obstacle_trigger
            and self.known_map_filtered_points_map
        ):
            points = self._merge_point_sets(
                points,
                self.known_map_filtered_points_map,
                self.pointcloud_cluster_resolution_m,
            )
        return points

    def _in_global_overlay_blind_zone(self, wx, wy):
        if self.global_pointcloud_overlay_blind_zone_radius_m <= 0.0:
            return False
        dx = wx - self.odom_x
        dy = wy - self.odom_y
        radius_sq = (
            self.global_pointcloud_overlay_blind_zone_radius_m
            * self.global_pointcloud_overlay_blind_zone_radius_m
        )
        return (dx * dx + dy * dy) <= radius_sq

    def _select_global_overlay_candidate_points(self, current_points_map):
        if (
            (not self.enable_global_pointcloud_overlay)
            or (not current_points_map)
        ):
            return []

        pts = self._global_path_points()
        if len(pts) < 2:
            return []
        i0 = self._nearest_idx(pts, self.odom_x, self.odom_y)
        ig = self._accum_distance(pts, i0, self.global_pointcloud_overlay_lookahead_m)
        path_slice = pts[i0 : ig + 1]
        if len(path_slice) < 2:
            return []

        max_range_sq = self.global_pointcloud_overlay_max_range_m * self.global_pointcloud_overlay_max_range_m
        corridor_half = self._pointcloud_corridor_half_width_m(
            self.global_pointcloud_overlay_corridor_margin_m
        )
        corridor_half_sq = corridor_half * corridor_half
        selected = []
        for wx, wy in current_points_map:
            dx = wx - self.odom_x
            dy = wy - self.odom_y
            if (dx * dx + dy * dy) > max_range_sq:
                continue
            for idx in range(len(path_slice) - 1):
                x0, y0 = path_slice[idx]
                x1, y1 = path_slice[idx + 1]
                if self._point_to_segment_distance_sq(wx, wy, x0, y0, x1, y1) <= corridor_half_sq:
                    selected.append((wx, wy))
                    break
        return selected

    def _prune_global_obstacle_overlay_memory(self, now_sec):
        if (
            (not self.enable_global_pointcloud_overlay)
            or self.global_pointcloud_overlay_ttl_s <= 0.0
            or self.global_pointcloud_overlay_max_points <= 0
        ):
            self.global_obstacle_overlay_memory = []
            self.global_obstacle_overlay_points_map = []
            self.global_obstacle_overlay_confirmed_count = 0
            return []

        max_range_sq = self.global_pointcloud_overlay_max_range_m * self.global_pointcloud_overlay_max_range_m
        kept = []
        for wx, wy, seen_sec, hits in self.global_obstacle_overlay_memory:
            effective_ttl_s = self.global_pointcloud_overlay_ttl_s
            if self._in_global_overlay_blind_zone(wx, wy):
                effective_ttl_s = max(
                    effective_ttl_s,
                    self.global_pointcloud_overlay_blind_zone_hold_ttl_s,
                )
            if (now_sec - seen_sec) > effective_ttl_s:
                continue
            dx = wx - self.odom_x
            dy = wy - self.odom_y
            if (dx * dx + dy * dy) > max_range_sq:
                continue
            kept.append((wx, wy, seen_sec, hits))

        kept.sort(key=lambda item: (item[3], item[2]), reverse=True)
        if len(kept) > self.global_pointcloud_overlay_max_points:
            kept = kept[: self.global_pointcloud_overlay_max_points]

        self.global_obstacle_overlay_memory = kept
        confirmed = [
            (wx, wy)
            for wx, wy, _seen_sec, hits in kept
            if hits >= self.global_pointcloud_overlay_persistence_frames
        ]
        self.global_obstacle_overlay_points_map = confirmed
        self.global_obstacle_overlay_confirmed_count = len(confirmed)
        return confirmed

    def _update_global_obstacle_overlay_memory(self, candidates_map, now_sec):
        confirmed = self._prune_global_obstacle_overlay_memory(now_sec)
        if (not self.enable_global_pointcloud_overlay) or (not candidates_map):
            return confirmed

        merge_radius_sq = (
            self.global_pointcloud_overlay_merge_radius_m
            * self.global_pointcloud_overlay_merge_radius_m
        )
        memory = list(self.global_obstacle_overlay_memory)
        for wx, wy in candidates_map:
            best_idx = None
            best_d2 = merge_radius_sq
            for idx, (mx, my, _seen_sec, _hits) in enumerate(memory):
                d2 = (wx - mx) * (wx - mx) + (wy - my) * (wy - my)
                if d2 <= best_d2:
                    best_d2 = d2
                    best_idx = idx
            if best_idx is None:
                memory.append((wx, wy, now_sec, 1))
            else:
                _mx, _my, _seen_sec, hits = memory[best_idx]
                memory[best_idx] = (
                    wx,
                    wy,
                    now_sec,
                    min(
                        hits + 1,
                        self.global_pointcloud_overlay_persistence_frames + 8,
                    ),
                )

        self.global_obstacle_overlay_memory = memory
        return self._prune_global_obstacle_overlay_memory(now_sec)

    def _publish_global_obstacle_overlay(self, stamp):
        if (
            (not self.enable_global_pointcloud_overlay)
            or self.pub_global_obstacle_overlay is None
            or self.drivable_grid is None
        ):
            return

        base = self.drivable_grid
        out = OccupancyGrid()
        if stamp is not None and stamp.to_sec() > 0.0:
            out.header.stamp = stamp
        else:
            out.header.stamp = rospy.Time.now()
        out.header.frame_id = base.header.frame_id if base.header.frame_id else "map"
        out.info = base.info
        w = int(base.info.width)
        h = int(base.info.height)
        data = [0] * (w * h)
        for wx, wy in self.global_obstacle_overlay_points_map:
            gx, gy = self._world_to_grid(base, wx, wy)
            if not self._in_bounds(base, gx, gy):
                continue
            data[gy * w + gx] = 100
        out.data = data
        self.pub_global_obstacle_overlay.publish(out)

    def cloud_callback(self, msg):
        if not self.have_odom:
            return
        raw_pts = []
        cluster_counts = {}
        cluster_sums = {}
        cluster_max_z = {}
        rr = self.obstacle_max_range_m * self.obstacle_max_range_m
        i = 0
        try:
            for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                i += 1
                if self.obstacle_downsample > 1 and (i % self.obstacle_downsample != 0):
                    continue
                x = float(p[0])
                y = float(p[1])
                z = float(p[2])
                if self.enable_ground_band_rejection:
                    ground_h = self._ground_relative_height(x, y, z)
                    if self.ground_reject_min_m <= ground_h <= self.ground_reject_max_m:
                        continue
                z_eval = self._leveled_z(x, y, z)
                if z_eval < self.obstacle_min_z or z_eval > self.obstacle_max_z:
                    continue
                if x * x + y * y > rr:
                    continue
                if abs(x) <= self.self_filter_radius_x and abs(y) <= self.self_filter_radius_y:
                    continue
                raw_pts.append((x, y))
                cell = self._pointcloud_cluster_cell(x, y)
                cluster_counts[cell] = cluster_counts.get(cell, 0) + 1
                sx, sy = cluster_sums.get(cell, (0.0, 0.0))
                cluster_sums[cell] = (sx + x, sy + y)
                cluster_max_z[cell] = max(cluster_max_z.get(cell, z), z)

            self.obstacle_raw_point_count = len(raw_pts)
            filtered_clusters = []
            for (cx, cy), count in cluster_counts.items():
                support = 0
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        support += cluster_counts.get((cx + dx, cy + dy), 0)
                if support < self.pointcloud_min_cluster_points:
                    continue
                sx, sy = cluster_sums[(cx, cy)]
                fx = sx / float(count)
                fy = sy / float(count)
                filtered_clusters.append(
                    {
                        "x": fx,
                        "y": fy,
                        "support": support,
                        "z_max": cluster_max_z.get((cx, cy), self.obstacle_max_z),
                    }
                )

            self.obstacle_cluster_count = len(filtered_clusters)
            current_points_map = []
            known_map_filtered_points_map = []
            map_filtered_count = 0
            for item in filtered_clusters:
                wx, wy = self._local_to_map(item["x"], item["y"])
                if self._is_known_map_static_obstacle(wx, wy):
                    map_filtered_count += 1
                    known_map_filtered_points_map.append((wx, wy))
                    continue
                current_points_map.append((wx, wy))
            self.known_map_filtered_count = map_filtered_count
            self.known_map_filtered_points_map = list(known_map_filtered_points_map)
            self.current_obstacle_points_map = list(current_points_map)

            stamp_sec = msg.header.stamp.to_sec()
            if stamp_sec <= 0.0:
                stamp_sec = rospy.Time.now().to_sec()
            self.current_obstacle_points_stamp_sec = stamp_sec
            if current_points_map:
                self.last_nonempty_current_obstacle_points_map = list(current_points_map)
                self.last_nonempty_current_obstacle_points_stamp_sec = stamp_sec

            memory_candidates_map = []
            max_range_sq = self.static_obstacle_memory_max_range_m * self.static_obstacle_memory_max_range_m
            for item in filtered_clusters:
                wx, wy = self._local_to_map(item["x"], item["y"])
                if self._is_known_map_static_obstacle(wx, wy):
                    continue
                if item["support"] > self.static_obstacle_memory_max_support:
                    continue
                if item["z_max"] > self.static_obstacle_memory_max_z_m:
                    continue
                if (item["x"] * item["x"] + item["y"] * item["y"]) > max_range_sq:
                    continue
                memory_candidates_map.append((wx, wy))

            remembered_points = self._update_obstacle_memory(memory_candidates_map, stamp_sec)
            self._refresh_merged_obstacle_points_map(
                now_sec=stamp_sec,
                remembered_points=remembered_points,
            )
            global_overlay_candidates = self._select_global_overlay_candidate_points(
                current_points_map
            )
            self.global_obstacle_overlay_candidate_count = len(global_overlay_candidates)
            self._update_global_obstacle_overlay_memory(global_overlay_candidates, stamp_sec)
            self._publish_global_obstacle_overlay(msg.header.stamp)
            self._publish_recognized_obstacle_markers(msg.header.stamp)
        except Exception as e:
            rospy.logwarn_throttle(1.0, "constrained_local_replanner cloud error: %s", str(e))

    def raw_near_obstacle_hits_callback(self, msg):
        if not self.have_odom:
            return
        try:
            cluster_counts = {}
            cluster_sums = {}
            for p in point_cloud2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True
            ):
                x = float(p[0])
                y = float(p[1])
                if (not math.isfinite(x)) or (not math.isfinite(y)):
                    continue
                wx, wy = self._xy_message_point_to_map(msg, x, y)
                if self._is_known_map_static_obstacle(wx, wy):
                    continue
                cell = self._pointcloud_cluster_cell(wx, wy)
                cluster_counts[cell] = cluster_counts.get(cell, 0) + 1
                sx, sy = cluster_sums.get(cell, (0.0, 0.0))
                cluster_sums[cell] = (sx + wx, sy + wy)

            points_map = []
            for cell, count in cluster_counts.items():
                sx, sy = cluster_sums[cell]
                points_map.append((sx / float(count), sy / float(count)))

            stamp_sec = msg.header.stamp.to_sec()
            if stamp_sec <= 0.0:
                stamp_sec = rospy.Time.now().to_sec()
            self.raw_near_obstacle_points_map = list(points_map)
            self.raw_near_obstacle_points_stamp_sec = stamp_sec
            if points_map:
                self.last_nonempty_raw_near_obstacle_points_map = list(points_map)
                self.last_nonempty_raw_near_obstacle_points_stamp_sec = stamp_sec
            self._refresh_merged_obstacle_points_map(now_sec=stamp_sec)
        except Exception as e:
            rospy.logwarn_throttle(
                1.0,
                "constrained_local_replanner raw-near obstacle hit error: %s",
                str(e),
            )

    def _pointcloud_corridor_half_width_m(self, margin_m):
        return max(
            0.05,
            0.5 * self.robot_width_m + self.footprint_padding_m + max(0.0, float(margin_m)),
        )

    def direct_goal_callback(self, msg):
        now_sec = rospy.Time.now().to_sec()
        if self.direct_goal is not None:
            prev_x = float(self.direct_goal.pose.position.x)
            prev_y = float(self.direct_goal.pose.position.y)
            new_x = float(msg.pose.position.x)
            new_y = float(msg.pose.position.y)
            prev_yaw = self._pose_yaw(self.direct_goal.pose)
            new_yaw = self._pose_yaw(msg.pose)
            if (
                math.hypot(new_x - prev_x, new_y - prev_y) <= self.direct_goal_refresh_distance_m
                and abs(self._angle_diff(new_yaw, prev_yaw))
                <= math.radians(self.direct_goal_refresh_yaw_deg)
            ):
                self.direct_goal = msg
                self.direct_goal_stamp_sec = now_sec
                return
        self.direct_goal = msg
        self.direct_goal_stamp_sec = now_sec
        self.cached_direct_goal_cell = None
        self.frozen_direct_goal_cell = None
        self.frozen_direct_grid_path = None
        self.frozen_direct_start_xy = None
        self.frozen_direct_goal_xy = None
        self.last_published_goal_cell = None
        self.last_published_end_cell = None
        self.last_path_publish_sec = 0.0
        self.avoidance_active = False
        self.avoidance_clear_count = 0
        self.last_avoidance_publish_sec = 0.0
        self.last_avoidance_grid_path = None
        self.last_avoidance_world_path = None
        self.last_avoidance_solution_sec = 0.0
        self.last_avoidance_validation_sec = 0.0
        self.last_avoidance_active_sec = 0.0
        self._reset_pending_avoidance_trigger()
        self.local_blocked_since_sec = 0.0
        self.local_clear_since_sec = 0.0
        self.last_avoidance_trigger_reason = ""
        self.last_avoidance_direction = "none"
        self.global_nominal_progress_idx = 0
        self.obstacle_memory_points = []
        self.obstacle_memory_count = 0
        self.obstacle_memory_locked_count = 0
        self.known_map_filtered_count = 0
        self._clear_path_history()
        self._clear_travel_history()
        self._clear_avoidance_path("map", rospy.Time.now(), force=True)
        rospy.loginfo(
            "constrained_local_replanner: direct goal set (%.2f, %.2f) frame=%s",
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            msg.header.frame_id if msg.header.frame_id else "map",
        )

    @staticmethod
    def _path_points(path):
        return [(float(ps.pose.position.x), float(ps.pose.position.y)) for ps in path.poses]

    def _global_path_points(self):
        if self.global_path is None or not self.global_path.poses:
            return []
        pts = self._path_points(self.global_path)
        if len(pts) >= 2 or (not self.have_odom):
            return pts
        goal_x, goal_y = pts[0]
        if math.hypot(goal_x - self.odom_x, goal_y - self.odom_y) <= 1e-3:
            return pts
        rospy.loginfo_throttle(
            1.0,
            "constrained_local_replanner: expanding single-pose global path to odom->goal fallback (goal=%.2f, %.2f)",
            goal_x,
            goal_y,
        )
        return [(self.odom_x, self.odom_y), (goal_x, goal_y)]

    def _nominal_global_start_index(self, points):
        if not points:
            self.global_nominal_progress_idx = 0
            return 0
        max_idx = max(0, len(points) - 2)
        if max_idx <= 0:
            self.global_nominal_progress_idx = 0
            return 0

        front_xy = self._robot_front_anchor_xy()
        anchor_x, anchor_y = (
            front_xy if front_xy is not None else (self.odom_x, self.odom_y)
        )
        nearest_i = max(0, min(max_idx, self._nearest_idx(points, anchor_x, anchor_y)))
        progress_i = max(0, min(max_idx, int(self.global_nominal_progress_idx)))

        search_start = progress_i
        if self.nominal_start_backtrack_m > 1e-6 and nearest_i < progress_i:
            back_i = progress_i
            back_dist = 0.0
            while back_i > 0 and back_dist < self.nominal_start_backtrack_m:
                x0, y0 = points[back_i - 1]
                x1, y1 = points[back_i]
                back_dist += math.hypot(float(x1) - float(x0), float(y1) - float(y0))
                back_i -= 1
            search_start = max(back_i, nearest_i)

        search_end = search_start
        travelled = 0.0
        horizon_m = self.nominal_start_heading_search_ahead_m
        while search_end < max_idx and (horizon_m <= 1e-6 or travelled < horizon_m):
            x0, y0 = points[search_end]
            x1, y1 = points[search_end + 1]
            travelled += math.hypot(float(x1) - float(x0), float(y1) - float(y0))
            search_end += 1
        search_end = max(search_start, min(max_idx, search_end))

        yaw_c = math.cos(self.odom_yaw)
        yaw_s = math.sin(self.odom_yaw)
        best_i = None
        best_score = float("inf")
        best_dist = float("inf")
        best_dot = 1.0
        fallback_i = progress_i
        fallback_score = float("inf")
        fallback_dist = float("inf")
        fallback_dot = 1.0
        fallback_heading_i = progress_i
        fallback_heading_dot = -2.0
        fallback_heading_dist = float("inf")
        penalty_m = self.nominal_start_heading_penalty_m
        min_heading_dot = self.nominal_start_min_heading_dot

        for i in range(search_start, search_end + 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            sx = float(x1) - float(x0)
            sy = float(y1) - float(y0)
            seg_len2 = sx * sx + sy * sy
            if seg_len2 <= 1e-12:
                continue

            rx = float(anchor_x) - float(x0)
            ry = float(anchor_y) - float(y0)
            t = max(0.0, min(1.0, (rx * sx + ry * sy) / seg_len2))
            proj_x = float(x0) + t * sx
            proj_y = float(y0) + t * sy
            dist2 = (
                (float(anchor_x) - proj_x) * (float(anchor_x) - proj_x)
                + (float(anchor_y) - proj_y) * (float(anchor_y) - proj_y)
            )

            # The published nominal local path is anchored at the robot front,
            # so score the first direction the controller will actually track.
            hx = float(x1) - float(anchor_x)
            hy = float(y1) - float(anchor_y)
            h_len = math.hypot(hx, hy)
            if h_len <= 1e-6:
                h_len = math.sqrt(seg_len2)
                hx = sx
                hy = sy
            heading_dot = 1.0
            if h_len > 1e-6:
                heading_dot = max(-1.0, min(1.0, (hx / h_len) * yaw_c + (hy / h_len) * yaw_s))

            # Treat poor heading agreement as an equivalent distance penalty.
            # This keeps nearby crossing/overlap segments from becoming the
            # first local path segment when they point sideways or backwards.
            heading_penalty = 0.5 * penalty_m * max(0.0, 1.0 - heading_dot)
            score = dist2 + heading_penalty * heading_penalty
            dist = math.sqrt(max(0.0, dist2))
            if score < fallback_score:
                fallback_score = score
                fallback_i = i
                fallback_dist = dist
                fallback_dot = heading_dot
            if (
                heading_dot > fallback_heading_dot + 1e-6
                or (
                    abs(heading_dot - fallback_heading_dot) <= 1e-6
                    and dist < fallback_heading_dist
                )
            ):
                fallback_heading_i = i
                fallback_heading_dot = heading_dot
                fallback_heading_dist = dist
            if heading_dot < min_heading_dot:
                continue
            if score < best_score:
                best_score = score
                best_i = i
                best_dist = dist
                best_dot = heading_dot

        used_fallback = best_i is None
        if used_fallback:
            if fallback_heading_dot > fallback_dot:
                best_i = fallback_heading_i
                best_dist = fallback_heading_dist
                best_dot = fallback_heading_dot
            else:
                best_i = fallback_i
                best_dist = fallback_dist
                best_dot = fallback_dot
        start_i = max(0, min(max_idx, best_i))
        if start_i != nearest_i or start_i < progress_i or best_dot < min_heading_dot:
            rospy.loginfo_throttle(
                1.0,
                "constrained_local_replanner: heading-aware nominal start idx=%d nearest=%d progress=%d search=%d..%d dist=%.2fm dot=%.2f min_dot=%.2f fallback=%s",
                start_i,
                nearest_i,
                progress_i,
                search_start,
                search_end,
                best_dist,
                best_dot,
                min_heading_dot,
                "yes" if used_fallback else "no",
            )
        self.global_nominal_progress_idx = start_i
        return start_i

    @staticmethod
    def _nearest_idx(points, x, y):
        best_i = 0
        best_d2 = 1e18
        for i, (px, py) in enumerate(points):
            d2 = (px - x) * (px - x) + (py - y) * (py - y)
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        return best_i

    @staticmethod
    def _accum_distance(points, i0, target_dist):
        s = 0.0
        i = i0
        while i + 1 < len(points):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            s += math.hypot(x1 - x0, y1 - y0)
            if s >= target_dist:
                return i + 1
            i += 1
        return len(points) - 1

    def _world_to_grid(self, g, x, y):
        res = float(g.info.resolution)
        gx = int(math.floor((x - float(g.info.origin.position.x)) / res))
        gy = int(math.floor((y - float(g.info.origin.position.y)) / res))
        return gx, gy

    def _grid_to_world(self, g, gx, gy):
        res = float(g.info.resolution)
        x = float(g.info.origin.position.x) + (gx + 0.5) * res
        y = float(g.info.origin.position.y) + (gy + 0.5) * res
        return x, y

    @staticmethod
    def _in_bounds(g, gx, gy):
        return 0 <= gx < int(g.info.width) and 0 <= gy < int(g.info.height)

    @staticmethod
    def _in_bounds_blocked(blocked, gx, gy):
        h = len(blocked)
        w = len(blocked[0]) if h > 0 else 0
        return 0 <= gx < w and 0 <= gy < h

    def _grid_cell_is_drivable_free(self, g, gx, gy):
        if not self._in_bounds(g, gx, gy):
            return False
        idx = gy * int(g.info.width) + gx
        return int(g.data[idx]) == 0

    def _is_known_map_static_obstacle(self, wx, wy):
        if (not self.known_map_subtraction_enabled) or self.drivable_grid is None:
            return False
        dg = self.drivable_grid
        gx, gy = self._world_to_grid(dg, wx, wy)
        if not self._grid_cell_is_drivable_free(dg, gx, gy):
            return True

        radius_m = self.known_map_subtraction_radius_m
        if radius_m <= 0.0:
            return False

        res = max(1e-3, float(dg.info.resolution))
        cell_radius = int(math.ceil(radius_m / res))
        radius_sq = radius_m * radius_m
        for ny in range(gy - cell_radius, gy + cell_radius + 1):
            for nx in range(gx - cell_radius, gx + cell_radius + 1):
                if not self._in_bounds(dg, nx, ny):
                    continue
                if self._grid_cell_is_drivable_free(dg, nx, ny):
                    continue
                cx, cy = self._grid_to_world(dg, nx, ny)
                dx = cx - float(wx)
                dy = cy - float(wy)
                if (dx * dx + dy * dy) <= radius_sq:
                    return True
        return False

    def _is_blocked_cell(self, dg, rg, gx, gy):
        if not self._in_bounds(dg, gx, gy):
            return True
        i = gy * int(dg.info.width) + gx
        val = int(dg.data[i])
        if val != 0:
            return True
        if rg is not None and int(rg.info.width) == int(dg.info.width) and int(rg.info.height) == int(dg.info.height):
            rv = int(rg.data[i])
            if rv >= self.risk_threshold:
                return True
        return False

    def _inflate_blocked(self, dg, rg, radius_override_m=None):
        w = int(dg.info.width)
        h = int(dg.info.height)
        res = float(dg.info.resolution)
        inflate_m = max(
            0.05,
            self.path_blocking_radius_m
            if radius_override_m is None
            else float(radius_override_m),
        )
        risk_sig = None
        if rg is not None and int(rg.info.width) == w and int(rg.info.height) == h:
            risk_sig = self.risk_grid_signature
        cache_key = (
            self.drivable_grid_signature,
            risk_sig,
            round(float(inflate_m), 4),
            int(self.risk_threshold),
        )
        cached = self._inflated_blocked_cache.get(cache_key)
        if cached is not None:
            return cached

        inflate_cells = max(1, int(math.ceil(inflate_m / max(1e-3, res))))
        base = [[False for _ in range(w)] for _ in range(h)]
        for y in range(h):
            row = y * w
            for x in range(w):
                i = row + x
                dv = int(dg.data[i])
                blocked = (dv != 0)
                if (not blocked) and rg is not None and int(rg.info.width) == w and int(rg.info.height) == h:
                    blocked = int(rg.data[i]) >= self.risk_threshold
                base[y][x] = blocked

        out = [[False for _ in range(w)] for _ in range(h)]
        for y in range(h):
            for x in range(w):
                if not base[y][x]:
                    continue
                for dx in range(-inflate_cells, inflate_cells + 1):
                    for dy in range(-inflate_cells, inflate_cells + 1):
                        if math.hypot(float(dx) * res, float(dy) * res) > inflate_m:
                            continue
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            out[ny][nx] = True
        if len(self._inflated_blocked_cache) >= self._inflated_blocked_cache_max_entries:
            self._inflated_blocked_cache.clear()
        self._inflated_blocked_cache[cache_key] = out
        return out

    def _nearest_free_cell(self, blocked, cell):
        cx, cy = cell
        if self._in_bounds_blocked(blocked, cx, cy) and not blocked[cy][cx]:
            return (cx, cy)

        best = None
        best_d2 = float("inf")
        max_r = self.snap_search_radius_cells
        for r in range(1, max_r + 1):
            found_this_ring = False
            x0 = cx - r
            x1 = cx + r
            y0 = cy - r
            y1 = cy + r
            for gx in range(x0, x1 + 1):
                for gy in range(y0, y1 + 1):
                    if max(abs(gx - cx), abs(gy - cy)) != r:
                        continue
                    if not self._in_bounds_blocked(blocked, gx, gy):
                        continue
                    if blocked[gy][gx]:
                        continue
                    d2 = float((gx - cx) * (gx - cx) + (gy - cy) * (gy - cy))
                    if d2 < best_d2:
                        best_d2 = d2
                        best = (gx, gy)
                        found_this_ring = True
            if found_this_ring:
                return best
        return None

    def _resolve_snap_cell(
        self,
        dg,
        rg,
        blocked,
        raw_cell,
        *,
        allow_raw_cell=False,
    ):
        snapped = self._nearest_free_cell(blocked, raw_cell)
        if snapped is not None:
            return snapped

        relaxed_radius_m = min(
            float(self.path_blocking_radius_m),
            float(self.relaxed_snap_path_blocking_radius_m),
        )
        if relaxed_radius_m + 1e-6 < float(self.path_blocking_radius_m):
            relaxed_blocked = self._inflate_blocked(
                dg, rg, radius_override_m=relaxed_radius_m
            )
            snapped = self._nearest_free_cell(relaxed_blocked, raw_cell)
            if snapped is not None:
                rospy.loginfo_throttle(
                    1.0,
                    "constrained_local_replanner: relaxed snap cell %s -> %s (radius %.2f -> %.2f)",
                    str(tuple(raw_cell)),
                    str(tuple(snapped)),
                    float(self.path_blocking_radius_m),
                    float(relaxed_radius_m),
                )
                return snapped

        if allow_raw_cell:
            gx, gy = int(raw_cell[0]), int(raw_cell[1])
            if not self._is_blocked_cell(dg, rg, gx, gy):
                rospy.loginfo_throttle(
                    1.0,
                    "constrained_local_replanner: using raw start cell %s despite inflated snap blockage",
                    str((gx, gy)),
                )
                return (gx, gy)

        return None

    def _resolve_direct_goal_cell(self, dg, rg, blocked, raw_goal_cell):
        if self.cached_direct_goal_cell is not None:
            cgx, cgy = self.cached_direct_goal_cell
            if self._in_bounds_blocked(blocked, cgx, cgy) and not blocked[cgy][cgx]:
                return self.cached_direct_goal_cell
        self.cached_direct_goal_cell = self._resolve_snap_cell(
            dg,
            rg,
            blocked,
            raw_goal_cell,
            allow_raw_cell=False,
        )
        return self.cached_direct_goal_cell

    def _build_direct_goal_planning_blocked(
        self,
        dg,
        rg,
        strict_blocked,
        start_cell,
        goal_cell,
    ):
        if not strict_blocked:
            return strict_blocked, "strict"

        def _cell_is_blocked(mask, cell):
            gx, gy = int(cell[0]), int(cell[1])
            return (not self._in_bounds_blocked(mask, gx, gy)) or mask[gy][gx]

        relaxed_needed = _cell_is_blocked(strict_blocked, start_cell) or _cell_is_blocked(
            strict_blocked, goal_cell
        )
        relaxed_radius_m = min(
            float(self.path_blocking_radius_m),
            float(self.relaxed_snap_path_blocking_radius_m),
        )

        if relaxed_needed and relaxed_radius_m + 1e-6 < float(self.path_blocking_radius_m):
            relaxed_blocked = self._inflate_blocked(
                dg, rg, radius_override_m=relaxed_radius_m
            )
            if not _cell_is_blocked(relaxed_blocked, start_cell) and not _cell_is_blocked(
                relaxed_blocked, goal_cell
            ):
                rospy.loginfo_throttle(
                    1.0,
                    "constrained_local_replanner: direct-goal planning using relaxed inflated grid (radius %.2f -> %.2f start=%s goal=%s)",
                    float(self.path_blocking_radius_m),
                    float(relaxed_radius_m),
                    str(tuple(start_cell)),
                    str(tuple(goal_cell)),
                )
                return relaxed_blocked, "relaxed"

        planning_blocked = [row[:] for row in strict_blocked]
        cleared = []
        for cell in (start_cell, goal_cell):
            gx, gy = int(cell[0]), int(cell[1])
            if (
                self._in_bounds_blocked(planning_blocked, gx, gy)
                and planning_blocked[gy][gx]
                and (not self._is_blocked_cell(dg, rg, gx, gy))
            ):
                planning_blocked[gy][gx] = False
                cleared.append((gx, gy))
        if cleared:
            rospy.loginfo_throttle(
                1.0,
                "constrained_local_replanner: direct-goal planning keeping raw free cells on strict grid %s",
                str(cleared),
            )
            return planning_blocked, "strict_keep"
        return strict_blocked, "strict"

    @staticmethod
    def _heur(a, b):
        return math.hypot(float(b[0] - a[0]), float(b[1] - a[1]))

    def _should_publish_path(self, goal_cell, path):
        if not path:
            return False
        end_cell = path[-1]
        if self.last_published_goal_cell != goal_cell:
            return True

        now_sec = rospy.Time.now().to_sec()
        last_end = self.last_published_end_cell
        if last_end is None:
            return True

        # Always accept a full path to the snapped goal.
        if end_cell == goal_cell:
            return True

        old_dist = self._heur(last_end, goal_cell)
        new_dist = self._heur(end_cell, goal_cell)

        # Keep the previously published partial path unless the new one is
        # meaningfully better or the hold period has elapsed.
        if new_dist + self.best_effort_improve_margin_cells < old_dist:
            return True
        if (now_sec - self.last_path_publish_sec) >= self.best_effort_update_period_s and new_dist <= old_dist:
            return True
        return False

    def _record_published_path(self, goal_cell, path):
        self.last_published_goal_cell = goal_cell
        self.last_published_end_cell = path[-1] if path else None
        self.last_path_publish_sec = rospy.Time.now().to_sec()

    def _best_effort_path_is_acceptable(self, goal_cell, path, dg, label):
        if (not path) or path[-1] == goal_cell:
            return True
        gap_m = self._heur(path[-1], goal_cell) * max(1e-3, float(dg.info.resolution))
        if gap_m <= self.best_effort_max_goal_gap_m:
            return True
        rospy.logwarn_throttle(
            1.0,
            "constrained_local_replanner: rejecting best-effort %s path (snapped_goal=%s reached=%s gap=%.2fm)",
            label,
            str(goal_cell),
            str(path[-1]),
            gap_m,
        )
        return False

    def _astar(self, blocked, start, goal, allow_best_effort=False, orthogonal_only=False):
        w = len(blocked[0]) if blocked else 0
        h = len(blocked)
        if w <= 0 or h <= 0:
            return None
        sx, sy = start
        gx, gy = goal
        if sx < 0 or sy < 0 or sx >= w or sy >= h:
            return None
        if gx < 0 or gy < 0 or gx >= w or gy >= h:
            return None
        if blocked[sy][sx] or blocked[gy][gx]:
            return None

        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if not orthogonal_only:
            nbrs.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
        pq = []
        heapq.heappush(pq, (self._heur(start, goal), 0.0, start))
        parent = {start: None}
        g_cost = {start: 0.0}
        expanded = 0
        best = start
        best_h = self._heur(start, goal)

        while pq:
            _, gc, cur = heapq.heappop(pq)
            cur_h = self._heur(cur, goal)
            if cur_h < best_h:
                best_h = cur_h
                best = cur
            if cur == goal:
                break
            expanded += 1
            if expanded > self.max_expand:
                break
            cx, cy = cur
            for dx, dy in nbrs:
                nx = cx + dx
                ny = cy + dy
                if nx < 0 or ny < 0 or nx >= w or ny >= h:
                    continue
                if blocked[ny][nx]:
                    continue
                step = math.sqrt(2.0) if (dx != 0 and dy != 0) else 1.0
                ng = gc + step
                nb = (nx, ny)
                if nb not in g_cost or ng < g_cost[nb]:
                    g_cost[nb] = ng
                    parent[nb] = cur
                    f = ng + self._heur(nb, goal)
                    heapq.heappush(pq, (f, ng, nb))

        if goal not in parent:
            if allow_best_effort and best in parent and best != start:
                goal = best
            else:
                return None
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path

    def _has_line_of_sight(self, blocked, start, goal):
        x0, y0 = start
        x1, y1 = goal
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if not self._in_bounds_blocked(blocked, x0, y0) or blocked[y0][x0]:
                return False
            if x0 == x1 and y0 == y1:
                return True
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def _simplify_grid_path(self, path, blocked, grid_resolution_m, force=False):
        if (not force and not self.smooth_path_line_of_sight) or len(path) <= 2:
            return path

        simplified = [path[0]]
        anchor_idx = 0
        while anchor_idx < len(path) - 1:
            farthest_idx = anchor_idx + 1
            probe_idx = farthest_idx + 1
            while probe_idx < len(path):
                seg_len_m = self._heur(path[anchor_idx], path[probe_idx]) * max(1e-3, grid_resolution_m)
                if seg_len_m > self.max_los_segment_m:
                    break
                if not self._has_line_of_sight(blocked, path[anchor_idx], path[probe_idx]):
                    break
                farthest_idx = probe_idx
                probe_idx += 1
            simplified.append(path[farthest_idx])
            anchor_idx = farthest_idx
        return simplified

    def _grid_path_to_world_points(self, grid_path, dg, start_xy=None, end_xy=None):
        world_points = []
        for i, (gx, gy) in enumerate(grid_path):
            if self.simplify_stride > 1 and i not in (0, len(grid_path) - 1) and (i % self.simplify_stride != 0):
                continue
            world_points.append(self._grid_to_world(dg, gx, gy))

        if start_xy is not None and world_points:
            world_points[0] = (float(start_xy[0]), float(start_xy[1]))
        if end_xy is not None and len(world_points) >= 2:
            world_points[-1] = (float(end_xy[0]), float(end_xy[1]))
        return world_points

    def _publish_grid_path(
        self,
        publisher,
        grid_path,
        dg,
        stamp,
        start_xy=None,
        end_xy=None,
        anchor_start=True,
    ):
        out = Path()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = dg.header.frame_id if dg.header.frame_id else "map"
        resolved_start_xy = self._resolve_path_start_xy(start_xy) if anchor_start else start_xy
        world_points = self._grid_path_to_world_points(
            grid_path,
            dg,
            start_xy=start_xy,
            end_xy=end_xy,
        )
        if anchor_start:
            world_points = self._anchor_world_points_to_resolved_start(
                world_points,
                start_xy,
                resolved_start_xy,
            )

        if len(world_points) < 2:
            return

        sampled_points = self._sample_world_points(world_points)
        sampled_yaws = self._path_yaws(sampled_points)

        for (x, y), yaw in zip(sampled_points, sampled_yaws):
            ps = PoseStamped()
            ps.header = out.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            self._set_pose_yaw(ps, yaw)
            out.poses.append(ps)
        if len(out.poses) >= 2:
            if publisher is self.pub_local_path:
                self._publish_local_path_msg(out)
            else:
                publisher.publish(out)
        return sampled_points, out.header.frame_id

    def _sample_world_points(self, world_points):
        if len(world_points) < 2:
            return list(world_points)
        sampled_points = [world_points[0]]
        for i in range(len(world_points) - 1):
            x0, y0 = world_points[i]
            x1, y1 = world_points[i + 1]
            seg_len = math.hypot(x1 - x0, y1 - y0)
            if seg_len <= 1e-9:
                continue
            steps = max(1, int(math.ceil(seg_len / self.published_path_spacing_m)))
            for step in range(1, steps + 1):
                t = float(step) / float(steps)
                sampled_points.append((
                    x0 + t * (x1 - x0),
                    y0 + t * (y1 - y0),
                ))
        return sampled_points

    def _trim_world_points_from_start(self, world_points, trim_m):
        if len(world_points) < 2 or trim_m <= 1e-6:
            return list(world_points)
        if self._world_path_length(world_points) <= trim_m + 1e-6:
            return list(world_points)

        trimmed = [(float(x), float(y)) for x, y in world_points]
        remain = float(trim_m)
        while len(trimmed) >= 2 and remain > 1e-6:
            x0, y0 = trimmed[0]
            x1, y1 = trimmed[1]
            seg_len = math.hypot(x1 - x0, y1 - y0)
            if seg_len <= 1e-9:
                trimmed.pop(0)
                continue
            if remain < seg_len:
                t = remain / seg_len
                trimmed[0] = (
                    x0 + t * (x1 - x0),
                    y0 + t * (y1 - y0),
                )
                remain = 0.0
                break
            remain -= seg_len
            trimmed.pop(0)
        return trimmed

    def _trim_world_points_from_robot_front(self, world_points):
        if not self.trim_published_path_to_robot_front:
            return list(world_points)
        return self._trim_world_points_from_start(
            world_points,
            self.path_start_front_offset_m,
        )

    def _drop_initial_points_behind_start(self, world_points, start_xy):
        pts = list(world_points)
        if len(pts) < 3 or start_xy is None or not self.have_odom:
            return pts
        sx, sy = float(start_xy[0]), float(start_xy[1])
        yaw_c = math.cos(self.odom_yaw)
        yaw_s = math.sin(self.odom_yaw)
        min_forward_m = -max(0.03, min(0.12, 0.5 * self.published_path_spacing_m))
        first_keep = 1
        while first_keep < len(pts) - 1:
            px, py = pts[first_keep]
            forward_m = (float(px) - sx) * yaw_c + (float(py) - sy) * yaw_s
            if forward_m >= min_forward_m:
                break
            first_keep += 1
        if first_keep <= 1:
            return pts
        return [pts[0]] + pts[first_keep:]

    def _anchor_world_points_to_resolved_start(
        self,
        world_points,
        original_start_xy,
        resolved_start_xy,
    ):
        pts = list(world_points)
        if not pts:
            return pts
        if resolved_start_xy is None:
            return self._trim_world_points_from_robot_front(pts)

        trim_m = 0.0
        if original_start_xy is not None:
            trim_m = math.hypot(
                float(resolved_start_xy[0]) - float(original_start_xy[0]),
                float(resolved_start_xy[1]) - float(original_start_xy[1]),
            )
        elif self.have_odom:
            trim_m = math.hypot(
                float(resolved_start_xy[0]) - float(self.odom_x),
                float(resolved_start_xy[1]) - float(self.odom_y),
            )
        if self.trim_published_path_to_robot_front and trim_m > 1e-6:
            pts = self._trim_world_points_from_start(pts, trim_m)
        if not pts:
            pts = [(float(resolved_start_xy[0]), float(resolved_start_xy[1]))]
        else:
            pts[0] = (float(resolved_start_xy[0]), float(resolved_start_xy[1]))
        pts = self._drop_initial_points_behind_start(pts, resolved_start_xy)
        return self._dedupe_world_points(pts)

    @staticmethod
    def _set_pose_yaw(pose_stamped, yaw):
        pose_stamped.pose.orientation.x = 0.0
        pose_stamped.pose.orientation.y = 0.0
        pose_stamped.pose.orientation.z = math.sin(0.5 * yaw)
        pose_stamped.pose.orientation.w = math.cos(0.5 * yaw)

    @staticmethod
    def _pose_yaw(pose_msg):
        q = pose_msg.orientation
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    @staticmethod
    def _angle_diff(a, b):
        return math.atan2(math.sin(a - b), math.cos(a - b))

    @staticmethod
    def _path_yaws(points):
        if not points:
            return []
        if len(points) == 1:
            return [0.0]

        yaws = []
        last_yaw = 0.0
        for idx in range(len(points)):
            if idx == 0:
                ax, ay = points[idx]
                bx, by = points[idx + 1]
            elif idx == len(points) - 1:
                ax, ay = points[idx - 1]
                bx, by = points[idx]
            else:
                ax, ay = points[idx - 1]
                bx, by = points[idx + 1]
            dx = float(bx) - float(ax)
            dy = float(by) - float(ay)
            if abs(dx) <= 1e-6 and abs(dy) <= 1e-6:
                yaws.append(last_yaw)
                continue
            last_yaw = math.atan2(dy, dx)
            yaws.append(last_yaw)
        return yaws

    @staticmethod
    def _path_signature_from_points(points):
        return tuple((round(float(x), 2), round(float(y), 2)) for x, y in points)

    def _publish_path_history_markers(self):
        markers = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        branch_entries = [
            entry
            for entry in self.path_history_entries
            if entry["source"] == "avoidance" and len(entry.get("points", [])) >= 2
        ]
        total = len(branch_entries)
        now = rospy.Time.now()
        for idx, entry in enumerate(branch_entries):
            age_norm = 0.0 if total <= 1 else float(total - 1 - idx) / float(total - 1)
            alpha = 0.45 + 0.45 * (1.0 - age_norm)
            marker = Marker()
            marker.header.stamp = now
            marker.header.frame_id = entry["frame_id"]
            marker.ns = "avoidance_branch"
            marker.id = int(entry["id"] * 10)
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.08
            marker.color.a = alpha
            marker.color.r = 1.0
            marker.color.g = 0.55 + 0.20 * age_norm
            marker.color.b = 0.0
            points = entry.get("points", [])
            for x, y in points:
                p = Point()
                p.x = float(x)
                p.y = float(y)
                p.z = 0.12
                marker.points.append(p)
            markers.markers.append(marker)

            entry_marker = Marker()
            entry_marker.header.stamp = now
            entry_marker.header.frame_id = entry["frame_id"]
            entry_marker.ns = "avoidance_branch_entry"
            entry_marker.id = int(entry["id"] * 10 + 1)
            entry_marker.type = Marker.SPHERE
            entry_marker.action = Marker.ADD
            entry_marker.pose.orientation.w = 1.0
            entry_marker.scale.x = 0.12
            entry_marker.scale.y = 0.12
            entry_marker.scale.z = 0.12
            entry_marker.color.r = marker.color.r
            entry_marker.color.g = marker.color.g
            entry_marker.color.b = marker.color.b
            entry_marker.color.a = min(1.0, alpha + 0.05)
            entry_marker.pose.position.x = float(points[0][0])
            entry_marker.pose.position.y = float(points[0][1])
            entry_marker.pose.position.z = 0.14
            markers.markers.append(entry_marker)

            exit_marker = Marker()
            exit_marker.header.stamp = now
            exit_marker.header.frame_id = entry["frame_id"]
            exit_marker.ns = "avoidance_branch_exit"
            exit_marker.id = int(entry["id"] * 10 + 2)
            exit_marker.type = Marker.SPHERE
            exit_marker.action = Marker.ADD
            exit_marker.pose.orientation.w = 1.0
            exit_marker.scale.x = 0.10
            exit_marker.scale.y = 0.10
            exit_marker.scale.z = 0.10
            exit_marker.color.r = 1.0
            exit_marker.color.g = 0.75
            exit_marker.color.b = 0.20
            exit_marker.color.a = alpha
            exit_marker.pose.position.x = float(points[-1][0])
            exit_marker.pose.position.y = float(points[-1][1])
            exit_marker.pose.position.z = 0.14
            markers.markers.append(exit_marker)

        used_local_entries = [
            entry
            for entry in self.path_history_entries
            if entry["source"] == "used_local" and len(entry.get("points", [])) >= 2
        ]
        for idx, entry in enumerate(used_local_entries):
            age_norm = (
                0.0
                if len(used_local_entries) <= 1
                else float(len(used_local_entries) - 1 - idx)
                / float(len(used_local_entries) - 1)
            )
            alpha = 0.50 + 0.35 * (1.0 - age_norm)
            marker = Marker()
            marker.header.stamp = now
            marker.header.frame_id = entry["frame_id"]
            marker.ns = "local_path_used"
            marker.id = int(entry["id"] * 10)
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.10
            marker.color.a = alpha
            marker.color.r = 0.92
            marker.color.g = 0.08 + 0.06 * age_norm
            marker.color.b = 0.10 + 0.10 * age_norm
            for x, y in entry.get("points", []):
                p = Point()
                p.x = float(x)
                p.y = float(y)
                p.z = 0.10
                marker.points.append(p)
            markers.markers.append(marker)
        self.pub_path_history.publish(markers)

    def _publish_travel_history_marker(self):
        if self.pub_travel_history is None:
            return
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "map"
        marker.ns = "travel_history"
        marker.id = 1
        marker.pose.orientation.w = 1.0
        if len(self.travel_history_points) < 2:
            marker.action = Marker.DELETE
            self.pub_travel_history.publish(marker)
            return

        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.08
        marker.color.a = 1.00
        marker.color.r = 0.00
        marker.color.g = 0.45
        marker.color.b = 1.00
        for x, y in self.travel_history_points:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.10
            marker.points.append(p)
        self.pub_travel_history.publish(marker)
        rospy.loginfo_throttle(
            2.0,
            "constrained_local_replanner: travel_history marker/path points=%d topic=%s path_topic=%s",
            len(self.travel_history_points),
            self.travel_history_topic,
            self.travel_history_path_topic,
        )

    def _publish_travel_history_path(self):
        if self.pub_travel_history_path is None:
            return
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = "map"
        for x, y in self.travel_history_points:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.10
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.pub_travel_history_path.publish(path)

    def _record_travel_history_point(self, x, y):
        if self.pub_travel_history is None and self.pub_travel_history_path is None:
            return
        x = float(x)
        y = float(y)
        if self.travel_history_points:
            lx, ly = self.travel_history_points[-1]
            if math.hypot(x - lx, y - ly) < self.travel_history_spacing_m:
                return
        self.travel_history_points.append((x, y))
        self._publish_travel_history_marker()
        self._publish_travel_history_path()

    def _clear_travel_history(self):
        self.travel_history_points.clear()
        self._publish_travel_history_marker()
        self._publish_travel_history_path()

    def _record_path_history(self, source, sampled_points, frame_id):
        if source not in ("avoidance", "used_local"):
            return
        if len(sampled_points) < 2:
            return
        sig = self._path_signature_from_points(sampled_points)
        if self.last_history_signature.get(source) == sig:
            return
        self.last_history_signature[source] = sig
        self.path_history_next_id += 1
        self.path_history_entries.append(
            {
                "id": self.path_history_next_id,
                "source": source,
                "frame_id": frame_id if frame_id else "map",
                "points": list(sampled_points),
            }
        )
        self._publish_path_history_markers()

    def _clear_path_history(self):
        self.path_history_entries.clear()
        self.path_history_next_id = 0
        self.last_history_signature = {
            "local": None,
            "avoidance": None,
            "used_local": None,
        }
        self._reset_pending_used_local_trace()
        self._publish_path_history_markers()

    def _arm_pending_used_local_trace(self, sampled_points, frame_id):
        if sampled_points is None or len(sampled_points) < 2 or not self.have_odom:
            self._reset_pending_used_local_trace()
            return
        if (
            self.pending_used_local_points is not None
            and not self.pending_used_local_committed
        ):
            return
        sig = self._path_signature_from_points(sampled_points)
        if sig == self.pending_used_local_signature:
            return
        self.pending_used_local_points = list(sampled_points)
        self.pending_used_local_frame_id = frame_id if frame_id else "map"
        self.pending_used_local_origin_xy = (float(self.odom_x), float(self.odom_y))
        self.pending_used_local_signature = sig
        self.pending_used_local_committed = False

    def _maybe_commit_pending_used_local_trace(self):
        if (
            self.pending_used_local_points is None
            or self.pending_used_local_origin_xy is None
            or self.pending_used_local_committed
            or not self.have_odom
        ):
            return
        dist_m = math.hypot(
            float(self.odom_x) - float(self.pending_used_local_origin_xy[0]),
            float(self.odom_y) - float(self.pending_used_local_origin_xy[1]),
        )
        if dist_m < self.used_local_path_commit_distance_m:
            return
        self._record_path_history(
            "used_local",
            self.pending_used_local_points,
            self.pending_used_local_frame_id,
        )
        self.pending_used_local_committed = True

    def _reset_pending_used_local_trace(self):
        self.pending_used_local_points = None
        self.pending_used_local_frame_id = "map"
        self.pending_used_local_origin_xy = None
        self.pending_used_local_signature = None
        self.pending_used_local_committed = False

    def _publish_local_path(self, grid_path, dg, stamp, start_xy=None, end_xy=None):
        if start_xy is None and self.have_odom:
            start_xy = (self.odom_x, self.odom_y)
        sampled_points, frame_id = self._publish_grid_path(
            self.pub_local_path,
            grid_path,
            dg,
            stamp,
            start_xy=start_xy,
            end_xy=end_xy,
        )

    def _publish_avoidance_path(self, grid_path, dg, stamp, history_points=None, start_xy=None, end_xy=None, record_history=True):
        if start_xy is None and self.have_odom:
            start_xy = (self.odom_x, self.odom_y)
        publish_grid_path = (
            self._collapse_straight_grid_runs(grid_path)
            if self.avoidance_local_collapse_straights
            else list(grid_path)
        )
        self._publish_path_mode("follow_avoidance")
        sampled_points, frame_id = self._publish_grid_path(
            self.pub_local_path,
            publish_grid_path,
            dg,
            stamp,
            start_xy=start_xy,
            end_xy=end_xy,
        )
        self._publish_empty_path(self.pub_avoidance_path, frame_id, stamp)
        self.last_avoidance_grid_path = list(grid_path) if grid_path is not None else None
        self.last_avoidance_world_path = (
            list(sampled_points) if sampled_points is not None else None
        )
        self.last_avoidance_active_sec = stamp.to_sec()
        if record_history:
            self._record_path_history(
                "avoidance",
                history_points if history_points is not None else sampled_points,
                frame_id,
            )

    def _publish_global_mode_detour_path(
        self,
        grid_path,
        dg,
        stamp,
        history_points=None,
        start_xy=None,
        end_xy=None,
        record_history=True,
    ):
        display_grid_path = self._collapse_straight_grid_runs(grid_path)
        sampled_points, frame_id = self._publish_grid_path(
            self.pub_local_path,
            display_grid_path,
            dg,
            stamp,
            start_xy=start_xy,
            end_xy=end_xy,
            anchor_start=False,
        )
        self._publish_empty_path(self.pub_avoidance_path, frame_id, stamp)
        self._publish_path_mode("follow_avoidance")
        self.last_avoidance_grid_path = list(grid_path) if grid_path is not None else None
        self.last_avoidance_world_path = (
            list(sampled_points) if sampled_points is not None else None
        )
        self.last_avoidance_active_sec = stamp.to_sec()
        self._arm_pending_used_local_trace(
            history_points if history_points is not None else sampled_points,
            frame_id,
        )
        if record_history:
            self._record_path_history(
                "avoidance",
                history_points if history_points is not None else sampled_points,
                frame_id,
            )

    def _publish_operational_detour_path(
        self,
        grid_path,
        dg,
        stamp,
        history_points=None,
        start_xy=None,
        end_xy=None,
        record_history=True,
    ):
        if self._use_global_nominal_reference():
            self._publish_global_mode_detour_path(
                grid_path,
                dg,
                stamp,
                history_points=history_points,
                start_xy=start_xy,
                end_xy=end_xy,
                record_history=record_history,
            )
            return
        self._publish_avoidance_path(
            grid_path,
            dg,
            stamp,
            history_points=history_points,
            start_xy=start_xy,
            end_xy=end_xy,
            record_history=record_history,
        )

    def _world_points_to_grid_path(self, world_points, dg):
        if not world_points:
            return []
        out = []
        last_cell = None
        for wx, wy in world_points:
            gx, gy = self._world_to_grid(dg, wx, wy)
            if not self._in_bounds(dg, gx, gy):
                return []
            cell = (int(gx), int(gy))
            if cell != last_cell:
                out.append(cell)
                last_cell = cell
        return out

    def _publish_world_avoidance_path(
        self,
        world_points,
        dg,
        stamp,
        history_points=None,
        record_history=True,
    ):
        if world_points is None or len(world_points) < 2:
            return
        frame_id = dg.header.frame_id if dg.header.frame_id else "map"
        sampled_points = self._sample_world_points(world_points)
        grid_path = self._world_points_to_grid_path(sampled_points, dg)
        if len(grid_path) < 2:
            return
        self._publish_path_mode("follow_avoidance")
        self._publish_world_path(world_points, frame_id, stamp)
        self._publish_empty_path(self.pub_avoidance_path, frame_id, stamp)
        self.last_avoidance_grid_path = list(grid_path)
        self.last_avoidance_world_path = list(sampled_points)
        self.last_avoidance_active_sec = stamp.to_sec()
        if record_history:
            self._record_path_history(
                "avoidance",
                history_points if history_points is not None else sampled_points,
                frame_id,
            )

    def _publish_world_path(self, world_points, frame_id, stamp):
        resolved_start_xy = self._resolve_path_start_xy(None)
        world_points = self._anchor_world_points_to_resolved_start(
            world_points,
            (self.odom_x, self.odom_y) if self.have_odom else None,
            resolved_start_xy,
        )
        out = Path()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = frame_id if frame_id else "map"
        yaws = self._path_yaws(world_points)
        for (x, y), yaw in zip(world_points, yaws):
            ps = PoseStamped()
            ps.header = out.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            self._set_pose_yaw(ps, yaw)
            out.poses.append(ps)
        if len(out.poses) >= 2:
            self._publish_local_path_msg(out)

    @staticmethod
    def _collapse_straight_grid_runs(grid_path):
        if not grid_path:
            return []
        if len(grid_path) < 3:
            return list(grid_path)

        def _step_dir(a, b):
            dx = int(b[0]) - int(a[0])
            dy = int(b[1]) - int(a[1])
            return (
                0 if dx == 0 else (1 if dx > 0 else -1),
                0 if dy == 0 else (1 if dy > 0 else -1),
            )

        collapsed = [grid_path[0]]
        prev_dir = _step_dir(grid_path[0], grid_path[1])
        for idx in range(1, len(grid_path) - 1):
            curr = grid_path[idx]
            next_dir = _step_dir(curr, grid_path[idx + 1])
            if next_dir != prev_dir:
                collapsed.append(curr)
                prev_dir = next_dir
        collapsed.append(grid_path[-1])
        return collapsed

    @staticmethod
    def _trace_grid_segment_cells(start_cell, end_cell):
        x0, y0 = int(start_cell[0]), int(start_cell[1])
        x1, y1 = int(end_cell[0]), int(end_cell[1])
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        cells = []
        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return cells

    def _compose_grid_path_from_waypoints(self, waypoints):
        if not waypoints:
            return []
        out = []
        for idx in range(len(waypoints) - 1):
            seg = self._trace_grid_segment_cells(waypoints[idx], waypoints[idx + 1])
            self._append_path_segment(out, seg)
        if not out:
            out.append((int(waypoints[0][0]), int(waypoints[0][1])))
        return out

    def _build_box_avoidance_path(
        self,
        dynamic_blocked,
        nominal_path,
        start_idx,
        start_cell,
        branch_start_idx,
        branch_start_cell,
        blocked_idx,
        dg,
        res_m,
        blocking_cells_world=None,
        blocking_points_world=None,
        preferred_direction=None,
    ):
        if blocked_idx < 0 or blocked_idx >= len(nominal_path):
            return None

        ref_prev = nominal_path[max(start_idx, blocked_idx - 1)]
        ref_next = nominal_path[min(len(nominal_path) - 1, blocked_idx + 1)]
        dir_dx = int(ref_next[0]) - int(ref_prev[0])
        dir_dy = int(ref_next[1]) - int(ref_prev[1])
        if abs(dir_dx) >= abs(dir_dy):
            step_x = 1 if dir_dx >= 0 else -1
            step_y = 0
        else:
            step_x = 0
            step_y = 1 if dir_dy >= 0 else -1

        if step_x == 0 and step_y == 0:
            return None

        _obstacle_key, stable_obstacle_points, _matched_entries = (
            self._stable_avoidance_obstacle_points(
                blocking_cells_world=blocking_cells_world,
                blocking_points_world=blocking_points_world,
            )
        )
        obstacle_cells = []
        for wx, wy in stable_obstacle_points:
            gx, gy = self._world_to_grid(dg, wx, wy)
            if self._in_bounds_blocked(dynamic_blocked, gx, gy):
                obstacle_cells.append((int(gx), int(gy)))

        if not obstacle_cells:
            lo = max(branch_start_idx, blocked_idx - 1)
            hi = min(len(nominal_path), blocked_idx + 2)
            obstacle_cells.extend(
                (int(gx), int(gy)) for gx, gy in nominal_path[lo:hi]
            )
        if not obstacle_cells:
            return None

        min_x = min(cell[0] for cell in obstacle_cells)
        max_x = max(cell[0] for cell in obstacle_cells)
        min_y = min(cell[1] for cell in obstacle_cells)
        max_y = max(cell[1] for cell in obstacle_cells)

        branch_gx = int(branch_start_cell[0])
        branch_gy = int(branch_start_cell[1])
        if not self._in_bounds_blocked(dynamic_blocked, branch_gx, branch_gy):
            return None

        lateral_clear_base_cells = max(
            1,
            int(
                math.ceil(
                    (
                        self.robot_half_width
                        + self.obstacle_block_margin_m
                        + min(self.avoidance_trigger_margin_m, 0.10)
                        + 0.05
                    )
                    / max(1e-3, res_m)
                )
            ),
        )
        approach_clear_base_cells = max(
            1,
            int(
                math.ceil(
                    (
                        self.robot_length_m * 0.35
                        + self.obstacle_block_margin_m
                        + 0.05
                    )
                    / max(1e-3, res_m)
                )
            ),
        )
        forward_clear_base_cells = max(
            approach_clear_base_cells + 1,
            int(
                math.ceil(
                    (
                        max(self.avoidance_rejoin_min_distance_m, self.robot_length_m * 0.9)
                        + self.obstacle_block_margin_m
                    )
                    / max(1e-3, res_m)
                )
            ),
        )
        lateral_clear_candidates = sorted(
            {
                lateral_clear_base_cells,
                lateral_clear_base_cells + 1,
                lateral_clear_base_cells + 2,
                lateral_clear_base_cells + 4,
            }
        )
        approach_clear_candidates = sorted(
            {
                approach_clear_base_cells,
                approach_clear_base_cells + 1,
                approach_clear_base_cells + 2,
            }
        )
        forward_clear_candidates = sorted(
            {
                forward_clear_base_cells,
                forward_clear_base_cells + 1,
                forward_clear_base_cells + 2,
                forward_clear_base_cells + 4,
            }
        )

        candidates = []
        if step_x != 0:
            lane_y = branch_gy
            for approach_clear_cells in approach_clear_candidates:
                if step_x > 0:
                    entry_x = max(branch_gx, min_x - approach_clear_cells)
                else:
                    entry_x = min(branch_gx, max_x + approach_clear_cells)
                for forward_clear_cells in forward_clear_candidates:
                    if step_x > 0:
                        exit_x = max(entry_x + 1, max_x + forward_clear_cells)
                    else:
                        exit_x = min(entry_x - 1, min_x - forward_clear_cells)
                    side_values = []
                    for lateral_clear_cells in lateral_clear_candidates:
                        side_values.extend(
                            [min_y - lateral_clear_cells, max_y + lateral_clear_cells]
                        )
                    side_values = sorted(set(side_values), key=lambda val: abs(val - lane_y))
                    for side_y in side_values:
                        candidates.append(
                            [
                                (branch_gx, lane_y),
                                (entry_x, lane_y),
                                (entry_x, side_y),
                                (exit_x, side_y),
                                (exit_x, lane_y),
                            ]
                        )
        else:
            lane_x = branch_gx
            for approach_clear_cells in approach_clear_candidates:
                if step_y > 0:
                    entry_y = max(branch_gy, min_y - approach_clear_cells)
                else:
                    entry_y = min(branch_gy, max_y + approach_clear_cells)
                for forward_clear_cells in forward_clear_candidates:
                    if step_y > 0:
                        exit_y = max(entry_y + 1, max_y + forward_clear_cells)
                    else:
                        exit_y = min(entry_y - 1, min_y - forward_clear_cells)
                    side_values = []
                    for lateral_clear_cells in lateral_clear_candidates:
                        side_values.extend(
                            [min_x - lateral_clear_cells, max_x + lateral_clear_cells]
                        )
                    side_values = sorted(set(side_values), key=lambda val: abs(val - lane_x))
                    for side_x in side_values:
                        candidates.append(
                            [
                                (lane_x, branch_gy),
                                (lane_x, entry_y),
                                (side_x, entry_y),
                                (side_x, exit_y),
                                (lane_x, exit_y),
                            ]
                        )

        best_path = None
        best_cost = None
        for raw_waypoints in candidates:
            box_waypoints = []
            for cell in raw_waypoints:
                cell = (int(cell[0]), int(cell[1]))
                if box_waypoints and cell == box_waypoints[-1]:
                    continue
                box_waypoints.append(cell)
            if len(box_waypoints) < 4:
                continue
            if any(
                (not self._in_bounds_blocked(dynamic_blocked, wx, wy))
                or dynamic_blocked[wy][wx]
                for wx, wy in box_waypoints
            ):
                continue
            clear = True
            for seg_idx in range(len(box_waypoints) - 1):
                if not self._has_line_of_sight(
                    dynamic_blocked,
                    box_waypoints[seg_idx],
                    box_waypoints[seg_idx + 1],
                ):
                    clear = False
                    break
            if not clear:
                continue

            detour = self._compose_grid_path_from_waypoints(box_waypoints)
            detour = self._collapse_straight_grid_runs(detour)
            if len(detour) < 2:
                continue

            if step_x != 0:
                lateral_extent = max(abs(box_waypoints[2][1] - branch_gy), 0)
                forward_extent = abs(box_waypoints[3][0] - box_waypoints[2][0])
            else:
                lateral_extent = max(abs(box_waypoints[2][0] - branch_gx), 0)
                forward_extent = abs(box_waypoints[3][1] - box_waypoints[2][1])
            candidate_direction, _candidate_offset = self._infer_avoid_direction(
                dg, detour
            )
            side_penalty = 0
            if (
                self.avoidance_same_side_commitment_enabled
                and preferred_direction in ("left", "right")
                and candidate_direction not in (preferred_direction, "none")
            ):
                side_penalty = 1
            cost = (side_penalty, lateral_extent, forward_extent, len(detour))
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_path = detour

        return best_path

    def _path_index_after_distance(self, path, start_idx, dg, target_dist_m):
        if not path:
            return 0
        idx = max(0, min(int(start_idx), len(path) - 1))
        if target_dist_m <= 1e-6 or idx >= len(path) - 1:
            return idx

        remain_m = 0.0
        scale = max(1e-3, float(dg.info.resolution))
        while idx + 1 < len(path):
            remain_m += self._heur(path[idx], path[idx + 1]) * scale
            idx += 1
            if remain_m >= target_dist_m:
                break
        return idx

    def _chaikin_smooth_world_points(self, world_points, passes=1):
        if len(world_points) < 3 or passes <= 0:
            return [(float(x), float(y)) for x, y in world_points]

        smoothed = [(float(x), float(y)) for x, y in world_points]
        for _ in range(int(passes)):
            if len(smoothed) < 3:
                break
            next_points = [smoothed[0]]
            for idx in range(len(smoothed) - 1):
                x0, y0 = smoothed[idx]
                x1, y1 = smoothed[idx + 1]
                q = (0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1)
                r = (0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1)
                next_points.append(q)
                next_points.append(r)
            next_points.append(smoothed[-1])
            smoothed = self._dedupe_world_points(next_points)
        return smoothed

    def _curved_world_path_metrics(self, world_points):
        sampled_points = self._sample_world_points(world_points)
        max_curvature = 0.0
        max_heading_delta = 0.0
        prev_yaw = None
        for idx in range(len(sampled_points) - 1):
            x0, y0 = sampled_points[idx]
            x1, y1 = sampled_points[idx + 1]
            seg_len = math.hypot(float(x1) - float(x0), float(y1) - float(y0))
            if seg_len <= 1e-6:
                continue
            yaw = math.atan2(float(y1) - float(y0), float(x1) - float(x0))
            if prev_yaw is not None:
                heading_delta = abs(self._angle_diff(yaw, prev_yaw))
                max_heading_delta = max(max_heading_delta, heading_delta)
                max_curvature = max(
                    max_curvature, heading_delta / max(seg_len, 1e-3)
                )
            prev_yaw = yaw
        return sampled_points, max_curvature, max_heading_delta

    def _candidate_world_path_is_safe(
        self,
        world_points,
        dynamic_blocked,
        dg,
        start_cell,
        now_sec=None,
    ):
        world_points = self._dedupe_world_points(world_points)
        if len(world_points) < 2:
            return None, None, None

        sampled_points, max_curvature, max_heading_delta = (
            self._curved_world_path_metrics(world_points)
        )
        if len(sampled_points) < 2:
            return None, None, None
        if max_curvature > self.short_curved_avoidance_max_curvature:
            return None, None, None
        if max_heading_delta > self.short_curved_avoidance_max_heading_delta_rad:
            return None, None, None

        for idx, (wx, wy) in enumerate(sampled_points):
            if idx == 0:
                continue
            lx, _ly = self._world_to_local(wx, wy)
            if lx < -0.10:
                return None, None, None

        grid_path = self._world_points_to_grid_path(sampled_points, dg)
        if len(grid_path) < 2:
            return None, None, None

        for gx, gy in grid_path:
            if (not self._in_bounds_blocked(dynamic_blocked, gx, gy)) or dynamic_blocked[gy][gx]:
                return None, None, None
        for idx in range(len(grid_path) - 1):
            if not self._has_line_of_sight(
                dynamic_blocked, grid_path[idx], grid_path[idx + 1]
            ):
                return None, None, None
        if self._path_blind_zone_turn_conflict(
            grid_path, dg, now_sec=now_sec
        ) is not None:
            return None, None, None
        if self._path_blocked_ahead(
            grid_path,
            dynamic_blocked,
            start_cell,
            float(dg.info.resolution),
            max_check_m=self.short_curved_avoidance_preview_m,
        ):
            return None, None, None

        return sampled_points, grid_path, {
            "max_curvature": float(max_curvature),
            "max_heading_delta_deg": float(math.degrees(max_heading_delta)),
        }

    def _build_short_curved_avoidance_path(
        self,
        nominal_path,
        dynamic_blocked,
        start_cell,
        dg,
        blocked_idx=None,
        blocking_cells_world=None,
        blocking_points_world=None,
        preferred_direction=None,
        now_sec=None,
    ):
        if (not self.short_curved_avoidance_enabled) or len(nominal_path) < 2:
            return None, None, None, None

        preferred_direction = (
            str(preferred_direction).strip().lower()
            if preferred_direction is not None
            else None
        )
        if preferred_direction not in ("left", "right"):
            preferred_direction = None

        start_idx = self._nearest_path_cell_index(nominal_path, start_cell)
        if blocked_idx is None:
            blocked_idx = self._first_blocked_path_index(
                nominal_path,
                dynamic_blocked,
                start_cell,
                dg,
                max_check_m=self.avoidance_trigger_ahead_m,
                include_pointcloud=self.use_pointcloud_avoidance_trigger,
            )
        if blocked_idx is None:
            return None, None, None, None

        res_m = max(1e-3, float(dg.info.resolution))
        blocked_delta_cells = max(0, blocked_idx - start_idx)
        close_blocked_cells = max(3, int(math.ceil(1.0 / res_m)))
        backtrack_cells = self.avoidance_branch_backtrack_cells
        if blocked_delta_cells <= close_blocked_cells:
            backtrack_cells = max(backtrack_cells, int(math.ceil(0.8 / res_m)))

        branch_start_idx = max(start_idx, blocked_idx - backtrack_cells)
        while branch_start_idx > start_idx:
            bx, by = nominal_path[branch_start_idx]
            if self._in_bounds_blocked(dynamic_blocked, bx, by) and not dynamic_blocked[by][bx]:
                break
            branch_start_idx -= 1
        branch_start_cell = nominal_path[branch_start_idx]

        box_path = self._build_box_avoidance_path(
            dynamic_blocked,
            nominal_path,
            start_idx,
            start_cell,
            branch_start_idx,
            branch_start_cell,
            blocked_idx,
            dg,
            res_m,
            blocking_cells_world=blocking_cells_world,
            blocking_points_world=blocking_points_world,
            preferred_direction=preferred_direction,
        )
        if box_path is None or len(box_path) < 2:
            return None, None, None, None

        preview_end_idx = self._path_index_after_distance(
            nominal_path, start_idx, dg, self.short_curved_avoidance_preview_m
        )
        rejoin_idx = self._nearest_path_cell_index(nominal_path, box_path[-1])
        tail_end_idx = max(
            preview_end_idx,
            self._path_index_after_distance(
                nominal_path, rejoin_idx, dg, self.short_curved_avoidance_tail_m
            ),
        )

        composed = []
        self._append_path_segment(composed, [start_cell])
        self._append_path_segment(composed, nominal_path[start_idx:branch_start_idx + 1])
        self._append_path_segment(composed, box_path[1:])

        rejoin_cell = nominal_path[rejoin_idx]
        if composed[-1] != rejoin_cell:
            connector = self._trace_grid_segment_cells(composed[-1], rejoin_cell)
            self._append_path_segment(composed, connector[1:])
        if tail_end_idx > rejoin_idx:
            self._append_path_segment(
                composed, nominal_path[rejoin_idx + 1 : tail_end_idx + 1]
            )
        composed = self._collapse_straight_grid_runs(composed)
        if len(composed) < 3:
            return None, None, None, None

        start_xy = self._resolve_path_start_xy((self.odom_x, self.odom_y))
        world_points = self._grid_path_to_world_points(
            composed,
            dg,
            start_xy=start_xy,
        )
        world_points = self._dedupe_world_points(world_points)
        if len(world_points) < 3:
            return None, None, None, None

        best_candidate = None
        for smooth_passes in range(self.short_curved_avoidance_smooth_passes, 0, -1):
            candidate_world = self._chaikin_smooth_world_points(
                world_points, passes=smooth_passes
            )
            sampled_points, grid_path, metrics = self._candidate_world_path_is_safe(
                candidate_world,
                dynamic_blocked,
                dg,
                start_cell,
                now_sec=now_sec,
            )
            if sampled_points is None or grid_path is None:
                continue
            best_candidate = (
                candidate_world,
                grid_path,
                sampled_points,
                metrics,
            )
            break

        if best_candidate is None:
            return None, None, None, None
        return best_candidate

    def _build_sidestep_avoidance_path(
        self,
        nominal_path,
        dynamic_blocked,
        start_cell,
        dg,
        blocked_idx=None,
        blocking_cells_world=None,
        blocking_points_world=None,
        preferred_direction=None,
        now_sec=None,
    ):
        if (
            (not self.sidestep_avoidance_enabled)
            or len(nominal_path) < 2
            or (not self.have_odom)
        ):
            return None, None, None, None

        preferred_direction = (
            str(preferred_direction).strip().lower()
            if preferred_direction is not None
            else None
        )
        if preferred_direction not in ("left", "right"):
            preferred_direction = None

        start_idx = self._nearest_path_cell_index(nominal_path, start_cell)
        if blocked_idx is None:
            blocked_idx = self._first_blocked_path_index(
                nominal_path,
                dynamic_blocked,
                start_cell,
                dg,
                max_check_m=self.avoidance_trigger_ahead_m,
                include_pointcloud=self.use_pointcloud_avoidance_trigger,
            )
        if blocked_idx is None:
            return None, None, None, None

        _obstacle_key, stable_points, _matched_entries = (
            self._stable_avoidance_obstacle_points(
                blocking_cells_world=blocking_cells_world,
                blocking_points_world=blocking_points_world,
            )
        )
        obstacle_local = []
        local_horizon_m = max(
            self.sidestep_avoidance_preview_m + self.sidestep_avoidance_forward_margin_m,
            self.avoidance_rejoin_min_distance_m + self.robot_length_m,
        )
        lateral_limit_m = (
            self.sidestep_avoidance_max_offset_m
            + self.robot_half_width
            + self.obstacle_block_margin_m
            + 0.20
        )
        for wx, wy in stable_points:
            lx, ly = self._world_to_local(wx, wy)
            if lx < -0.20 or lx > local_horizon_m:
                continue
            if abs(ly) > lateral_limit_m:
                continue
            obstacle_local.append((float(lx), float(ly)))

        if not obstacle_local:
            bx, by = nominal_path[blocked_idx]
            wx, wy = self._grid_to_world(dg, bx, by)
            lx, ly = self._world_to_local(wx, wy)
            if lx < -0.20 or lx > local_horizon_m:
                return None, None, None, None
            obstacle_local.append((float(lx), float(ly)))

        min_x = min(lx for lx, _ly in obstacle_local)
        max_x = max(lx for lx, _ly in obstacle_local)
        min_y = min(ly for _lx, ly in obstacle_local)
        max_y = max(ly for _lx, ly in obstacle_local)
        center_y = 0.5 * (min_y + max_y)
        start_x = max(0.0, float(self.path_start_front_offset_m))

        side_order = []
        if preferred_direction == "left":
            side_order = [1, -1]
        elif preferred_direction == "right":
            side_order = [-1, 1]
        else:
            side_order = [-1, 1] if center_y >= 0.0 else [1, -1]

        clearance_y = (
            self.robot_half_width
            + self.obstacle_block_margin_m
            + min(self.avoidance_trigger_margin_m, 0.15)
            + 0.08
        )
        preview_x = max(
            self.sidestep_avoidance_preview_m,
            max_x + self.sidestep_avoidance_forward_margin_m + self.robot_length_m,
            start_x + 1.0,
        )

        best_candidate = None
        best_score = None
        for side in side_order:
            if side > 0:
                required_offset = max_y + clearance_y
            else:
                required_offset = -(min_y - clearance_y)
            required_offset = max(0.0, float(required_offset))
            offset_candidates = sorted(
                {
                    self.sidestep_avoidance_min_offset_m,
                    required_offset,
                    required_offset + 0.15,
                    required_offset + 0.30,
                }
            )
            for offset_m in offset_candidates:
                offset_m = max(
                    self.sidestep_avoidance_min_offset_m, float(offset_m)
                )
                if offset_m > self.sidestep_avoidance_max_offset_m + 1e-6:
                    continue
                target_y = float(side) * offset_m
                entry_x = max(start_x + 0.15, min_x - 0.20)
                pass_x = max(
                    entry_x + 0.35,
                    max_x + self.sidestep_avoidance_forward_margin_m,
                )
                end_x = max(preview_x, pass_x + 0.35)
                mid_x = start_x + 0.5 * max(0.20, entry_x - start_x)
                local_waypoints = [
                    (start_x, 0.0),
                    (mid_x, 0.0),
                    (entry_x, 0.35 * target_y),
                    (0.5 * (entry_x + pass_x), 0.80 * target_y),
                    (pass_x, target_y),
                    (end_x, target_y),
                ]
                world_points = [
                    self._local_to_map(local_x, local_y)
                    for local_x, local_y in local_waypoints
                ]
                world_points = self._dedupe_world_points(world_points)
                if len(world_points) < 3:
                    continue

                for smooth_passes in range(
                    self.short_curved_avoidance_smooth_passes, -1, -1
                ):
                    if smooth_passes > 0:
                        candidate_world = self._chaikin_smooth_world_points(
                            world_points, passes=smooth_passes
                        )
                    else:
                        candidate_world = world_points
                    sampled_points, grid_path, metrics = (
                        self._candidate_world_path_is_safe(
                            candidate_world,
                            dynamic_blocked,
                            dg,
                            start_cell,
                            now_sec=now_sec,
                        )
                    )
                    if sampled_points is None or grid_path is None:
                        continue
                    score = (
                        0 if preferred_direction is None else int(side_order[0] != side),
                        offset_m,
                        float(metrics.get("max_curvature", 0.0))
                        if metrics is not None
                        else 0.0,
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best_candidate = (
                            candidate_world,
                            grid_path,
                            sampled_points,
                            metrics,
                        )
                    break

        if best_candidate is None:
            return None, None, None, None
        return best_candidate

    @staticmethod
    def _dedupe_world_points(world_points):
        deduped = []
        for x, y in world_points:
            if deduped and math.hypot(float(x) - deduped[-1][0], float(y) - deduped[-1][1]) <= 1e-3:
                continue
            deduped.append((float(x), float(y)))
        return deduped

    def _publish_nominal_local_segment(self, pts, i0, ig, blocked, start_cell, dg, stamp):
        grid_path, sampled_points, _ = self._build_nominal_local_path(pts, i0, ig, dg)
        if grid_path is None or sampled_points is None:
            return False
        if self._path_blocked_ahead(
            grid_path,
            blocked,
            start_cell,
            float(dg.info.resolution),
            max_check_m=self.lookahead_m,
        ):
            return False
        self._publish_world_path(sampled_points, dg.header.frame_id, stamp)
        return True

    def _should_follow_nominal_local_on_no_solution(
        self,
        label,
        trigger_reason,
        source_summary=None,
        now_sec=None,
    ):
        if self._use_global_nominal_reference():
            return False

        label_str = str(label or "").strip().lower()
        if label_str not in ("local", "local_hold"):
            return False

        reason = str(trigger_reason or "").strip().lower()
        if reason not in ("predicted_overlap", "dynamic_points_overlap"):
            return False

        weak_grid_fallback = self._weak_grid_no_solution_fallback_allowed(
            source_summary
        )
        if (
            not self.allow_nominal_local_fallback_on_no_solution
            and not weak_grid_fallback
        ):
            return False

        if source_summary is not None:
            if str(source_summary.get("blind_zone", "none")).strip().lower() != "none":
                return False
            if int(source_summary.get("risk", 0)) > 0:
                return False
            if int(source_summary.get("tracked_current", 0)) > 0:
                return False
            if int(source_summary.get("tracked_memory", 0)) > 0:
                return False

        if self.tracked_object_count > 0 or self.tracked_object_memory_count > 0:
            return False
        if self._effective_raw_near_obstacle_points_map(now_sec=now_sec):
            return False
        return True

    def _weak_grid_no_solution_fallback_allowed(self, summary):
        if not self.weak_grid_no_solution_fallback_enabled or not summary:
            return False
        if str(summary.get("blind_zone", "none")).strip().lower() != "none":
            return False
        grid_occ = int(summary.get("grid_occ", 0))
        if grid_occ <= 0 or grid_occ > self.weak_grid_no_solution_fallback_max_cells:
            return False
        if int(summary.get("risk", 0)) > 0:
            return False
        if int(summary.get("pc_current", 0)) > 0:
            return False
        if int(summary.get("map_filtered_path", 0)) > 0:
            return False
        if (
            int(summary.get("pc_memory", 0))
            > self.weak_grid_no_solution_fallback_max_memory_points
        ):
            return False
        if int(summary.get("tracked_current", 0)) > 0:
            return False
        if int(summary.get("tracked_memory", 0)) > 0:
            return False
        return True

    def _build_nominal_world_segment(self, pts, i0, ig):
        if not pts:
            return None
        end_idx = max(i0 + 1, ig)
        segment_world = [pts[i] for i in range(i0, min(len(pts), end_idx + 1))]
        if not segment_world:
            return None
        if len(segment_world) == 1:
            # Keep a short near-goal segment alive by preserving the goal point
            # and prepending the current robot pose instead of overwriting it.
            segment_world = [(self.odom_x, self.odom_y), segment_world[0]]
        else:
            segment_world[0] = (self.odom_x, self.odom_y)
        segment_world = self._dedupe_world_points(segment_world)
        if len(segment_world) < 2:
            return None
        return segment_world

    @staticmethod
    def _world_path_length(world_points):
        if world_points is None or len(world_points) < 2:
            return 0.0
        total = 0.0
        for idx in range(len(world_points) - 1):
            x0, y0 = world_points[idx]
            x1, y1 = world_points[idx + 1]
            total += math.hypot(float(x1) - float(x0), float(y1) - float(y0))
        return total

    def _build_nominal_local_path(self, pts, i0, ig, dg):
        segment_world = self._build_nominal_world_segment(pts, i0, ig)
        if segment_world is None:
            return None, None, "segment_too_short"
        sampled_points = self._sample_world_points(segment_world)
        grid_path = []
        last_cell = None
        for wx, wy in sampled_points:
            gx, gy = self._world_to_grid(dg, wx, wy)
            if not self._in_bounds(dg, gx, gy):
                return None, sampled_points, "out_of_bounds"
            cell = (gx, gy)
            if cell != last_cell:
                grid_path.append(cell)
                last_cell = cell
        if len(grid_path) < 2:
            return None, sampled_points, "degenerate_grid"
        return grid_path, sampled_points, ""

    def _publish_empty_path(self, publisher, frame_id, stamp):
        out = Path()
        out.header.stamp = stamp
        out.header.frame_id = frame_id if frame_id else "map"
        publisher.publish(out)
        if publisher is self.pub_local_path:
            self._forget_local_path_msg()

    def _clear_local_path(self, frame_id, stamp, force=False):
        self._publish_empty_path(self.pub_local_path, frame_id, stamp)
        if self._use_global_nominal_reference():
            self._reset_pending_used_local_trace()

    def _clear_avoidance_path(self, frame_id, stamp, force=False):
        if self.avoidance_active and (not force):
            now_sec = stamp.to_sec()
            if now_sec > 0.0 and (now_sec - self.last_avoidance_publish_sec) < self.avoidance_hold_s:
                return
            self.avoidance_clear_count += 1
            if self.avoidance_clear_count < self.avoidance_clear_confirm_cycles:
                return
        self._publish_empty_path(self.pub_avoidance_path, frame_id, stamp)
        if self._use_global_nominal_reference():
            self._clear_local_path(frame_id, stamp, force=force)
        if self.avoidance_active:
            self.avoidance_active = False
            rospy.loginfo("constrained_local_replanner: avoidance path cleared")
        self._reset_pending_avoidance_trigger()
        self.last_avoidance_trigger_reason = ""
        self.last_avoidance_direction = "none"
        self.active_avoidance_obstacle_key = None
        self.avoidance_clear_count = 0
        self.last_avoidance_publish_sec = 0.0
        self.last_avoidance_grid_path = None
        self.last_avoidance_world_path = None
        self.last_avoidance_solution_sec = 0.0
        self.last_avoidance_validation_sec = 0.0

    def _grid_path_min_distance_to_xy(self, grid_path, dg, x, y):
        if not grid_path:
            return float("inf")
        best = float("inf")
        for gx, gy in grid_path:
            wx, wy = self._grid_to_world(dg, gx, gy)
            d = math.hypot(float(wx) - float(x), float(wy) - float(y))
            if d < best:
                best = d
        return best

    def _grid_path_endpoint_distance_to_xy(self, grid_path, dg, x, y):
        if not grid_path:
            return float("inf")
        wx, wy = self._grid_to_world(dg, grid_path[-1][0], grid_path[-1][1])
        return math.hypot(float(wx) - float(x), float(wy) - float(y))

    def _publish_stored_avoidance_path(self, dg, stamp, record_history=False):
        if self.last_avoidance_world_path is not None and len(self.last_avoidance_world_path) >= 2:
            self._publish_world_avoidance_path(
                self.last_avoidance_world_path,
                dg,
                stamp,
                record_history=record_history,
            )
            return True
        if self.last_avoidance_grid_path is not None and len(self.last_avoidance_grid_path) >= 2:
            self._publish_operational_detour_path(
                self.last_avoidance_grid_path,
                dg,
                stamp,
                start_xy=(self.odom_x, self.odom_y),
                record_history=record_history,
            )
            return True
        return False

    def _hold_active_avoidance_until_endpoint(self, dg, stamp, clear_reason="clear"):
        if (not self.avoidance_active) or self.last_avoidance_grid_path is None:
            return False
        if len(self.last_avoidance_grid_path) < 2:
            return False
        if (
            str(clear_reason) == "nominal_path_clear"
            and self.last_avoidance_solution_sec > 0.0
        ):
            clear_age_s = stamp.to_sec() - self.last_avoidance_solution_sec
            if self.avoidance_clear_detour_hold_s <= 0.0:
                frame_id = dg.header.frame_id if dg.header.frame_id else "map"
                rospy.loginfo_throttle(
                    1.0,
                    "constrained_local_replanner: releasing active avoidance immediately after nominal path clear | age=%.2fs",
                    clear_age_s,
                )
                self._clear_avoidance_path(frame_id, stamp, force=True)
                return False
            if clear_age_s > self.avoidance_clear_detour_hold_s:
                frame_id = dg.header.frame_id if dg.header.frame_id else "map"
                rospy.loginfo_throttle(
                    1.0,
                    "constrained_local_replanner: releasing active avoidance after clear hold | age=%.2fs limit=%.2fs",
                    clear_age_s,
                    self.avoidance_clear_detour_hold_s,
                )
                self._clear_avoidance_path(frame_id, stamp, force=True)
                return False
        endpoint_dist_m = self._grid_path_endpoint_distance_to_xy(
            self.last_avoidance_grid_path,
            dg,
            self.odom_x,
            self.odom_y,
        )
        if endpoint_dist_m <= self.avoidance_keep_until_endpoint_distance_m:
            return False
        deviation_limit_m = max(
            0.45,
            self.avoidance_reuse_max_deviation_m,
            self.robot_length_m * 1.6,
            self.robot_width_m * 2.2,
        )
        deviation_m = self._grid_path_min_distance_to_xy(
            self.last_avoidance_grid_path,
            dg,
            self.odom_x,
            self.odom_y,
        )
        if deviation_m > deviation_limit_m:
            return False
        if self._path_blind_zone_turn_conflict(
            self.last_avoidance_grid_path, dg, now_sec=stamp.to_sec()
        ) is not None:
            return False
        if not self._publish_stored_avoidance_path(dg, stamp, record_history=False):
            return False
        self.avoidance_clear_count = 0
        self.last_avoidance_publish_sec = stamp.to_sec()
        rospy.loginfo_throttle(
            1.0,
            "constrained_local_replanner: keeping active avoidance until detour end | end=%.2fm dev=%.2fm reason=%s",
            endpoint_dist_m,
            deviation_m,
            str(clear_reason),
        )
        return True

    def _republish_last_avoidance_path(self, dg, stamp):
        if not self.allow_avoidance_reuse_on_no_solution:
            return False
        if self.last_avoidance_grid_path is None or len(self.last_avoidance_grid_path) < 2:
            return False
        reuse_reference_sec = max(
            float(self.last_avoidance_solution_sec),
            float(self.last_avoidance_publish_sec),
        )
        if reuse_reference_sec <= 0.0:
            return False
        age_s = stamp.to_sec() - reuse_reference_sec
        if age_s > self.avoidance_reuse_on_failure_s:
            return False
        deviation_m = self._grid_path_min_distance_to_xy(
            self.last_avoidance_grid_path,
            dg,
            self.odom_x,
            self.odom_y,
        )
        if deviation_m > self.avoidance_reuse_max_deviation_m:
            return False
        if not self._publish_stored_avoidance_path(dg, stamp, record_history=False):
            return False
        self.avoidance_active = True
        self.avoidance_clear_count = 0
        self.last_avoidance_publish_sec = stamp.to_sec()
        rospy.loginfo_throttle(
            1.0,
            "constrained_local_replanner: keeping previous avoidance path briefly | age=%.2fs dev=%.2fm",
            age_s,
            deviation_m,
        )
        return True

    def _fast_reuse_active_avoidance_path(self, dg, stamp):
        if (
            (not self.avoidance_fast_reuse_enabled)
            or self.avoidance_fast_reuse_window_s <= 0.0
            or (not self.avoidance_active)
            or self.last_avoidance_grid_path is None
            or len(self.last_avoidance_grid_path) < 2
        ):
            return False
        validation_sec = float(self.last_avoidance_validation_sec)
        if validation_sec <= 0.0:
            return False
        now_sec = stamp.to_sec()
        validation_age_s = now_sec - validation_sec
        if validation_age_s < 0.0 or validation_age_s > self.avoidance_fast_reuse_window_s:
            return False
        endpoint_dist_m = self._grid_path_endpoint_distance_to_xy(
            self.last_avoidance_grid_path,
            dg,
            self.odom_x,
            self.odom_y,
        )
        if endpoint_dist_m <= self.avoidance_keep_until_endpoint_distance_m:
            return False
        deviation_m = self._grid_path_min_distance_to_xy(
            self.last_avoidance_grid_path,
            dg,
            self.odom_x,
            self.odom_y,
        )
        deviation_limit_m = max(
            0.45,
            self.avoidance_reuse_max_deviation_m,
            self.robot_length_m * 1.6,
            self.robot_width_m * 2.2,
        )
        if deviation_m > deviation_limit_m:
            return False
        if self._path_blind_zone_turn_conflict(
            self.last_avoidance_grid_path, dg, now_sec=now_sec
        ) is not None:
            return False
        if not self._publish_stored_avoidance_path(dg, stamp, record_history=False):
            return False
        self.avoidance_active = True
        self.avoidance_clear_count = 0
        self.last_avoidance_publish_sec = now_sec
        rospy.loginfo_throttle(
            1.0,
            "constrained_local_replanner: fast-reusing active avoidance path | age=%.2fs dev=%.2fm end=%.2fm",
            validation_age_s,
            deviation_m,
            endpoint_dist_m,
        )
        self._publish_debug_text(
            self._build_debug_text(
                "avoid_fast_reuse",
                stamp,
                trigger_reason=self.last_avoidance_trigger_reason
                if self.last_avoidance_trigger_reason
                else "recent_valid_avoidance",
                path_len=len(self.last_avoidance_grid_path),
            ),
            stamp=stamp,
        )
        return True

    def _continue_active_avoidance_path(
        self,
        dg,
        dynamic_blocked,
        start_cell,
        stamp,
        trigger_key=None,
    ):
        if (not self.avoidance_active) or self.last_avoidance_grid_path is None:
            return False
        if len(self.last_avoidance_grid_path) < 2:
            return False
        same_obstacle_episode = (
            trigger_key is not None
            and self.active_avoidance_obstacle_key is not None
            and trigger_key == self.active_avoidance_obstacle_key
        )
        deviation_limit_m = max(0.25, min(0.55, self.avoidance_reuse_max_deviation_m))
        if same_obstacle_episode:
            deviation_limit_m = max(
                deviation_limit_m,
                self.robot_length_m * 1.40,
                self.robot_width_m * 2.0,
            )
        deviation_m = self._grid_path_min_distance_to_xy(
            self.last_avoidance_grid_path,
            dg,
            self.odom_x,
            self.odom_y,
        )
        if deviation_m > deviation_limit_m:
            return False
        path_still_blocked = self._path_blocked_ahead(
            self.last_avoidance_grid_path,
            dynamic_blocked,
            start_cell,
            float(dg.info.resolution),
            max_check_m=self.lookahead_m,
        )
        if path_still_blocked and (not same_obstacle_episode):
            return False
        if self._path_blind_zone_turn_conflict(
            self.last_avoidance_grid_path, dg, now_sec=stamp.to_sec()
        ) is not None:
            return False
        if not self._publish_stored_avoidance_path(dg, stamp, record_history=False):
            return False
        self.avoidance_clear_count = 0
        self.last_avoidance_publish_sec = stamp.to_sec()
        self.last_avoidance_validation_sec = stamp.to_sec()
        rospy.loginfo_throttle(
            1.0,
            "constrained_local_replanner: reusing active avoidance path | dev=%.2fm cells=%d blocked=%s same_obstacle=%s",
            deviation_m,
            len(self.last_avoidance_grid_path),
            "yes" if path_still_blocked else "no",
            "yes" if same_obstacle_episode else "no",
        )
        return True

    def _reset_pending_avoidance_trigger(self):
        self.pending_avoidance_trigger_key = None
        self.pending_avoidance_trigger_count = 0
        self.pending_avoidance_trigger_stamp_sec = 0.0

    @staticmethod
    def _avoidance_trigger_keys_compatible(new_key, pending_key):
        if new_key == pending_key:
            return True
        if new_key is None or pending_key is None:
            return False
        try:
            new_reason = str(new_key[0])
            pending_reason = str(pending_key[0])
        except (IndexError, TypeError):
            return False
        # Static memory ids and raw obstacle centroids can churn while the robot
        # approaches the same front obstacle.  For the debounce stage, treat
        # repeated overlap evidence as the same episode so we can actually enter
        # avoidance instead of staying in avoid_pending forever.
        return new_reason == pending_reason and new_reason in (
            "overlap",
            "map_filtered_path_overlap",
            "blind_zone_turn_conflict",
        )

    def _make_avoidance_trigger_key(
        self,
        trigger_reason,
        blocking_cells,
        blocking_points,
        blind_zone_conflict=None,
    ):
        normalized_reason = str(trigger_reason).strip().lower()
        if normalized_reason in ("predicted_overlap", "dynamic_points_overlap"):
            normalized_reason = "overlap"

        def _bucket(v):
            return round(float(v) * 2.5) / 2.5

        if blind_zone_conflict is not None:
            return (
                normalized_reason,
                _bucket(blind_zone_conflict.get("x", 0.0)),
                _bucket(blind_zone_conflict.get("y", 0.0)),
            )
        obstacle_key, _stable_points, _matched_entries = (
            self._stable_avoidance_obstacle_points(
                blocking_cells_world=blocking_cells,
                blocking_points_world=blocking_points,
            )
        )
        if obstacle_key is not None:
            return (normalized_reason, obstacle_key)
        pts = blocking_points if blocking_points else blocking_cells
        if pts:
            sample = pts[: min(6, len(pts))]
            cx = sum(float(p[0]) for p in sample) / float(len(sample))
            cy = sum(float(p[1]) for p in sample) / float(len(sample))
            return (normalized_reason, _bucket(cx), _bucket(cy))
        return (normalized_reason,)

    def _avoidance_trigger_confirmed(self, trigger_key, stamp):
        if trigger_key is None:
            self._reset_pending_avoidance_trigger()
            return False
        if self.avoidance_active or self.avoidance_trigger_confirm_cycles <= 1:
            self.pending_avoidance_trigger_key = trigger_key
            self.pending_avoidance_trigger_count = self.avoidance_trigger_confirm_cycles
            self.pending_avoidance_trigger_stamp_sec = stamp.to_sec()
            return True

        now_sec = stamp.to_sec()
        if (
            self.pending_avoidance_trigger_stamp_sec > 0.0
            and self.avoidance_trigger_confirm_max_gap_s > 0.0
            and (now_sec - self.pending_avoidance_trigger_stamp_sec)
            > self.avoidance_trigger_confirm_max_gap_s
        ):
            self._reset_pending_avoidance_trigger()

        if self._avoidance_trigger_keys_compatible(
            trigger_key, self.pending_avoidance_trigger_key
        ):
            self.pending_avoidance_trigger_count += 1
        else:
            self.pending_avoidance_trigger_key = trigger_key
            self.pending_avoidance_trigger_count = 1
        self.pending_avoidance_trigger_stamp_sec = now_sec
        return self.pending_avoidance_trigger_count >= self.avoidance_trigger_confirm_cycles

    def _debug_avoidance_log(self, message):
        if not self.debug_avoidance_logging:
            return
        rospy.loginfo_throttle(self.debug_avoidance_log_period_s, message)

    def _build_debug_text(self, state, stamp, **kwargs):
        wait_s = max(0.0, float(kwargs.get("wait_s", 0.0)))
        path_len = int(kwargs.get("path_len", 0))
        trigger_reason = str(kwargs.get("trigger_reason", "clear") or "clear")
        avoid_direction = str(kwargs.get("avoid_direction", "none") or "none")
        overlay_points = int(kwargs.get("overlay_points", 0))
        return (
            "local_replanner state={} reason={} avoid={} dir={} wait={}/{} "
            "path_len={} raw_pts={} clustered={} map_filtered={} memory_pts={} locked_static={} tracked_objs={} tracked_pts={} tracked_mem={} overlay_pts={} blocked_since={}"
        ).format(
            state,
            trigger_reason,
            "on" if self.avoidance_active else "off",
            avoid_direction,
            self._fmt_debug_float(wait_s),
            self._fmt_debug_float(self.blocked_stop_before_avoidance_s),
            path_len,
            int(self.obstacle_raw_point_count),
            int(self.obstacle_cluster_count),
            int(self.known_map_filtered_count),
            int(self.obstacle_memory_count),
            int(self.obstacle_memory_locked_count),
            int(self.tracked_object_count),
            len(self.current_tracked_object_points_map),
            int(self.tracked_object_memory_count),
            overlay_points,
            self._fmt_debug_float(self.local_blocked_since_sec),
        )

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
        ttc_s=-1.0,
        tracked_object_id=-1,
        tracked_object_label="",
        summary_text="",
    ):
        msg = ExplainabilityEvent()
        msg.header.stamp = stamp if stamp is not None else rospy.Time.now()
        msg.source_node = "constrained_local_replanner"
        msg.event_type = str(event_type)
        msg.decision_layer = "local_replanner"
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
        msg.ttc_s = float(ttc_s)
        msg.tracked_object_id = int(tracked_object_id)
        msg.tracked_object_label = str(tracked_object_label)
        msg.summary_text = str(summary_text)

        key = (
            msg.event_type,
            msg.trigger_reason,
            msg.action_taken,
            msg.avoid_direction,
            msg.local_planning_active,
            msg.stop_commanded,
            msg.slowdown_commanded,
            round(float(msg.obstacle_lateral_offset_m), 2),
            msg.summary_text,
        )
        if key == self._last_explain_key:
            return
        stamp_sec = msg.header.stamp.to_sec() if msg.header.stamp.to_sec() > 0.0 else rospy.get_time()
        self._last_explain_key = key
        self._last_explain_time = stamp_sec
        self.pub_explainability.publish(msg)

    def _infer_avoid_direction(self, dg, avoid_path):
        if avoid_path is None or len(avoid_path) < 3:
            return "none", 0.0

        sx, sy = self._grid_to_world(dg, avoid_path[0][0], avoid_path[0][1])
        ex, ey = self._grid_to_world(dg, avoid_path[-1][0], avoid_path[-1][1])
        dx = float(ex) - float(sx)
        dy = float(ey) - float(sy)
        norm = math.hypot(dx, dy)
        if norm <= 1e-6:
            return "none", 0.0

        best_signed_offset = 0.0
        for gx, gy in avoid_path[1:-1]:
            px, py = self._grid_to_world(dg, gx, gy)
            signed = (dx * (float(py) - float(sy)) - dy * (float(px) - float(sx))) / norm
            if abs(signed) > abs(best_signed_offset):
                best_signed_offset = signed

        if abs(best_signed_offset) < 0.05:
            return "none", best_signed_offset
        return ("left" if best_signed_offset > 0.0 else "right"), best_signed_offset

    def _nearest_path_cell_index(self, path, cell):
        best_i = 0
        best_d2 = float("inf")
        cx, cy = cell
        for i, (px, py) in enumerate(path):
            d2 = float((px - cx) * (px - cx) + (py - cy) * (py - cy))
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        return best_i

    def _overlay_pointcloud_obstacles(
        self, blocked, dg, keep_cells=None, enabled=True, margin_m=None, points_map=None
    ):
        obstacle_points = self.obstacle_points_map if points_map is None else points_map
        if (not enabled) or (not obstacle_points):
            return [row[:] for row in blocked], 0

        res = max(1e-3, float(dg.info.resolution))
        if margin_m is None:
            margin_m = self.obstacle_block_margin_m
        inflate_m = self._pointcloud_corridor_half_width_m(margin_m)
        inflate_cells = max(1, int(math.ceil(inflate_m / res)))
        out = [row[:] for row in blocked]
        keep = set(keep_cells or [])
        marked_sources = 0

        for wx, wy in obstacle_points:
            gx, gy = self._world_to_grid(dg, wx, wy)
            if not self._in_bounds_blocked(out, gx, gy):
                continue
            marked_sources += 1
            for dx in range(-inflate_cells, inflate_cells + 1):
                for dy in range(-inflate_cells, inflate_cells + 1):
                    if (dx * dx + dy * dy) > (inflate_cells * inflate_cells):
                        continue
                    nx = gx + dx
                    ny = gy + dy
                    if not self._in_bounds_blocked(out, nx, ny):
                        continue
                    if (nx, ny) in keep:
                        continue
                    out[ny][nx] = True
        return out, marked_sources

    def _overlay_dynamic_obstacles(
        self,
        blocked,
        dg,
        keep_cells=None,
        include_tracked=None,
        path=None,
        start_cell=None,
        max_check_m=None,
    ):
        # Dynamic avoidance should react to live returns plus short-lived tracked-object memory.
        if include_tracked is None:
            include_tracked = (
                self.tracked_object_virtual_obstacles_enabled
                and self.tracked_object_avoidance_enabled
            )
        dynamic_points = self._avoidance_trigger_points_map(
            include_tracked=include_tracked
        )
        if path is not None and start_cell is not None:
            dynamic_points = self._filter_path_relevant_obstacle_points(
                path,
                dg,
                start_cell,
                dynamic_points,
                max_check_m=max_check_m,
            )
        return self._overlay_pointcloud_obstacles(
            blocked,
            dg,
            keep_cells=keep_cells,
            enabled=(
                self.use_pointcloud_avoidance_trigger
                or self.use_map_filtered_path_obstacle_trigger
                or (
                    self.tracked_object_virtual_obstacles_enabled
                    and include_tracked
                )
            ),
            margin_m=self.obstacle_block_margin_m,
            points_map=dynamic_points,
        )

    def _path_forward_unit_vector(self, path, dg, start_cell):
        if not path:
            return None

        start_idx = self._nearest_path_cell_index(path, start_cell)
        if start_idx >= len(path):
            return None

        prev_x = float(self.odom_x)
        prev_y = float(self.odom_y)
        for idx in range(start_idx, len(path)):
            wx, wy = self._grid_to_world(dg, path[idx][0], path[idx][1])
            dx = float(wx) - prev_x
            dy = float(wy) - prev_y
            seg_len = math.hypot(dx, dy)
            if seg_len > 1e-6:
                return dx / seg_len, dy / seg_len
            prev_x = float(wx)
            prev_y = float(wy)

        return math.cos(self.odom_yaw), math.sin(self.odom_yaw)

    def _filter_path_relevant_obstacle_points(
        self,
        path,
        dg,
        start_cell,
        points_map,
        max_check_m=None,
    ):
        if (not points_map) or (not self.forward_path_obstacle_filter_enabled):
            return list(points_map or [])

        forward = self._path_forward_unit_vector(path, dg, start_cell)
        if forward is None:
            return list(points_map)
        ux, uy = forward

        local_forward_x, _local_forward_y = self._world_to_local(
            self.odom_x + ux,
            self.odom_y + uy,
        )
        if local_forward_x <= 0.1:
            return list(points_map)

        far_limit_m = None
        if max_check_m is not None:
            far_limit_m = (
                float(max_check_m)
                + self.path_blocking_radius_m
                + self.obstacle_block_margin_m
                + self.avoidance_trigger_margin_m
            )

        filtered = []
        for wx, wy in points_map:
            progress_m = (
                (float(wx) - self.odom_x) * ux + (float(wy) - self.odom_y) * uy
            )
            if progress_m < (-self.forward_path_obstacle_rear_tolerance_m):
                continue
            if far_limit_m is not None and progress_m > far_limit_m:
                continue
            filtered.append((float(wx), float(wy)))
        if filtered:
            return filtered

        # Fallback: if the forward projection is slightly misaligned with the
        # current path heading, do not drop nearby obstacle evidence entirely.
        # Keep close-by points with a looser rear tolerance so the replanner can
        # still spawn a local detour instead of missing the obstacle outright.
        fallback = []
        rear_allow_m = max(0.25, self.forward_path_obstacle_rear_tolerance_m * 4.0)
        for wx, wy in points_map:
            progress_m = (
                (float(wx) - self.odom_x) * ux + (float(wy) - self.odom_y) * uy
            )
            if progress_m < (-rear_allow_m):
                continue
            if far_limit_m is not None:
                dist_m = math.hypot(float(wx) - self.odom_x, float(wy) - self.odom_y)
                if dist_m > far_limit_m:
                    continue
            fallback.append((float(wx), float(wy)))
        return fallback

    def _path_blocked_ahead(self, path, blocked, start_cell, grid_resolution_m, max_check_m=None):
        if not path:
            return False

        start_idx = self._nearest_path_cell_index(path, start_cell)
        if start_idx >= len(path):
            return False

        remain_m = 0.0
        blocked_run = 0
        for i in range(start_idx, len(path)):
            gx, gy = path[i]
            cell_blocked = (not self._in_bounds_blocked(blocked, gx, gy)) or blocked[gy][gx]
            seg_blocked = i + 1 < len(path) and (not self._has_line_of_sight(blocked, path[i], path[i + 1]))
            if cell_blocked or seg_blocked:
                blocked_run += 1
                if blocked_run >= self.risk_block_confirm_cells:
                    return True
            else:
                blocked_run = 0
            if i + 1 < len(path):
                remain_m += self._heur(path[i], path[i + 1]) * max(1e-3, grid_resolution_m)
                if max_check_m is not None and remain_m >= max_check_m:
                    break
        return False

    @staticmethod
    def _point_to_segment_distance_sq(px, py, x0, y0, x1, y1):
        vx = x1 - x0
        vy = y1 - y0
        seg_len_sq = vx * vx + vy * vy
        if seg_len_sq <= 1e-9:
            dx = px - x0
            dy = py - y0
            return dx * dx + dy * dy
        t = ((px - x0) * vx + (py - y0) * vy) / seg_len_sq
        t = max(0.0, min(1.0, t))
        proj_x = x0 + t * vx
        proj_y = y0 + t * vy
        dx = px - proj_x
        dy = py - proj_y
        return dx * dx + dy * dy

    def _path_blocked_by_obstacles(
        self,
        path,
        dg,
        start_cell,
        points_map=None,
        max_check_m=None,
    ):
        obstacle_points = self.obstacle_points_map if points_map is None else points_map
        if not path or not obstacle_points:
            return False

        obstacle_points = self._filter_path_relevant_obstacle_points(
            path,
            dg,
            start_cell,
            obstacle_points,
            max_check_m=max_check_m,
        )
        if not obstacle_points:
            return False

        start_idx = self._nearest_path_cell_index(path, start_cell)
        if start_idx >= len(path) - 1:
            return False

        world_path = [self._grid_to_world(dg, gx, gy) for gx, gy in path]
        corridor_half = self._pointcloud_corridor_half_width_m(
            self.obstacle_block_margin_m + self.avoidance_trigger_margin_m
        )
        corridor_half_sq = corridor_half * corridor_half
        remain_m = 0.0
        hit_indices = set()

        for seg_idx in range(start_idx, len(world_path) - 1):
            x0, y0 = world_path[seg_idx]
            x1, y1 = world_path[seg_idx + 1]
            seg_len = math.hypot(x1 - x0, y1 - y0)
            if seg_len <= 1e-6:
                continue
            remain_m += seg_len
            for obs_idx, (ox, oy) in enumerate(obstacle_points):
                if obs_idx in hit_indices:
                    continue
                if self._point_to_segment_distance_sq(ox, oy, x0, y0, x1, y1) <= corridor_half_sq:
                    hit_indices.add(obs_idx)
                    if len(hit_indices) >= self.pointcloud_block_confirm_points:
                        return True
            if max_check_m is not None and remain_m >= max_check_m:
                break
            if max_check_m is None and remain_m >= self.avoidance_trigger_ahead_m:
                break
        return False

    def _collect_path_overlap_points(
        self,
        path,
        dg,
        start_cell,
        points_map,
        corridor_margin_m,
        max_check_m=None,
    ):
        if not path or not points_map:
            return []

        points_map = self._filter_path_relevant_obstacle_points(
            path,
            dg,
            start_cell,
            points_map,
            max_check_m=max_check_m,
        )
        if not points_map:
            return []

        start_idx = self._nearest_path_cell_index(path, start_cell)
        if start_idx >= len(path) - 1:
            return []

        world_path = [self._grid_to_world(dg, gx, gy) for gx, gy in path]
        corridor_half = self._pointcloud_corridor_half_width_m(corridor_margin_m)
        corridor_half_sq = corridor_half * corridor_half
        remain_m = 0.0
        hit_indices = set()
        hits = []

        for seg_idx in range(start_idx, len(world_path) - 1):
            x0, y0 = world_path[seg_idx]
            x1, y1 = world_path[seg_idx + 1]
            seg_len = math.hypot(x1 - x0, y1 - y0)
            if seg_len <= 1e-6:
                continue
            remain_m += seg_len
            for obs_idx, (ox, oy) in enumerate(points_map):
                if obs_idx in hit_indices:
                    continue
                if self._point_to_segment_distance_sq(ox, oy, x0, y0, x1, y1) <= corridor_half_sq:
                    hit_indices.add(obs_idx)
                    hits.append((float(ox), float(oy)))
            if max_check_m is not None and remain_m >= max_check_m:
                break
        return hits

    def _collect_confirmed_blocked_path_world_points(
        self,
        path,
        blocked,
        start_cell,
        dg,
        max_check_m=None,
    ):
        if not path:
            return []

        start_idx = self._nearest_path_cell_index(path, start_cell)
        if start_idx >= len(path):
            return []

        remain_m = 0.0
        blocked_run = 0
        run_cells = []
        run_keys = set()

        for i in range(start_idx, len(path)):
            gx, gy = path[i]
            cell_blocked = (not self._in_bounds_blocked(blocked, gx, gy)) or blocked[gy][gx]
            seg_blocked = i + 1 < len(path) and (not self._has_line_of_sight(blocked, path[i], path[i + 1]))

            if cell_blocked or seg_blocked:
                if blocked_run == 0:
                    run_cells = []
                    run_keys = set()
                blocked_run += 1

                candidate_cells = []
                if cell_blocked:
                    candidate_cells.append((gx, gy))
                if seg_blocked and i + 1 < len(path):
                    candidate_cells.append(path[i + 1])

                for cx, cy in candidate_cells:
                    if (cx, cy) in run_keys:
                        continue
                    run_keys.add((cx, cy))
                    run_cells.append(self._grid_to_world(dg, cx, cy))

                if blocked_run >= self.risk_block_confirm_cells:
                    return run_cells
            else:
                blocked_run = 0
                run_cells = []
                run_keys = set()

            if i + 1 < len(path):
                remain_m += self._heur(path[i], path[i + 1]) * max(
                    1e-3, float(dg.info.resolution)
                )
                if max_check_m is not None and remain_m >= max_check_m:
                    break
        return []

    def _inflate_source_mask(self, dg, source_mask, radius_override_m=None):
        w = len(source_mask[0]) if source_mask else 0
        h = len(source_mask)
        if w <= 0 or h <= 0:
            return []

        res = float(dg.info.resolution)
        inflate_m = max(
            0.05,
            self.path_blocking_radius_m
            if radius_override_m is None
            else float(radius_override_m),
        )
        inflate_cells = max(1, int(math.ceil(inflate_m / max(1e-3, res))))
        out = [[False for _ in range(w)] for _ in range(h)]
        for y in range(h):
            for x in range(w):
                if not source_mask[y][x]:
                    continue
                for dx in range(-inflate_cells, inflate_cells + 1):
                    for dy in range(-inflate_cells, inflate_cells + 1):
                        if math.hypot(float(dx) * res, float(dy) * res) > inflate_m:
                            continue
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            out[ny][nx] = True
        return out

    def _build_base_source_blocked_grids(self, dg, rg=None, radius_override_m=None):
        w = int(dg.info.width)
        h = int(dg.info.height)
        dims_match = (
            rg is not None
            and int(rg.info.width) == w
            and int(rg.info.height) == h
        )

        drivable_mask = [[False for _ in range(w)] for _ in range(h)]
        risk_mask = [[False for _ in range(w)] for _ in range(h)] if dims_match else None
        for y in range(h):
            row = y * w
            for x in range(w):
                idx = row + x
                drivable_mask[y][x] = int(dg.data[idx]) != 0
                if dims_match:
                    risk_mask[y][x] = int(rg.data[idx]) >= self.risk_threshold

        return {
            "grid_occ": self._inflate_source_mask(
                dg, drivable_mask, radius_override_m=radius_override_m
            ),
            "risk": self._inflate_source_mask(
                dg, risk_mask, radius_override_m=radius_override_m
            )
            if risk_mask is not None
            else None,
        }

    def _build_path_blocker_source_summary(
        self,
        path,
        dg,
        start_cell,
        *,
        max_check_m,
        point_margin_m,
        rg=None,
        radius_override_m=None,
        blind_zone_conflict=None,
    ):
        summary = {
            "grid_occ": 0,
            "risk": 0,
            "pc_current": 0,
            "map_filtered_path": 0,
            "pc_memory": 0,
            "tracked_current": 0,
            "tracked_memory": 0,
            "blind_zone": "none",
        }
        if blind_zone_conflict is not None:
            summary["blind_zone"] = (
                "left" if int(blind_zone_conflict.get("side", 0)) > 0 else "right"
            )
        if not path or dg is None:
            return summary

        source_blocked = self._build_base_source_blocked_grids(
            dg,
            rg=rg,
            radius_override_m=radius_override_m,
        )
        summary["grid_occ"] = len(
            self._collect_confirmed_blocked_path_world_points(
                path,
                source_blocked["grid_occ"],
                start_cell,
                dg,
                max_check_m=max_check_m,
            )
        )
        if source_blocked["risk"] is not None:
            summary["risk"] = len(
                self._collect_confirmed_blocked_path_world_points(
                    path,
                    source_blocked["risk"],
                    start_cell,
                    dg,
                    max_check_m=max_check_m,
                )
            )

        summary["pc_current"] = len(
            self._collect_path_overlap_points(
                path,
                dg,
                start_cell,
                self._effective_current_obstacle_points_map(),
                point_margin_m,
                max_check_m=max_check_m,
            )
        )
        summary["map_filtered_path"] = len(
            self._collect_path_overlap_points(
                path,
                dg,
                start_cell,
                self.known_map_filtered_points_map,
                self.map_filtered_path_trigger_margin_m,
                max_check_m=max_check_m,
            )
        )
        summary["pc_memory"] = len(
            self._collect_path_overlap_points(
                path,
                dg,
                start_cell,
                self._memory_points_from_entries(
                    self.obstacle_memory_points, confirmed_only=True
                ),
                point_margin_m,
                max_check_m=max_check_m,
            )
        )
        summary["tracked_current"] = len(
            self._collect_path_overlap_points(
                path,
                dg,
                start_cell,
                self.current_tracked_object_points_map,
                point_margin_m,
                max_check_m=max_check_m,
            )
        )
        summary["tracked_memory"] = len(
            self._collect_path_overlap_points(
                path,
                dg,
                start_cell,
                self._memory_points_from_entries(self.tracked_object_memory_points),
                point_margin_m,
                max_check_m=max_check_m,
            )
        )
        return summary

    def _log_blocker_source_summary(self, context, base_label, trigger_reason, summary):
        self._debug_avoidance_log(
            "constrained_local_replanner: blocker_sources | context={} base={} reason={} grid_occ={} risk={} pc_current={} map_filtered_path={} pc_memory={} tracked_current={} tracked_memory={} blind_zone={}".format(
                context,
                base_label,
                trigger_reason,
                int(summary.get("grid_occ", 0)),
                int(summary.get("risk", 0)),
                int(summary.get("pc_current", 0)),
                int(summary.get("map_filtered_path", 0)),
                int(summary.get("pc_memory", 0)),
                int(summary.get("tracked_current", 0)),
                int(summary.get("tracked_memory", 0)),
                str(summary.get("blind_zone", "none")),
            )
        )

    def _first_blocked_path_index(
        self,
        path,
        blocked,
        start_cell,
        dg,
        max_check_m=None,
        include_pointcloud=False,
        points_map=None,
        point_margin_m=None,
    ):
        if not path:
            return None

        start_idx = self._nearest_path_cell_index(path, start_cell)
        remain_m = 0.0
        blocked_run = 0
        first_blocked_i = None
        for i in range(start_idx, len(path)):
            gx, gy = path[i]
            cell_blocked = (not self._in_bounds_blocked(blocked, gx, gy)) or blocked[gy][gx]
            seg_blocked = i + 1 < len(path) and (not self._has_line_of_sight(blocked, path[i], path[i + 1]))
            if cell_blocked or seg_blocked:
                if blocked_run == 0:
                    first_blocked_i = i if cell_blocked else min(i + 1, len(path) - 1)
                blocked_run += 1
                if blocked_run >= self.risk_block_confirm_cells:
                    return first_blocked_i
            else:
                blocked_run = 0
                first_blocked_i = None
            if i + 1 < len(path):
                remain_m += self._heur(path[i], path[i + 1]) * max(1e-3, float(dg.info.resolution))
                if max_check_m is not None and remain_m >= max_check_m:
                    return None

        if not include_pointcloud:
            return None

        point_source = self.obstacle_points_map if points_map is None else points_map
        if not point_source:
            return None

        point_source = self._filter_path_relevant_obstacle_points(
            path,
            dg,
            start_cell,
            point_source,
            max_check_m=max_check_m,
        )
        if not point_source:
            return None

        world_path = [self._grid_to_world(dg, gx, gy) for gx, gy in path]
        corridor_half = self._pointcloud_corridor_half_width_m(
            self.obstacle_block_margin_m + self.avoidance_trigger_margin_m
            if point_margin_m is None
            else point_margin_m
        )
        corridor_half_sq = corridor_half * corridor_half
        remain_m = 0.0
        hit_indices = set()
        first_hit_i = None

        for seg_idx in range(start_idx, len(world_path) - 1):
            x0, y0 = world_path[seg_idx]
            x1, y1 = world_path[seg_idx + 1]
            seg_len = math.hypot(x1 - x0, y1 - y0)
            if seg_len <= 1e-6:
                continue
            remain_m += seg_len
            for obs_idx, (ox, oy) in enumerate(point_source):
                if obs_idx in hit_indices:
                    continue
                if self._point_to_segment_distance_sq(ox, oy, x0, y0, x1, y1) <= corridor_half_sq:
                    hit_indices.add(obs_idx)
                    if first_hit_i is None:
                        first_hit_i = min(seg_idx + 1, len(path) - 1)
                    if len(hit_indices) >= self.pointcloud_block_confirm_points:
                        return first_hit_i
            if max_check_m is not None and remain_m >= max_check_m:
                break
        return None

    @staticmethod
    def _append_path_segment(out_path, segment):
        for cell in segment:
            if not out_path or out_path[-1] != cell:
                out_path.append(cell)

    def _path_remaining_distance_m(self, path, dg, start_idx):
        if not path or start_idx >= len(path) - 1:
            return 0.0
        remain_m = 0.0
        scale = max(1e-3, float(dg.info.resolution))
        for idx in range(max(0, start_idx), len(path) - 1):
            remain_m += self._heur(path[idx], path[idx + 1]) * scale
        return remain_m

    def _get_active_local_blind_zone_memory(self, now_sec=None):
        if (
            (not self.local_blind_zone_guard_enabled)
            or (not self.static_obstacle_memory_enabled)
            or self.local_blind_zone_guard_radius_m <= 0.0
            or self.local_blind_zone_guard_ttl_s <= 0.0
            or (not self.obstacle_memory_points)
        ):
            return None
        if now_sec is None:
            now_sec = rospy.get_time()

        lateral_limit = (
            self.robot_half_width
            + self.local_blind_zone_guard_side_margin_m
        )
        best = None
        best_score = None
        for entry in self.obstacle_memory_points:
            wx, wy = self._memory_entry_position(entry)
            seen_sec = self._memory_entry_last_seen(entry)
            age_s = float(now_sec - seen_sec)
            if age_s < 0.0 or age_s > self.local_blind_zone_guard_ttl_s:
                continue
            lx, ly = self._world_to_local(wx, wy)
            if lx <= (-self.robot_half_length):
                continue
            range_m = math.hypot(lx, ly)
            if range_m > self.local_blind_zone_guard_radius_m:
                continue
            if abs(ly) <= lateral_limit:
                continue
            if abs(ly) > self.local_blind_zone_guard_side_lateral_limit_m:
                continue
            side = 1 if ly > 0.0 else -1
            if side != 0 and lx <= 0.0:
                continue
            score = (range_m, age_s)
            if best is None or score < best_score:
                best = {
                    "world_x": float(wx),
                    "world_y": float(wy),
                    "x": float(lx),
                    "y": float(ly),
                    "range_m": float(range_m),
                    "age_s": float(age_s),
                    "side": int(side),
                }
                best_score = score
        return best

    def _path_blind_zone_turn_conflict(self, path, dg, now_sec=None):
        memory = self._get_active_local_blind_zone_memory(now_sec)
        if memory is None or not path:
            return None

        lookahead_m = max(0.05, self.local_blind_zone_guard_lookahead_m)
        prev_x = self.odom_x
        prev_y = self.odom_y
        travelled_m = 0.0
        target_world = None

        for gx, gy in path:
            wx, wy = self._grid_to_world(dg, gx, gy)
            seg_len = math.hypot(wx - prev_x, wy - prev_y)
            if seg_len <= 1e-6:
                prev_x = wx
                prev_y = wy
                continue
            next_travelled_m = travelled_m + seg_len
            if next_travelled_m >= lookahead_m:
                ratio = (lookahead_m - travelled_m) / seg_len
                ratio = max(0.0, min(1.0, ratio))
                target_world = (
                    prev_x + ratio * (wx - prev_x),
                    prev_y + ratio * (wy - prev_y),
                )
                travelled_m = lookahead_m
                break
            travelled_m = next_travelled_m
            prev_x = wx
            prev_y = wy
            target_world = (wx, wy)

        if target_world is None:
            return None

        target_x, target_y = self._world_to_local(target_world[0], target_world[1])
        if target_x <= 0.05:
            return None

        heading_rad = math.atan2(target_y, target_x)
        heading_deadband_rad = math.radians(
            self.local_blind_zone_guard_heading_deadband_deg
        )
        if abs(heading_rad) < heading_deadband_rad:
            return None

        turn_side = 1 if heading_rad > 0.0 else -1
        if turn_side != int(memory["side"]):
            return None

        out = dict(memory)
        out.update(
            {
                "path_target_x": float(target_x),
                "path_target_y": float(target_y),
                "path_heading_deg": float(math.degrees(heading_rad)),
                "lookahead_m": float(travelled_m),
            }
        )
        return out

    def _had_recent_avoidance_activity(self, now_sec=None):
        if self.avoidance_active:
            return True
        if self.near_goal_block_ignore_after_avoidance_s <= 0.0:
            return False
        last_active_sec = float(self.last_avoidance_active_sec)
        if last_active_sec <= 0.0:
            return False
        if now_sec is None:
            now_sec = rospy.get_time()
        return now_sec > 0.0 and (
            now_sec - last_active_sec
        ) <= self.near_goal_block_ignore_after_avoidance_s

    def _can_use_near_goal_shortcut(self, remaining_to_goal_m, now_sec=None):
        if remaining_to_goal_m > self.near_goal_block_ignore_distance_m:
            return False
        if not self._had_recent_avoidance_activity(now_sec):
            return True
        if self.avoidance_active:
            return False
        return remaining_to_goal_m <= self.near_goal_recent_avoidance_release_distance_m

    def _path_segment_blocked(self, path, blocked, start_idx):
        if not path or start_idx >= len(path):
            return False
        blocked_run = 0
        for i in range(max(0, start_idx), len(path)):
            gx, gy = path[i]
            cell_blocked = (not self._in_bounds_blocked(blocked, gx, gy)) or blocked[gy][gx]
            seg_blocked = i + 1 < len(path) and (not self._has_line_of_sight(blocked, path[i], path[i + 1]))
            if cell_blocked or seg_blocked:
                blocked_run += 1
                if blocked_run >= self.risk_block_confirm_cells:
                    return True
            else:
                blocked_run = 0
        return False

    def _should_ignore_near_goal_block(self, path, blocked_idx, start_cell, dg, relaxed_blocked=None):
        if blocked_idx is None or len(path) < 2:
            return False
        if self.near_goal_block_ignore_distance_m <= 0.0:
            return False

        start_idx = self._nearest_path_cell_index(path, start_cell)
        if start_idx >= len(path):
            return False
        blocked_idx = max(start_idx, min(blocked_idx, len(path) - 1))
        remaining_to_goal_m = self._path_remaining_distance_m(path, dg, start_idx)
        blocked_tail_m = self._path_remaining_distance_m(path, dg, blocked_idx)
        if not self._can_use_near_goal_shortcut(remaining_to_goal_m):
            return False
        if blocked_tail_m > self.near_goal_tail_block_ignore_distance_m:
            return False
        if relaxed_blocked is None:
            return True
        return not self._path_segment_blocked(path, relaxed_blocked, blocked_idx)

    @staticmethod
    def _is_grid_only_blocker_source_summary(summary):
        if not summary:
            return False
        if int(summary.get("grid_occ", 0)) <= 0:
            return False
        if str(summary.get("blind_zone", "none")) != "none":
            return False
        return (
            int(summary.get("risk", 0)) <= 0
            and int(summary.get("pc_current", 0)) <= 0
            and int(summary.get("map_filtered_path", 0)) <= 0
            and int(summary.get("pc_memory", 0)) <= 0
            and int(summary.get("tracked_current", 0)) <= 0
            and int(summary.get("tracked_memory", 0)) <= 0
        )

    def _grid_only_nominal_fallback_allowed(self, summary):
        if not self._is_grid_only_blocker_source_summary(summary):
            return False
        if self.enable_avoidance_path and self.grid_only_avoidance_search_enabled:
            return False
        if self.allow_grid_only_nominal_fallback:
            return True
        max_cells = int(self.grid_only_nominal_fallback_max_cells)
        return max_cells > 0 and int(summary.get("grid_occ", 0)) <= max_cells

    def _should_ignore_grid_only_nominal_block(
        self,
        path,
        start_cell,
        dg,
        rg,
        source_summary,
    ):
        if not self.grid_only_relaxed_path_blocking_enabled:
            return False
        if not self._is_grid_only_blocker_source_summary(source_summary):
            return False
        if (
            self.grid_only_relaxed_path_blocking_radius_m + 1e-6
            >= self.path_blocking_radius_m
        ):
            return False

        relaxed_blocked = self._inflate_blocked(
            dg,
            rg,
            radius_override_m=self.grid_only_relaxed_path_blocking_radius_m,
        )
        relaxed_idx = self._first_blocked_path_index(
            path,
            relaxed_blocked,
            start_cell,
            dg,
            max_check_m=self.lookahead_m,
            include_pointcloud=False,
        )
        return relaxed_idx is None

    def _build_branch_avoidance_path(
        self,
        nominal_path,
        dynamic_blocked,
        start_cell,
        dg,
        now_sec=None,
        points_map=None,
        blocking_cells_world=None,
        blocking_points_world=None,
        preferred_direction=None,
    ):
        if len(nominal_path) < 2:
            return None, None

        preferred_direction = (
            str(preferred_direction).strip().lower()
            if preferred_direction is not None
            else None
        )
        if preferred_direction not in ("left", "right"):
            preferred_direction = None

        start_idx = self._nearest_path_cell_index(nominal_path, start_cell)
        blocked_idx = self._first_blocked_path_index(
            nominal_path,
            dynamic_blocked,
            start_cell,
            dg,
            max_check_m=self.avoidance_trigger_ahead_m,
            include_pointcloud=self.use_pointcloud_avoidance_trigger or bool(points_map),
            points_map=points_map,
            point_margin_m=self.obstacle_block_margin_m + self.avoidance_trigger_margin_m,
        )
        if blocked_idx is None:
            return None, None

        res_m = max(1e-3, float(dg.info.resolution))
        blocked_delta_cells = max(0, blocked_idx - start_idx)
        close_blocked_cells = max(3, int(math.ceil(1.0 / res_m)))
        backtrack_cells = self.avoidance_branch_backtrack_cells
        clip_to_rejoin_only = self._use_global_nominal_reference()
        orthogonal_detour = False
        if blocked_delta_cells <= close_blocked_cells:
            backtrack_cells = max(backtrack_cells, int(math.ceil(0.8 / res_m)))

        branch_start_idx = max(start_idx, blocked_idx - backtrack_cells)
        while branch_start_idx > start_idx:
            bx, by = nominal_path[branch_start_idx]
            if self._in_bounds_blocked(dynamic_blocked, bx, by) and not dynamic_blocked[by][bx]:
                break
            branch_start_idx -= 1

        branch_start = nominal_path[branch_start_idx]

        def _compose_detour_path(detour_cells, rejoin_idx=None):
            composed = []
            if clip_to_rejoin_only or rejoin_idx is None:
                # In global-reference mode, or when we only have a best-effort
                # partial branch, publish only the temporary branch segment.
                prefix = self._collapse_straight_grid_runs(
                    nominal_path[start_idx:branch_start_idx + 1]
                )
                self._append_path_segment(composed, [start_cell])
                self._append_path_segment(composed, prefix)
                self._append_path_segment(composed, detour_cells[1:])
            else:
                self._append_path_segment(composed, [start_cell])
                self._append_path_segment(
                    composed, nominal_path[start_idx:branch_start_idx + 1]
                )
                self._append_path_segment(composed, detour_cells[1:])
                self._append_path_segment(composed, nominal_path[rejoin_idx + 1:])
            return composed

        rejoin_distance_candidates_m = [self.avoidance_rejoin_min_distance_m]
        if self.avoidance_rejoin_min_distance_m > 0.8:
            rejoin_distance_candidates_m.append(
                max(0.8, self.avoidance_rejoin_min_distance_m * 0.75)
            )
        if blocked_delta_cells <= close_blocked_cells and self.avoidance_rejoin_min_distance_m > 1.0:
            rejoin_distance_candidates_m.append(
                max(0.8, self.avoidance_rejoin_min_distance_m * 0.5)
            )

        seen_rejoin_distances = set()
        ordered_rejoin_distances = []
        for candidate_m in rejoin_distance_candidates_m:
            key = round(float(candidate_m), 3)
            if key in seen_rejoin_distances:
                continue
            seen_rejoin_distances.add(key)
            ordered_rejoin_distances.append(float(candidate_m))

        # Try the normal rejoin spacing first, then relax it for close blockers so
        # the robot can still slip around obstacles that appear only a short
        # distance ahead instead of dropping straight into hold-stop.
        exact_fallback_candidate = None
        exact_fallback_history = None
        best_effort_candidate = None
        best_effort_score = None
        for rejoin_distance_m in ordered_rejoin_distances:
            min_rejoin_cells = max(1, int(math.ceil(rejoin_distance_m / res_m)))
            first_rejoin_idx = max(branch_start_idx + 2, blocked_idx + min_rejoin_cells)

            for rejoin_idx in range(first_rejoin_idx, len(nominal_path)):
                rejoin_cell = nominal_path[rejoin_idx]
                rx, ry = rejoin_cell
                if not self._in_bounds_blocked(dynamic_blocked, rx, ry) or dynamic_blocked[ry][rx]:
                    continue
                detour = self._astar(
                    dynamic_blocked,
                    branch_start,
                    rejoin_cell,
                    allow_best_effort=False,
                    orthogonal_only=orthogonal_detour,
                )
                if detour is not None and len(detour) >= 2:
                    detour = self._simplify_grid_path(
                        detour,
                        dynamic_blocked,
                        float(dg.info.resolution),
                        force=self.smooth_avoidance_line_of_sight,
                    )
                    if len(detour) >= 2:
                        composed = _compose_detour_path(detour, rejoin_idx=rejoin_idx)
                        blind_zone_conflict = self._path_blind_zone_turn_conflict(
                            composed, dg, now_sec=now_sec
                        )
                        if blind_zone_conflict is not None:
                            rospy.loginfo_throttle(
                                1.0,
                                "constrained_local_replanner: rejecting avoidance branch into blind zone | side=%s obstacle=(%.2f,%.2f) age=%.2fs heading=%.1fdeg",
                                "left" if int(blind_zone_conflict["side"]) > 0 else "right",
                                float(blind_zone_conflict["x"]),
                                float(blind_zone_conflict["y"]),
                                float(blind_zone_conflict["age_s"]),
                                float(blind_zone_conflict["path_heading_deg"]),
                            )
                        else:
                            branch_history_points = self._sample_world_points(
                                [self._grid_to_world(dg, gx, gy) for gx, gy in composed]
                            )
                            candidate_direction, _candidate_offset = self._infer_avoid_direction(
                                dg, composed
                            )
                            if (
                                self.avoidance_same_side_commitment_enabled
                                and preferred_direction is not None
                                and candidate_direction not in (preferred_direction, "none")
                            ):
                                if exact_fallback_candidate is None:
                                    exact_fallback_candidate = composed
                                    exact_fallback_history = branch_history_points
                                continue
                            return composed, branch_history_points

                if not self.allow_best_effort_path:
                    continue

                partial_detour = self._astar(
                    dynamic_blocked,
                    branch_start,
                    rejoin_cell,
                    allow_best_effort=True,
                    orthogonal_only=orthogonal_detour,
                )
                if partial_detour is None or len(partial_detour) < 2:
                    continue

                partial_detour = self._simplify_grid_path(
                    partial_detour,
                    dynamic_blocked,
                    float(dg.info.resolution),
                    force=self.smooth_avoidance_line_of_sight,
                )
                if len(partial_detour) < 2:
                    continue

                partial_end = partial_detour[-1]
                if partial_end == branch_start:
                    continue

                progress_cells = self._heur(branch_start, partial_end)
                remaining_gap_cells = self._heur(partial_end, rejoin_cell)
                min_progress_cells = max(2.0, 0.35 * float(min_rejoin_cells))
                if progress_cells + 1e-6 < min_progress_cells:
                    continue

                composed = _compose_detour_path(partial_detour, rejoin_idx=None)
                blind_zone_conflict = self._path_blind_zone_turn_conflict(
                    composed, dg, now_sec=now_sec
                )
                if blind_zone_conflict is not None:
                    continue

                candidate_direction, _candidate_offset = self._infer_avoid_direction(
                    dg, composed
                )
                side_penalty = 0
                if (
                    self.avoidance_same_side_commitment_enabled
                    and preferred_direction is not None
                    and candidate_direction not in (preferred_direction, "none")
                ):
                    side_penalty = 1

                candidate_score = (
                    side_penalty,
                    remaining_gap_cells,
                    -progress_cells,
                    -float(len(partial_detour)),
                )
                if best_effort_score is None or candidate_score < best_effort_score:
                    best_effort_score = candidate_score
                    best_effort_candidate = composed

        if exact_fallback_candidate is not None:
            return exact_fallback_candidate, exact_fallback_history

        if best_effort_candidate is not None:
            rospy.loginfo_throttle(
                1.0,
                "constrained_local_replanner: using best-effort avoidance branch (exact rejoin path unavailable)"
            )
            branch_history_points = self._sample_world_points(
                [self._grid_to_world(dg, gx, gy) for gx, gy in best_effort_candidate]
            )
            return best_effort_candidate, branch_history_points

        return None, None

    def _update_avoidance_path(
        self,
        nominal_path,
        base_blocked,
        start_cell,
        goal_cell,
        dg,
        stamp,
        label,
    ):
        frame_id = dg.header.frame_id if dg.header.frame_id else "map"
        if not self.enable_avoidance_path or len(nominal_path) < 2:
            self._clear_avoidance_path(frame_id, stamp)
            self._publish_debug_text(
                self._build_debug_text(
                    "avoidance_disabled",
                    stamp,
                    trigger_reason="disabled",
                    path_len=len(nominal_path),
                ),
                stamp=stamp,
            )
            return "avoidance" if self.avoidance_active else "clear"

        tracked_for_avoidance = (
            self.tracked_object_virtual_obstacles_enabled
            and self.tracked_object_avoidance_enabled
        )
        live_trigger_points_map = self._combined_dynamic_obstacle_points(
            include_tracked=tracked_for_avoidance
        )
        dynamic_points_map = self._avoidance_trigger_points_map(
            include_tracked=tracked_for_avoidance
        )
        dynamic_blocked, obstacle_count = self._overlay_dynamic_obstacles(
            base_blocked,
            dg,
            keep_cells=(start_cell, goal_cell),
            include_tracked=tracked_for_avoidance,
            path=nominal_path,
            start_cell=start_cell,
            max_check_m=self.avoidance_trigger_ahead_m,
        )
        base_predicted_overlap = self._path_blocked_ahead(
            nominal_path,
            base_blocked,
            start_cell,
            float(dg.info.resolution),
            max_check_m=self.avoidance_trigger_ahead_m,
        )
        predicted_overlap = self._path_blocked_ahead(
            nominal_path,
            dynamic_blocked,
            start_cell,
            float(dg.info.resolution),
            max_check_m=self.avoidance_trigger_ahead_m,
        )
        direct_points_enabled = (
            self.use_pointcloud_avoidance_trigger
            or tracked_for_avoidance
        )
        blocking_points = []
        blocking_cells = []
        point_margin_m = self.obstacle_block_margin_m
        if predicted_overlap:
            blocking_cells = self._collect_confirmed_blocked_path_world_points(
                nominal_path,
                dynamic_blocked,
                start_cell,
                dg,
                max_check_m=self.avoidance_trigger_ahead_m,
            )
        if direct_points_enabled:
            point_margin_m = (
                self.obstacle_block_margin_m
                if predicted_overlap
                else (self.obstacle_block_margin_m + self.avoidance_trigger_margin_m)
            )
            blocking_points = self._collect_path_overlap_points(
                nominal_path,
                dg,
                start_cell,
                live_trigger_points_map,
                point_margin_m,
                max_check_m=self.avoidance_trigger_ahead_m,
            )
        direct_points_overlap = (
            direct_points_enabled
            and len(blocking_points) >= self.pointcloud_block_confirm_points
        )
        map_filtered_blocking_points = []
        if self.use_map_filtered_path_obstacle_trigger:
            map_filtered_blocking_points = self._collect_path_overlap_points(
                nominal_path,
                dg,
                start_cell,
                self.known_map_filtered_points_map,
                self.map_filtered_path_trigger_margin_m,
                max_check_m=self.avoidance_trigger_ahead_m,
            )
            if map_filtered_blocking_points:
                blocking_points = self._merge_point_sets(
                    blocking_points,
                    map_filtered_blocking_points,
                    self.pointcloud_cluster_resolution_m,
                )
        map_filtered_path_overlap = (
            self.use_map_filtered_path_obstacle_trigger
            and len(map_filtered_blocking_points)
            >= self.map_filtered_path_trigger_min_points
        )
        relevant_static_entries = self._relevant_static_memory_entries(
            blocking_points_world=blocking_points,
            blocking_cells_world=blocking_cells,
        )

        clustered_point_count = self.obstacle_cluster_count
        overlay_evidence_confirmed = (
            obstacle_count >= self.avoidance_min_overlay_points
            and clustered_point_count >= self.avoidance_min_cluster_count
        )
        point_evidence_present = (
            direct_points_enabled
            and len(blocking_points) >= self.pointcloud_block_confirm_points
        )
        grid_evidence_present = (
            len(blocking_cells) >= self.risk_block_confirm_cells
        )
        tracked_evidence_present = (
            self.tracked_object_count > 0
            or self.tracked_object_memory_count > 0
        )
        locked_static_evidence_present = len(relevant_static_entries) > 0
        obstacle_evidence_confirmed = (
            overlay_evidence_confirmed
            or point_evidence_present
            or map_filtered_path_overlap
            or grid_evidence_present
            or tracked_evidence_present
            or locked_static_evidence_present
        )
        if direct_points_enabled and direct_points_overlap and not obstacle_evidence_confirmed:
            direct_points_overlap = False
        if (
            predicted_overlap
            and not base_predicted_overlap
            and not obstacle_evidence_confirmed
        ):
            predicted_overlap = False
        self._debug_avoidance_log(
            "constrained_local_replanner: avoid_eval | base={} risk_grid={} predicted_overlap={} base_predicted_overlap={} direct_points_enabled={} tracked_avoidance={} direct_points_overlap={} map_filtered_path={} overlay_confirmed={} raw_points={} clustered_points={} map_filtered={} memory_points={} locked_static={} tracked_objects={} tracked_points={} tracked_memory_points={} overlay_points={} ahead={:.1f}m".format(
                label,
                "on" if self.risk_grid is not None else "off",
                "yes" if predicted_overlap else "no",
                "yes" if base_predicted_overlap else "no",
                "on" if direct_points_enabled else "off",
                "on" if tracked_for_avoidance else "off",
                "yes" if direct_points_overlap else "no",
                len(map_filtered_blocking_points),
                "yes" if overlay_evidence_confirmed else "no",
                self.obstacle_raw_point_count,
                clustered_point_count,
                self.known_map_filtered_count,
                self.obstacle_memory_count,
                self.obstacle_memory_locked_count,
                self.tracked_object_count,
                len(self.current_tracked_object_points_map),
                self.tracked_object_memory_count,
                obstacle_count,
                self.avoidance_trigger_ahead_m,
            )
        )

        trigger_reason = None
        if predicted_overlap:
            trigger_reason = "predicted_overlap"
        elif direct_points_overlap:
            trigger_reason = "dynamic_points_overlap"
        elif map_filtered_path_overlap:
            trigger_reason = "map_filtered_path_overlap"

        blind_zone_conflict = None
        if trigger_reason is None:
            blind_zone_conflict = self._path_blind_zone_turn_conflict(
                nominal_path, dg, now_sec=stamp.to_sec()
            )
            if blind_zone_conflict is not None:
                trigger_reason = "blind_zone_turn_conflict"

        trigger_key = None
        same_obstacle_episode = False
        preferred_direction = None
        if trigger_reason is not None:
            trigger_key = self._make_avoidance_trigger_key(
                trigger_reason,
                blocking_cells,
                blocking_points,
                blind_zone_conflict=blind_zone_conflict,
            )
            same_obstacle_episode = (
                trigger_key is not None
                and self.active_avoidance_obstacle_key is not None
                and trigger_key == self.active_avoidance_obstacle_key
            )
            if same_obstacle_episode and self.last_avoidance_direction in ("left", "right"):
                preferred_direction = self.last_avoidance_direction
            if (
                trigger_reason in ("predicted_overlap", "dynamic_points_overlap")
                and blind_zone_conflict is None
                and (not self.avoidance_memory_only_trigger_enabled)
            ):
                blocker_source_summary = self._build_path_blocker_source_summary(
                    nominal_path,
                    dg,
                    start_cell,
                    max_check_m=self.avoidance_trigger_ahead_m,
                    point_margin_m=(
                        self.obstacle_block_margin_m + self.avoidance_trigger_margin_m
                    ),
                )
                memory_only_trigger = (
                    int(blocker_source_summary.get("grid_occ", 0)) <= 0
                    and int(blocker_source_summary.get("risk", 0)) <= 0
                    and int(blocker_source_summary.get("pc_current", 0)) <= 0
                    and int(blocker_source_summary.get("map_filtered_path", 0)) <= 0
                    and int(blocker_source_summary.get("tracked_current", 0)) <= 0
                    and (
                        int(blocker_source_summary.get("pc_memory", 0)) > 0
                        or int(blocker_source_summary.get("tracked_memory", 0)) > 0
                    )
                )
                if memory_only_trigger and (not same_obstacle_episode):
                    self._debug_avoidance_log(
                        "constrained_local_replanner: suppressing memory-only avoidance trigger | reason={} pc_mem={} tracked_mem={}".format(
                            trigger_reason,
                            int(blocker_source_summary.get("pc_memory", 0)),
                            int(blocker_source_summary.get("tracked_memory", 0)),
                        )
                    )
                    trigger_reason = None
                    trigger_key = None
                    same_obstacle_episode = False
                    preferred_direction = None
                elif self._grid_only_nominal_fallback_allowed(blocker_source_summary):
                    rospy.logwarn_throttle(
                        1.0,
                        "constrained_local_replanner: suppressing sparse grid-only avoidance trigger | reason=%s grid_occ=%d limit=%d",
                        trigger_reason,
                        int(blocker_source_summary.get("grid_occ", 0)),
                        int(self.grid_only_nominal_fallback_max_cells),
                    )
                    trigger_reason = None
                    trigger_key = None
                    same_obstacle_episode = False
                    preferred_direction = None

        if trigger_reason is None:
            self._avoidance_trigger_confirmed(None, stamp)
            self._clear_blocking_obstacle_markers(stamp)
            if (not self._use_global_nominal_reference()) and self._hold_active_avoidance_until_endpoint(
                dg,
                stamp,
                clear_reason="nominal_path_clear",
            ):
                self._publish_debug_text(
                    self._build_debug_text(
                        "avoid_hold",
                        stamp,
                        trigger_reason="detour_endpoint_pending",
                        path_len=len(self.last_avoidance_grid_path)
                        if self.last_avoidance_grid_path is not None
                        else 0,
                        overlay_points=obstacle_count,
                    ),
                    stamp=stamp,
                )
                return "avoidance"
            if self._use_global_nominal_reference():
                self._clear_avoidance_path(frame_id, stamp, force=True)
            else:
                self._clear_avoidance_path(frame_id, stamp)
                if (
                    self.avoidance_active
                    and self.last_avoidance_grid_path is not None
                    and len(self.last_avoidance_grid_path) >= 2
                ):
                    # Keep the last valid detour fresh while the clear-hold / clear-confirm
                    # logic is still active.  Otherwise the controller can fall back to the
                    # nominal local path too early and re-approach the obstacle corridor
                    # before the recent person / object evidence has really cleared.
                    self._publish_stored_avoidance_path(
                        dg,
                        stamp,
                        record_history=False,
                    )
                    self._publish_debug_text(
                        self._build_debug_text(
                            "avoid_hold",
                            stamp,
                            trigger_reason="recent_clear_hold",
                            path_len=len(self.last_avoidance_grid_path),
                            overlay_points=obstacle_count,
                        ),
                        stamp=stamp,
                    )
                    return "avoidance"
            self._publish_debug_text(
                self._build_debug_text(
                    "follow_nominal",
                    stamp,
                    trigger_reason="clear",
                    path_len=len(nominal_path),
                    overlay_points=obstacle_count,
                ),
                stamp=stamp,
            )
            return "avoidance" if self.avoidance_active else "clear"
        if not self._avoidance_trigger_confirmed(trigger_key, stamp):
            self._publish_debug_text(
                self._build_debug_text(
                    "avoid_pending",
                    stamp,
                    trigger_reason=trigger_reason,
                    overlay_points=obstacle_count,
                    path_len=len(nominal_path),
                ),
                stamp=stamp,
            )
            return "avoidance" if self.avoidance_active else "clear"

        if self._continue_active_avoidance_path(
            dg,
            dynamic_blocked,
            start_cell,
            stamp,
            trigger_key=trigger_key,
        ):
            self._publish_debug_text(
                self._build_debug_text(
                    "avoid_reuse",
                    stamp,
                    trigger_reason=trigger_reason,
                    path_len=len(self.last_avoidance_grid_path)
                    if self.last_avoidance_grid_path is not None
                    else 0,
                    overlay_points=obstacle_count,
                ),
                stamp=stamp,
            )
            return "avoidance"
        source_summary = self._build_path_blocker_source_summary(
            nominal_path,
            dg,
            start_cell,
            max_check_m=self.avoidance_trigger_ahead_m,
            point_margin_m=point_margin_m,
            rg=self.risk_grid,
            blind_zone_conflict=blind_zone_conflict,
        )
        self._log_blocker_source_summary(
            "avoidance_eval",
            label,
            trigger_reason,
            source_summary,
        )
        if (
            label in ("local", "local_hold")
            and trigger_reason == "predicted_overlap"
            and self._grid_only_nominal_fallback_allowed(source_summary)
        ):
            rospy.logwarn_throttle(
                1.0,
                "constrained_local_replanner: grid-only blocker on %s path (grid_occ=%d); skipping expensive avoidance search and continuing nominal local path",
                label,
                int(source_summary.get("grid_occ", 0)),
            )
            self._clear_blocking_obstacle_markers(stamp)
            self._clear_avoidance_path(frame_id, stamp, force=True)
            self._publish_debug_text(
                self._build_debug_text(
                    "grid_only_nominal_fallback",
                    stamp,
                    trigger_reason=trigger_reason,
                    path_len=len(nominal_path),
                    overlay_points=obstacle_count,
                ),
                stamp=stamp,
                force=True,
            )
            return "nominal_fallback"
        self._publish_blocking_obstacle_markers(
            stamp,
            blocking_points=blocking_points,
            blocked_cells=blocking_cells,
            blind_zone_conflict=blind_zone_conflict,
            cell_scale_m=max(0.05, float(dg.info.resolution) * 0.95),
        )

        blocked_idx = self._first_blocked_path_index(
            nominal_path,
            dynamic_blocked,
            start_cell,
            dg,
            max_check_m=self.avoidance_trigger_ahead_m,
            include_pointcloud=direct_points_enabled,
            points_map=dynamic_points_map if direct_points_enabled else None,
            point_margin_m=self.obstacle_block_margin_m + self.avoidance_trigger_margin_m,
        )
        if self._should_ignore_near_goal_block(nominal_path, blocked_idx, start_cell, dg):
            rospy.loginfo_throttle(
                1.0,
                "constrained_local_replanner: ignoring near-goal %s trigger at path tail (blocked_idx=%d)",
                trigger_reason,
                int(blocked_idx),
            )
            self._clear_avoidance_path(frame_id, stamp)
            return "avoidance" if self.avoidance_active else "clear"

        avoid_solution_kind = "branch"
        curved_world_path = None
        curved_metrics = None
        avoid_path = None
        branch_history_points = None
        if self.sidestep_avoidance_enabled:
            (
                curved_world_path,
                avoid_path,
                branch_history_points,
                curved_metrics,
            ) = self._build_sidestep_avoidance_path(
                nominal_path,
                dynamic_blocked,
                start_cell,
                dg,
                blocked_idx=blocked_idx,
                blocking_cells_world=blocking_cells,
                blocking_points_world=blocking_points,
                preferred_direction=preferred_direction,
                now_sec=stamp.to_sec(),
            )
            if curved_world_path is not None and avoid_path is not None:
                avoid_solution_kind = "sidestep"
                rospy.loginfo_throttle(
                    1.0,
                    "constrained_local_replanner: using sidestep avoidance path | cells=%d pts=%d max_curv=%.2f max_heading_step=%.1fdeg",
                    len(avoid_path),
                    len(branch_history_points)
                    if branch_history_points is not None
                    else 0,
                    float(curved_metrics.get("max_curvature", 0.0))
                    if curved_metrics is not None
                    else 0.0,
                    float(curved_metrics.get("max_heading_delta_deg", 0.0))
                    if curved_metrics is not None
                    else 0.0,
                )
        if avoid_path is None and self.short_curved_avoidance_enabled:
            (
                curved_world_path,
                avoid_path,
                branch_history_points,
                curved_metrics,
            ) = self._build_short_curved_avoidance_path(
                nominal_path,
                dynamic_blocked,
                start_cell,
                dg,
                blocked_idx=blocked_idx,
                blocking_cells_world=blocking_cells,
                blocking_points_world=blocking_points,
                preferred_direction=preferred_direction,
                now_sec=stamp.to_sec(),
            )
            if curved_world_path is not None and avoid_path is not None:
                avoid_solution_kind = "short_curved"
                rospy.loginfo_throttle(
                    1.0,
                    "constrained_local_replanner: using short curved avoidance path | cells=%d pts=%d max_curv=%.2f max_heading_step=%.1fdeg",
                    len(avoid_path),
                    len(branch_history_points) if branch_history_points is not None else 0,
                    float(curved_metrics.get("max_curvature", 0.0))
                    if curved_metrics is not None
                    else 0.0,
                    float(curved_metrics.get("max_heading_delta_deg", 0.0))
                    if curved_metrics is not None
                    else 0.0,
                )
        if avoid_path is None:
            avoid_path, branch_history_points = self._build_branch_avoidance_path(
                nominal_path,
                dynamic_blocked,
                start_cell,
                dg,
                now_sec=stamp.to_sec(),
                # Use the same dynamic point source that triggered direct overlap so
                # near-goal tail ignores and branching start from a consistent index.
                points_map=dynamic_points_map if direct_points_enabled else None,
                blocking_cells_world=blocking_cells,
                blocking_points_world=blocking_points,
                preferred_direction=preferred_direction,
            )
        if avoid_path is None:
            if self.avoidance_active and self.last_avoidance_grid_path is not None and len(self.last_avoidance_grid_path) >= 2:
                self._publish_stored_avoidance_path(
                    dg,
                    stamp,
                    record_history=False,
                )
                self.avoidance_clear_count = 0
                self.last_avoidance_publish_sec = stamp.to_sec()
                rospy.loginfo_throttle(
                    1.0,
                    "constrained_local_replanner: keeping current avoidance path during no-solution hold | reason=%s",
                    trigger_reason,
                )
                return "hold"
            if self.avoidance_active and self._republish_last_avoidance_path(dg, stamp):
                return "avoidance"
            if self._should_follow_nominal_local_on_no_solution(
                label,
                trigger_reason,
                source_summary=source_summary,
                now_sec=stamp.to_sec(),
            ):
                rospy.logwarn_throttle(
                    1.0,
                    "constrained_local_replanner: obstacle detected on %s path (%s) but no valid avoidance branch was found; continuing nominal local path until a stronger stop trigger appears",
                    label,
                    trigger_reason,
                )
                self._publish_explainability(
                    event_type="LOCAL_REPLAN_NO_SOLUTION",
                    stamp=stamp,
                    trigger_reason=trigger_reason,
                    action_taken="follow_nominal_local",
                    local_planning_active=True,
                    stop_commanded=False,
                    summary_text=(
                        "Local replanning detected '{}' on the {} path but could not find a valid avoidance branch, so it kept the rolling nominal local path while near-field stop remained clear."
                    ).format(trigger_reason, label),
                )
                self._publish_debug_text(
                    self._build_debug_text(
                        "avoid_nominal_fallback",
                        stamp,
                        trigger_reason=trigger_reason,
                        path_len=len(nominal_path),
                        overlay_points=obstacle_count,
                    ),
                    stamp=stamp,
                    force=True,
                )
                self._clear_avoidance_path(frame_id, stamp, force=True)
                return "nominal_fallback"
            rospy.logwarn_throttle(
                1.0,
                "constrained_local_replanner: obstacle detected on %s path (%s) but no short curved or branch-rejoin avoidance was valid",
                label,
                trigger_reason,
            )
            self._publish_explainability(
                event_type="LOCAL_REPLAN_NO_SOLUTION",
                stamp=stamp,
                trigger_reason=trigger_reason,
                action_taken="hold_stop",
                local_planning_active=True,
                stop_commanded=True,
                summary_text=(
                    "Local replanning detected '{}' on the {} path but could not find a valid avoidance branch."
                ).format(trigger_reason, label),
            )
            self._publish_debug_text(
                self._build_debug_text(
                    "avoid_no_solution",
                    stamp,
                    trigger_reason=trigger_reason,
                    path_len=len(nominal_path),
                    overlay_points=obstacle_count,
                ),
                stamp=stamp,
                force=True,
            )
            # When no avoidance branch exists, publish an empty local path so the
            # cmd_vel relay enters local hold instead of continuing along a stale
            # nominal path toward the detected obstacle.
            self._clear_local_path(frame_id, stamp)
            self._clear_avoidance_path(frame_id, stamp, force=True)
            return "hold"

        if len(avoid_path) < 2:
            if self.avoidance_active and self.last_avoidance_grid_path is not None and len(self.last_avoidance_grid_path) >= 2:
                self._publish_stored_avoidance_path(
                    dg,
                    stamp,
                    record_history=False,
                )
                self.avoidance_clear_count = 0
                self.last_avoidance_publish_sec = stamp.to_sec()
                rospy.loginfo_throttle(
                    1.0,
                    "constrained_local_replanner: keeping current avoidance path during short-path hold | reason=%s",
                    trigger_reason,
                )
                return "hold"
            if self.avoidance_active and self._republish_last_avoidance_path(dg, stamp):
                return "avoidance"
            if self._should_follow_nominal_local_on_no_solution(
                label,
                trigger_reason,
                source_summary=source_summary,
                now_sec=stamp.to_sec(),
            ):
                rospy.logwarn_throttle(
                    1.0,
                    "constrained_local_replanner: avoidance candidate on %s path (%s) was too short; continuing nominal local path instead of clearing control reference",
                    label,
                    trigger_reason,
                )
                self._publish_debug_text(
                    self._build_debug_text(
                        "avoid_nominal_fallback",
                        stamp,
                        trigger_reason=trigger_reason,
                        path_len=len(avoid_path),
                        overlay_points=obstacle_count,
                    ),
                    stamp=stamp,
                    force=True,
                )
                self._clear_avoidance_path(frame_id, stamp, force=True)
                return "nominal_fallback"
            self._publish_debug_text(
                self._build_debug_text(
                    "avoid_too_short",
                    stamp,
                    trigger_reason=trigger_reason,
                    path_len=len(avoid_path),
                    overlay_points=obstacle_count,
                ),
                stamp=stamp,
                force=True,
            )
            self._clear_local_path(frame_id, stamp)
            self._clear_avoidance_path(frame_id, stamp, force=True)
            return "hold"

        if avoid_solution_kind in ("short_curved", "sidestep") and curved_world_path is not None:
            self._publish_world_avoidance_path(
                curved_world_path,
                dg,
                stamp,
                history_points=branch_history_points,
            )
        else:
            self._publish_operational_detour_path(
                avoid_path,
                dg,
                stamp,
                history_points=branch_history_points,
                start_xy=(self.odom_x, self.odom_y),
            )
        avoid_direction, lateral_offset = self._infer_avoid_direction(dg, avoid_path)
        self.avoidance_clear_count = 0
        self.last_avoidance_publish_sec = stamp.to_sec()
        self.last_avoidance_solution_sec = stamp.to_sec()
        self.last_avoidance_validation_sec = stamp.to_sec()
        if (not self.avoidance_active) or trigger_reason != self.last_avoidance_trigger_reason or avoid_direction != self.last_avoidance_direction:
            action_taken = "avoid_{}".format(avoid_direction) if avoid_direction in ("left", "right") else "avoid"
            self._publish_explainability(
                event_type="LOCAL_REPLAN_AVOIDANCE_ACTIVE",
                stamp=stamp,
                trigger_reason=trigger_reason,
                action_taken=action_taken,
                avoid_direction=avoid_direction,
                local_planning_active=True,
                obstacle_lateral_offset_m=lateral_offset,
                summary_text=(
                    "Local replanning activated an avoidance {} because '{}' blocked the {} path."
                ).format(
                    avoid_direction if avoid_direction in ("left", "right") else "detour",
                    trigger_reason,
                    label,
                ),
            )
        if not self.avoidance_active:
            rospy.loginfo(
                "constrained_local_replanner: avoidance path active | base=%s reason=%s solution=%s obstacle_points=%d cells=%d",
                label,
                trigger_reason,
                avoid_solution_kind,
                clustered_point_count,
                len(avoid_path),
            )
        self.last_avoidance_trigger_reason = trigger_reason
        self.last_avoidance_direction = avoid_direction
        self.active_avoidance_obstacle_key = trigger_key
        self.avoidance_active = True
        self._publish_debug_text(
            self._build_debug_text(
                "avoid_active",
                stamp,
                trigger_reason=trigger_reason,
                avoid_direction=avoid_direction,
                path_len=len(avoid_path),
                overlay_points=obstacle_count,
            ),
            stamp=stamp,
            force=True,
        )
        return "avoidance"

    def _plan_direct_goal(self, dg, rg, stamp):
        if self.direct_goal is None:
            self._clear_avoidance_path(dg.header.frame_id, stamp)
            return False
        if (
            self.direct_goal_timeout_s > 0.0
            and self.direct_goal_stamp_sec > 0.0
            and (stamp.to_sec() - self.direct_goal_stamp_sec) > self.direct_goal_timeout_s
        ):
            self._clear_avoidance_path(dg.header.frame_id, stamp)
            return False

        start_xy = (self.odom_x, self.odom_y)
        goal_xy = (
            float(self.direct_goal.pose.position.x),
            float(self.direct_goal.pose.position.y),
        )
        dist_to_goal = math.hypot(goal_xy[0] - start_xy[0], goal_xy[1] - start_xy[1])
        if dist_to_goal <= self.goal_tolerance_m:
            self._publish_world_path([start_xy, goal_xy], dg.header.frame_id, stamp)
            self._clear_avoidance_path(dg.header.frame_id, stamp)
            self.rejoin_mode_until_sec = 0.0
            self._publish_path_mode("follow_local")
            return True

        sx, sy = self._world_to_grid(dg, start_xy[0], start_xy[1])
        gx, gy = self._world_to_grid(dg, goal_xy[0], goal_xy[1])

        blocked = self._inflate_blocked(dg, rg)
        start_cell = self._resolve_snap_cell(
            dg,
            rg,
            blocked,
            (sx, sy),
            allow_raw_cell=True,
        )
        goal_cell = self._resolve_direct_goal_cell(dg, rg, blocked, (gx, gy))
        if start_cell is None or goal_cell is None:
            rospy.logwarn_throttle(
                1.0,
                "constrained_local_replanner: no free snapped cell for direct goal (start=%s goal=%s)",
                str((sx, sy)),
                str((gx, gy)),
            )
            self._clear_avoidance_path(dg.header.frame_id, stamp)
            self.rejoin_mode_until_sec = 0.0
            self._publish_path_mode("hold")
            return True

        planning_blocked, planning_mode = self._build_direct_goal_planning_blocked(
            dg,
            rg,
            blocked,
            start_cell,
            goal_cell,
        )

        if (
            self.freeze_path_on_first_plan
            and self.frozen_direct_grid_path
        ):
            frozen_goal_cell = (
                self.frozen_direct_goal_cell
                if self.frozen_direct_goal_cell is not None
                else self.frozen_direct_grid_path[-1]
            )
            self._publish_local_path(
                self.frozen_direct_grid_path,
                dg,
                stamp,
                start_xy=self.frozen_direct_start_xy,
                end_xy=self.frozen_direct_goal_xy,
            )
            avoidance_state = self._update_avoidance_path(
                self.frozen_direct_grid_path,
                planning_blocked,
                start_cell,
                frozen_goal_cell,
                dg,
                stamp,
                "direct(frozen)",
            )
            self.rejoin_mode_until_sec = 0.0
            if avoidance_state == "hold":
                self._publish_path_mode("hold")
            elif avoidance_state != "avoidance":
                self._publish_path_mode("follow_local")
            return True

        path = self._astar(
            planning_blocked,
            start_cell,
            goal_cell,
            allow_best_effort=self.allow_best_effort_path,
        )
        if path is None:
            rospy.logwarn_throttle(
                1.0,
                "constrained_local_replanner: no direct-goal path (mode=%s start=%s goal=%s snapped_start=%s snapped_goal=%s)",
                planning_mode,
                str((sx, sy)),
                str((gx, gy)),
                str(start_cell),
                str(goal_cell),
            )
            self._clear_avoidance_path(dg.header.frame_id, stamp)
            self.rejoin_mode_until_sec = 0.0
            self._publish_path_mode("hold")
            return True
        # If the direct goal is visible on the blocked grid, prefer a single
        # straight segment over the staircase-like A* cell path.
        if path[-1] == goal_cell and self._has_line_of_sight(
            planning_blocked, start_cell, goal_cell
        ):
            path = [start_cell, goal_cell]
        else:
            path = self._simplify_grid_path(
                path, planning_blocked, float(dg.info.resolution)
            )
        if not self._best_effort_path_is_acceptable(goal_cell, path, dg, "direct"):
            self._clear_local_path(dg.header.frame_id, stamp)
            self._clear_avoidance_path(dg.header.frame_id, stamp)
            self.rejoin_mode_until_sec = 0.0
            self._publish_path_mode("hold")
            return True

        if not self._should_publish_path(goal_cell, path):
            avoidance_state = self._update_avoidance_path(
                path,
                planning_blocked,
                start_cell,
                goal_cell,
                dg,
                stamp,
                "direct",
            )
            self.rejoin_mode_until_sec = 0.0
            if avoidance_state == "hold":
                self._publish_path_mode("hold")
            elif avoidance_state != "avoidance":
                self._publish_path_mode("follow_local")
            return True

        if path[-1] != goal_cell:
            rospy.logwarn_throttle(
                1.0,
                "constrained_local_replanner: best-effort direct path only (snapped_goal=%s reached=%s)",
                str(goal_cell),
                str(path[-1]),
            )
        if self.freeze_path_on_first_plan:
            self.frozen_direct_goal_cell = goal_cell
            self.frozen_direct_grid_path = list(path)
            self.frozen_direct_start_xy = start_xy
            self.frozen_direct_goal_xy = goal_xy
            rospy.loginfo(
                "constrained_local_replanner: path frozen for goal=%s with %d cells",
                str(goal_cell),
                len(path),
            )
        self._publish_local_path(
            path,
            dg,
            stamp,
            start_xy=start_xy,
            end_xy=goal_xy if path[-1] == goal_cell else None,
        )
        avoidance_state = self._update_avoidance_path(
            path,
            planning_blocked,
            start_cell,
            goal_cell,
            dg,
            stamp,
            "direct",
        )
        self.rejoin_mode_until_sec = 0.0
        if avoidance_state == "hold":
            self._publish_path_mode("hold")
        elif avoidance_state != "avoidance":
            self._publish_path_mode("follow_local")
        self._record_published_path(goal_cell, path)
        return True

    @staticmethod
    def _age_from_stamp_sec(stamp_sec, now_sec):
        try:
            stamp_sec = float(stamp_sec)
        except (TypeError, ValueError):
            return -1.0
        if stamp_sec <= 0.0:
            return -1.0
        return max(0.0, float(now_sec) - stamp_sec)

    def _grid_stamp_age_s(self, grid, now_sec):
        if grid is None:
            return -1.0
        stamp_sec = grid.header.stamp.to_sec()
        return self._age_from_stamp_sec(stamp_sec, now_sec)

    def _log_timer_timing(self, loop_start_sec, timer_gap_s):
        if not self.debug_timing_logging or loop_start_sec <= 0.0:
            return
        now_sec = rospy.Time.now().to_sec()
        loop_s = max(0.0, now_sec - loop_start_sec)
        gap_s = max(0.0, float(timer_gap_s)) if timer_gap_s > 0.0 else 0.0
        overrun = loop_s > self.debug_timing_overrun_s or (
            gap_s > 0.0 and gap_s > 1.5 * self.replan_period_s
        )
        log_fn = rospy.logwarn_throttle if overrun else rospy.loginfo_throttle
        log_fn(
            self.debug_timing_log_period_s,
            "constrained_local_replanner timing | status=%s loop=%.3fs gap=%.3fs target=%.3fs odom_age=%.2fs cloud_age=%.2fs raw_age=%.2fs grid_age=%.2fs raw_pts=%d clustered=%d",
            "overrun" if overrun else "ok",
            loop_s,
            gap_s,
            self.replan_period_s,
            self._age_from_stamp_sec(self.odom_stamp_sec, now_sec),
            self._age_from_stamp_sec(self.current_obstacle_points_stamp_sec, now_sec),
            self._age_from_stamp_sec(self.raw_near_obstacle_points_stamp_sec, now_sec),
            self._grid_stamp_age_s(self.drivable_grid, now_sec),
            int(self.obstacle_raw_point_count),
            int(self.obstacle_cluster_count),
        )

    def on_timer(self, _evt):
        loop_start_sec = rospy.Time.now().to_sec()
        timer_gap_s = (
            loop_start_sec - self._last_timer_start_sec
            if self._last_timer_start_sec > 0.0
            else 0.0
        )
        self._last_timer_start_sec = loop_start_sec
        try:
            if (not self.have_odom) or self.drivable_grid is None:
                return
            self._maybe_commit_pending_used_local_trace()
            dg = self.drivable_grid
            rg = self.risk_grid
            stamp = rospy.Time.now()
            self._clear_blocking_obstacle_markers(stamp)

            if self.use_direct_goal and self._plan_direct_goal(dg, rg, stamp):
                return

            pts = self._global_path_points()
            if len(pts) < 2:
                self.global_nominal_progress_idx = 0
                self._clear_avoidance_path(dg.header.frame_id, stamp)
                self.rejoin_mode_until_sec = 0.0
                self._publish_path_mode("hold")
                self._publish_debug_text(
                    self._build_debug_text("no_global_path", stamp, trigger_reason="missing_global"),
                    stamp=stamp,
                )
                return
            if self._fast_reuse_active_avoidance_path(dg, stamp):
                return
            i0 = self._nominal_global_start_index(pts)
            nominal_horizon_m = self.lookahead_m
            planning_horizon_m = max(nominal_horizon_m, self.avoidance_plan_horizon_m)
            nominal_ig = self._accum_distance(pts, i0, nominal_horizon_m)
            planning_ig = self._accum_distance(pts, i0, planning_horizon_m)
            start_xy = (self.odom_x, self.odom_y)
            goal_xy = pts[planning_ig]

            sx, sy = self._world_to_grid(dg, start_xy[0], start_xy[1])
            gx, gy = self._world_to_grid(dg, goal_xy[0], goal_xy[1])

            blocked = self._inflate_blocked(dg, rg)
            start_cell = self._resolve_snap_cell(
                dg,
                rg,
                blocked,
                (sx, sy),
                allow_raw_cell=True,
            )
            goal_cell = self._resolve_snap_cell(
                dg,
                rg,
                blocked,
                (gx, gy),
                allow_raw_cell=False,
            )
            if start_cell is None or goal_cell is None:
                rospy.logwarn_throttle(
                    1.0,
                    "constrained_local_replanner: no local path snap cell (start=%s goal=%s)",
                    str((sx, sy)),
                    str((gx, gy)),
                )
                self.local_blocked_since_sec = 0.0
                self.rejoin_mode_until_sec = 0.0
                self._clear_local_path(dg.header.frame_id, stamp)
                self._clear_avoidance_path(dg.header.frame_id, stamp)
                self._publish_path_mode("hold")
                self._publish_debug_text(
                    self._build_debug_text("no_snap_cell", stamp, trigger_reason="snap_failed"),
                    stamp=stamp,
                    force=True,
                )
                return
            nominal_path, nominal_world, nominal_fail_reason = self._build_nominal_local_path(
                pts, i0, nominal_ig, dg
            )
            if nominal_path is None or nominal_world is None:
                nominal_world_remain_m = self._world_path_length(nominal_world)
                if (
                    nominal_fail_reason == "degenerate_grid"
                    and nominal_world is not None
                    and len(nominal_world) >= 2
                    and nominal_world_remain_m <= self.near_goal_block_ignore_distance_m
                    and self._can_use_near_goal_shortcut(nominal_world_remain_m, stamp.to_sec())
                ):
                    rospy.loginfo_throttle(
                        1.0,
                        "constrained_local_replanner: keeping short near-goal nominal segment in world frame (points=%d remain=%.2f m)",
                        len(nominal_world),
                        nominal_world_remain_m,
                    )
                    self.local_blocked_since_sec = 0.0
                    self.rejoin_mode_until_sec = 0.0
                    self._publish_nominal_reference_path(
                        nominal_world, dg.header.frame_id, stamp
                    )
                    self._clear_avoidance_path(dg.header.frame_id, stamp)
                    self._publish_debug_text(
                        self._build_debug_text(
                            "follow_nominal_world",
                            stamp,
                            trigger_reason="nominal_segment_short_near_goal",
                            path_len=len(nominal_world),
                        ),
                        stamp=stamp,
                        force=True,
                    )
                    return
                rospy.logwarn_throttle(
                    1.0,
                    "constrained_local_replanner: failed to build nominal local segment from global path (reason=%s)",
                    nominal_fail_reason if nominal_fail_reason else "unknown",
                )
                self.local_blocked_since_sec = 0.0
                self.rejoin_mode_until_sec = 0.0
                self._clear_local_path(dg.header.frame_id, stamp)
                self._clear_avoidance_path(dg.header.frame_id, stamp)
                self._publish_path_mode("hold")
                self._publish_debug_text(
                    self._build_debug_text(
                        "nominal_build_failed",
                        stamp,
                        trigger_reason="nominal_segment_invalid",
                    ),
                    stamp=stamp,
                    force=True,
                )
                return
            planning_path, planning_world, planning_fail_reason = self._build_nominal_local_path(
                pts, i0, planning_ig, dg
            )
            if planning_path is None or planning_world is None:
                rospy.logwarn_throttle(
                    1.0,
                    "constrained_local_replanner: failed to build long-horizon planning segment from global path (reason=%s); falling back to nominal horizon",
                    planning_fail_reason if planning_fail_reason else "unknown",
                )
                planning_path = nominal_path
                planning_world = nominal_world

            blocked_idx = self._first_blocked_path_index(
                planning_path,
                blocked,
                start_cell,
                dg,
                max_check_m=planning_horizon_m,
                include_pointcloud=self.use_pointcloud_static_blocking,
            )
            nominal_blocked = blocked_idx is not None
            source_summary = None
            if nominal_blocked:
                blocked_reason = "nominal_path_blocked"
                relaxed_blocked = None
                if (
                    self.near_goal_relaxed_path_blocking_radius_m + 1e-6
                    < self.path_blocking_radius_m
                ):
                    relaxed_blocked = self._inflate_blocked(
                        dg,
                        rg,
                        radius_override_m=self.near_goal_relaxed_path_blocking_radius_m,
                    )
                if self._should_ignore_near_goal_block(
                    planning_path,
                    blocked_idx,
                    start_cell,
                    dg,
                    relaxed_blocked=relaxed_blocked,
                ):
                    rospy.loginfo_throttle(
                        1.0,
                        "constrained_local_replanner: ignoring near-goal nominal block at path tail (blocked_idx=%d)",
                        int(blocked_idx),
                    )
                    nominal_blocked = False
                if nominal_blocked:
                    blind_zone_conflict = self._path_blind_zone_turn_conflict(
                        planning_path, dg, now_sec=stamp.to_sec()
                    )
                    if blind_zone_conflict is not None:
                        blocked_reason = "blind_zone_turn_conflict"
            else:
                blocked_reason = "nominal_path_blocked"
                blind_zone_conflict = self._path_blind_zone_turn_conflict(
                    planning_path, dg, now_sec=stamp.to_sec()
                )
                if blind_zone_conflict is not None:
                    nominal_blocked = True
                    blocked_reason = "blind_zone_turn_conflict"
                    rospy.loginfo_throttle(
                        1.0,
                        "constrained_local_replanner: nominal path turn conflicts with blind zone | side=%s obstacle=(%.2f,%.2f) age=%.2fs heading=%.1fdeg",
                        "left" if int(blind_zone_conflict["side"]) > 0 else "right",
                        float(blind_zone_conflict["x"]),
                        float(blind_zone_conflict["y"]),
                        float(blind_zone_conflict["age_s"]),
                        float(blind_zone_conflict["path_heading_deg"]),
                    )
            if nominal_blocked:
                source_summary = self._build_path_blocker_source_summary(
                    planning_path,
                    dg,
                    start_cell,
                    max_check_m=planning_horizon_m,
                    point_margin_m=self.pointcloud_static_block_margin_m,
                    rg=rg,
                    blind_zone_conflict=blind_zone_conflict,
                )
                if (
                    blocked_reason == "nominal_path_blocked"
                    and self._should_ignore_grid_only_nominal_block(
                        planning_path,
                        start_cell,
                        dg,
                        rg,
                        source_summary,
                    )
                ):
                    rospy.loginfo_throttle(
                        1.0,
                        "constrained_local_replanner: ignoring grid-only nominal block using relaxed radius (grid_occ=%d radius=%.2f->%.2f)",
                        int(source_summary.get("grid_occ", 0)),
                        float(self.path_blocking_radius_m),
                        float(self.grid_only_relaxed_path_blocking_radius_m),
                    )
                    nominal_blocked = False
                elif (
                    blocked_reason == "nominal_path_blocked"
                    and self._grid_only_nominal_fallback_allowed(source_summary)
                    and self._is_grid_only_blocker_source_summary(source_summary)
                ):
                    rospy.logwarn_throttle(
                        1.0,
                        "constrained_local_replanner: treating grid-only nominal block as passable before avoidance search (grid_occ=%d limit=%d)",
                        int(source_summary.get("grid_occ", 0)),
                        int(self.grid_only_nominal_fallback_max_cells),
                    )
                    nominal_blocked = False
                    self.local_blocked_since_sec = 0.0
                    self.local_clear_since_sec = 0.0
                    self._clear_blocking_obstacle_markers(stamp)
                    self._publish_debug_text(
                        self._build_debug_text(
                            "grid_only_nominal_fallback",
                            stamp,
                            trigger_reason=blocked_reason,
                            path_len=len(planning_path),
                            overlay_points=0,
                        ),
                        stamp=stamp,
                        force=True,
                    )

            if nominal_blocked:
                blocking_points = []
                if self.use_pointcloud_static_blocking:
                    blocking_points = self._collect_path_overlap_points(
                        planning_path,
                        dg,
                        start_cell,
                        self.obstacle_points_map,
                        self.pointcloud_static_block_margin_m,
                        max_check_m=planning_horizon_m,
                    )
                blocking_cells = self._collect_confirmed_blocked_path_world_points(
                    planning_path,
                    blocked,
                    start_cell,
                    dg,
                    max_check_m=planning_horizon_m,
                )
                if source_summary is None:
                    source_summary = self._build_path_blocker_source_summary(
                        planning_path,
                        dg,
                        start_cell,
                        max_check_m=planning_horizon_m,
                        point_margin_m=self.pointcloud_static_block_margin_m,
                        rg=rg,
                        blind_zone_conflict=blind_zone_conflict,
                    )
                self._log_blocker_source_summary(
                    "nominal_block",
                    "local",
                    blocked_reason,
                    source_summary,
                )
                self._publish_blocking_obstacle_markers(
                    stamp,
                    blocking_points=blocking_points,
                    blocked_cells=blocking_cells,
                    blind_zone_conflict=blind_zone_conflict,
                    cell_scale_m=max(0.05, float(dg.info.resolution) * 0.95),
                )
                now_sec = stamp.to_sec()
                self.local_clear_since_sec = 0.0
                if self.local_blocked_since_sec <= 0.0:
                    self.local_blocked_since_sec = now_sec
                    if self.blocked_stop_before_avoidance_s > 0.0:
                        rospy.loginfo(
                            "constrained_local_replanner: nominal path blocked; holding %.1fs before avoidance",
                            self.blocked_stop_before_avoidance_s,
                        )
                        self._publish_explainability(
                            event_type="LOCAL_REPLAN_START",
                            stamp=stamp,
                            trigger_reason=blocked_reason,
                            action_taken="hold_stop",
                            local_planning_active=True,
                            stop_commanded=True,
                            summary_text=(
                                "Nominal local path became unsafe, so local replanning started and the robot entered a hold state for %.1f seconds before avoidance."
                            )
                            % self.blocked_stop_before_avoidance_s,
                        )
                    else:
                        rospy.loginfo(
                            "constrained_local_replanner: nominal path blocked; evaluating avoidance immediately"
                        )
                        self._publish_explainability(
                            event_type="LOCAL_REPLAN_START",
                            stamp=stamp,
                            trigger_reason=blocked_reason,
                            action_taken="avoid_immediate",
                            local_planning_active=True,
                            stop_commanded=False,
                            summary_text=(
                                "Nominal local path became unsafe, so local replanning immediately evaluated an avoidance path without a hold delay."
                            ),
                        )
                wait_s = now_sec - self.local_blocked_since_sec
                if wait_s < self.blocked_stop_before_avoidance_s:
                    self._clear_local_path(dg.header.frame_id, stamp)
                    self.rejoin_mode_until_sec = 0.0
                    self._publish_path_mode("hold")
                    self._publish_debug_text(
                        self._build_debug_text(
                            "hold_wait",
                            stamp,
                            trigger_reason=blocked_reason,
                            wait_s=wait_s,
                            path_len=len(planning_path),
                        ),
                        stamp=stamp,
                        force=True,
                    )
                    if self._republish_last_avoidance_path(dg, stamp):
                        return
                    self._clear_avoidance_path(dg.header.frame_id, stamp)
                    return
                avoidance_state = self._update_avoidance_path(
                    planning_path,
                    blocked,
                    start_cell,
                    goal_cell,
                    dg,
                    stamp,
                    "local_hold",
                )
                self.rejoin_mode_until_sec = 0.0
                if avoidance_state == "hold":
                    self._publish_path_mode("hold")
                elif avoidance_state in ("clear", "nominal_fallback"):
                    self.local_blocked_since_sec = 0.0
                    self.local_clear_since_sec = 0.0
                    self._publish_nominal_reference_path(
                        nominal_world, dg.header.frame_id, stamp
                    )
                return

            resumed_from_local_block = self.local_blocked_since_sec > 0.0
            now_sec = stamp.to_sec()
            if self.local_blocked_since_sec > 0.0:
                if self.blocked_clear_hold_s > 1e-6:
                    if self.local_clear_since_sec <= 0.0:
                        self.local_clear_since_sec = now_sec
                        rospy.loginfo(
                            "constrained_local_replanner: nominal path appears clear; holding %.2fs before resuming global tracking",
                            self.blocked_clear_hold_s,
                        )
                    clear_wait_s = now_sec - self.local_clear_since_sec
                    if clear_wait_s < self.blocked_clear_hold_s:
                        if self.avoidance_active and self._republish_last_avoidance_path(dg, stamp):
                            self.rejoin_mode_until_sec = 0.0
                            self._publish_debug_text(
                                self._build_debug_text(
                                    "clear_wait_follow",
                                    stamp,
                                    trigger_reason="nominal_path_clear",
                                    wait_s=clear_wait_s,
                                    path_len=len(nominal_path),
                                ),
                                stamp=stamp,
                                force=True,
                            )
                            return
                        self._clear_local_path(dg.header.frame_id, stamp)
                        self._clear_avoidance_path(dg.header.frame_id, stamp, force=True)
                        self.rejoin_mode_until_sec = 0.0
                        self._publish_path_mode("hold")
                        self._publish_debug_text(
                            self._build_debug_text(
                                "clear_wait",
                                stamp,
                                trigger_reason="nominal_path_clear",
                                wait_s=clear_wait_s,
                                path_len=len(nominal_path),
                            ),
                            stamp=stamp,
                            force=True,
                        )
                        return
                rospy.loginfo(
                    "constrained_local_replanner: nominal path clear; resuming nominal local tracking"
                )
                self._publish_explainability(
                    event_type="LOCAL_REPLAN_END",
                    stamp=stamp,
                    trigger_reason="nominal_path_clear",
                    action_taken="resume_nominal_local",
                    local_planning_active=False,
                    summary_text="Nominal local path is clear again, so the replanner ended the active detour and returned control to the rolling nominal local path.",
                )
            self.local_blocked_since_sec = 0.0
            self.local_clear_since_sec = 0.0
            avoidance_state = self._update_avoidance_path(planning_path, blocked, start_cell, goal_cell, dg, stamp, "local")
            if avoidance_state == "avoidance":
                self.rejoin_mode_until_sec = 0.0
                return
            if avoidance_state == "hold":
                self.rejoin_mode_until_sec = 0.0
                self._publish_path_mode("hold")
                return
            self._publish_nominal_reference_path(
                nominal_world, dg.header.frame_id, stamp
            )
            # Keep the controller on the replanner's nominal local path even
            # after an avoidance episode clears.  Falling back to the full
            # global path here reintroduces late switching and S-curve chasing
            # because the mux stops using the fresh local segment that was just
            # built around the current pose.
            self.rejoin_mode_until_sec = 0.0
            self._publish_path_mode(
                "follow_global" if self._use_global_nominal_reference() else "follow_local"
            )
            self._publish_debug_text(
                self._build_debug_text(
                    "follow_nominal",
                    stamp,
                    trigger_reason="clear",
                    path_len=len(nominal_path),
                ),
                stamp=stamp,
            )
        except Exception as e:
            rospy.logwarn_throttle(
                1.0,
                "constrained_local_replanner error: %s\n%s",
                str(e),
                traceback.format_exc(),
            )
        finally:
            self._log_timer_timing(loop_start_sec, timer_gap_s)


def main():
    rospy.init_node("constrained_local_replanner", anonymous=False)
    ConstrainedLocalReplanner()
    rospy.spin()


if __name__ == "__main__":
    main()
