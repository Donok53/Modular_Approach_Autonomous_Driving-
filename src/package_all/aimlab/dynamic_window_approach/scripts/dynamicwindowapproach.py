#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DWA local planner — stable path tracking for A* overlap/crossing segments

Fixes vs. original:
  • Ignore /astar/path if it hasn't actually changed (hash signature).
  • Track along-path progress with arc-length s (monotonic, jitter tolerant).
  • Pure-Pursuit style target: point at s + lookahead_distance on the polyline.
  • Progress cost to discourage backwards/sideways motions near junctions.
  • Rotate-only mode with clean hysteresis and path-tangent gating.
  • Minimal forward obstacle stop using front ROI from PointCloud2.
  • Snap-to-path when lateral error is large, then follow lookahead once on track.
"""

import math
import numpy as np
import rospy
import tf.transformations as transformations

from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32MultiArray

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
        self.local_path_timeout_s = float(rospy.get_param("~local_path_timeout_s", 4.0))
        self.avoidance_path_timeout_s = float(
            rospy.get_param("~avoidance_path_timeout_s", max(0.5, self.local_path_timeout_s))
        )
        self.use_muxed_active_path = bool(rospy.get_param("~use_muxed_active_path", True))
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
            0.05, float(rospy.get_param("~avoidance_hard_stop_distance", 0.20))
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
        self.max_z = rospy.get_param("~max_z", 1.5)
        self.self_filter_radius_x = max(
            0.0, float(rospy.get_param("~self_filter_radius_x", self.robot_half_length_m))
        )
        self.self_filter_radius_y = max(
            0.0, float(rospy.get_param("~self_filter_radius_y", self.robot_half_width_m))
        )
        self.cloud_downsample = rospy.get_param("~cloud_downsample", 4)
        self.traj_check_step = max(1, int(rospy.get_param("~traj_check_step", 2)))
        self.max_obstacle_points = max(20, int(rospy.get_param("~max_obstacle_points", 300)))
        self.emergency_bin_size_m = max(
            0.05, float(rospy.get_param("~emergency_bin_size_m", 0.10))
        )
        self.emergency_min_close_points = max(
            1, int(rospy.get_param("~emergency_min_close_points", 4))
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
        self.cmd_publish_hz = max(5.0, float(rospy.get_param("~cmd_publish_hz", 20.0)))
        self.last_cmd = Twist()

        # ===== ROS I/O =====
        self.sub_path_global = rospy.Subscriber(self.global_path_topic, Path, self.path_callback_global, queue_size=5)
        self.sub_path_local = rospy.Subscriber(self.local_path_topic, Path, self.path_callback_local, queue_size=5)
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
        self.sub_behavior = rospy.Subscriber(self.behavior_cmd_topic, BehaviorCommand, self.behavior_cmd_callback, queue_size=10)
        self.sub_cloud = rospy.Subscriber(self.cloud_topic, PointCloud2, self.cloud_callback, queue_size=1)
        self.sub_grid = None
        if self.use_drivable_grid:
            self.sub_grid = rospy.Subscriber(self.drivable_grid_topic, OccupancyGrid, self.drivable_grid_callback, queue_size=5)
        self.sub_risk_grid = None
        if self.use_dynamic_risk_grid:
            self.sub_risk_grid = rospy.Subscriber(self.dynamic_risk_grid_topic, OccupancyGrid, self.risk_grid_callback, queue_size=5)

        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.target_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
        self.current_pub = rospy.Publisher('visualization_marker_2', Marker, queue_size=10)
        self.trajectory_pub = rospy.Publisher('predicted_trajectory', Marker, queue_size=10)
        self.traj_info_pub = rospy.Publisher('/traj_info', Float32MultiArray, queue_size=10)
        self.cmd_timer = rospy.Timer(rospy.Duration(1.0 / self.cmd_publish_hz), self._cmd_timer_callback)

    def _cmd_timer_callback(self, _event):
        self.cmd_vel_pub.publish(self.last_cmd)

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
            immediate_contact = False
            min_front_clearance = float("inf")
            influence_sq = self.obstacle_influence_distance * self.obstacle_influence_distance
            stop_half_w = 0.5 * max(self.stop_width, self.robot_width_m) + self.footprint_padding_m
            i = 0
            for pt in point_cloud2.read_points(msg, field_names=('x','y','z'), skip_nans=True):
                i += 1
                if self.cloud_downsample > 1 and (i % self.cloud_downsample != 0):
                    continue
                x, y, z = pt
                if self._point_in_local_rect(x, y, self.self_filter_radius_x, self.self_filter_radius_y):
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
                    immediate_contact = True
                    close_points.append((x, y))
                    continue
                if front_clearance < 0.0:
                    continue
                if front_clearance <= self.emergency_stop_distance:
                    close_points.append((x, y))

            near = immediate_contact or self._emergency_band_is_blocked(
                close_points,
                self.obstacle_consider_side_m,
            )

            if obs:
                step = max(1, len(obs) // self.max_obstacle_points)
                self.obstacle_local_points = np.array(obs[::step], dtype=np.float32)
            else:
                self.obstacle_local_points = np.empty((0, 2), dtype=np.float32)
            self.front_obstacle_clearance = min_front_clearance

            if near:
                self._blk_on += 1
                self._blk_off = 0
            else:
                self._blk_off += 1
                self._blk_on = 0
            if not self.emergency_blocked and self._blk_on >= self.block_on_count:
                self.emergency_blocked = True
                rospy.logwarn("Emergency STOP: obstacle <= %.2fm", self.emergency_stop_distance)
            elif self.emergency_blocked and self._blk_off >= self.block_off_count:
                self.emergency_blocked = False
                rospy.loginfo("Emergency STOP cleared")
        except Exception as e:
            rospy.logwarn("cloud_callback error: %s", str(e))

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
            self._rot_mode = False
            self._rot_yaw_target = None
        if path_msg is None or len(path_msg.poses) < 2:
            self.path_pts = []
            self.seg_lens = []
            self.cum_len = [0.0]
            self.s_total = 0.0
            self.s_cur = 0.0
            return
        self._rebuild_path_geometry()
        self._sync_progress_to_current_pose()

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
    def _project_to_path(self, x, y):
        """Return (s_proj, lateral_err, seg_idx, t) where s is arc-length along path."""
        if len(self.path_pts) < 2:
            return 0.0, 0.0, 0, 0.0
        best_s = 0.0
        best_d2 = 1e18
        best_i = 0
        best_t = 0.0
        best_px = self.path_pts[0][0]
        best_py = self.path_pts[0][1]
        for i in range(len(self.path_pts) - 1):
            x0, y0 = self.path_pts[i]
            x1, y1 = self.path_pts[i + 1]
            vx, vy = x1 - x0, y1 - y0
            denom = vx * vx + vy * vy
            if denom < 1e-12:
                t = 0.0
                px, py = x0, y0
            else:
                t = ((x - x0) * vx + (y - y0) * vy) / denom
                t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
                px, py = x0 + t * vx, y0 + t * vy
            d2 = (x - px) ** 2 + (y - py) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
                best_t = t
                best_px = px
                best_py = py
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

    def _update_progress_and_target(self, pose_x, pose_y, yaw):
        if not self.path_pts:
            return None, None, None, None, False, None, None

        s_proj, lat_err, idx, t = self._project_to_path(pose_x, pose_y)

        # enforce monotonic progress with tiny back jitter allowed
        if s_proj + self.back_jitter_m >= self.s_cur:
            self.s_cur = max(self.s_cur, s_proj)

        base_s = max(self.s_cur, s_proj)

        # Off-path recovery should still look forward along the path, otherwise
        # the target jumps to the perpendicular foot-point and causes zig-zag.
        if abs(lat_err) > self.snap_lat_err:
            s_target = min(self.s_total, base_s + self.snap_target_ahead_m)
        else:
            s_target = min(self.s_total, base_s + self.lookahead_distance)

        tx, ty, t_hat = self._interp_xy_tangent_at_s(s_target)

        # goal metrics
        gx, gy = self.path_pts[-1]
        dist_to_goal = math.hypot(gx - pose_x, gy - pose_y)
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
        target_dx = float(goal_xy[0]) - float(x[0])
        target_dy = float(goal_xy[1]) - float(x[1])
        target_dist = math.hypot(target_dx, target_dy)
        target_point_yaw = path_yaw_raw if target_dist <= 1e-6 else math.atan2(target_dy, target_dx)
        goal_bearing_err = angdiff(target_point_yaw, path_yaw_raw)
        goal_bearing_err = max(
            -self.path_tracking_goal_bearing_cap,
            min(self.path_tracking_goal_bearing_cap, goal_bearing_err),
        )
        goal_heading_weight = self.path_tracking_goal_bearing_gain
        if abs(lat_err) > self.snap_lat_err:
            goal_heading_weight = max(goal_heading_weight, 0.75)
        if remaining_dist <= max(self.lookahead_distance * 1.5, 1.0):
            goal_heading_weight = min(1.0, goal_heading_weight + 0.20)
        cte_correction = math.atan2(
            self.path_tracking_cte_gain * lat_err,
            self.path_tracking_cte_soft_mps + max(0.0, abs(x[3])),
        )
        cte_correction = max(
            -self.path_tracking_cte_yaw_cap,
            min(self.path_tracking_cte_yaw_cap, cte_correction),
        )
        # Blend the preview tangent with the actual lookahead-point bearing so the
        # robot follows the shown path geometry instead of only the path direction.
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
        far_from_goal = remaining_dist > self.path_tracking_stop_distance_m
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
        cmd.linear.x = u[0]
        cmd.angular.z = u[1]
        self.last_cmd = cmd
        self.cmd_vel_pub.publish(cmd)

    def run(self):
        rospy.loginfo(
            "DWA node started | pose=%s global=%s local=%s avoidance=%s active=%s active_mux=%s behavior=%s drivable=%s risk=%s obstacle_avoid=on emergency_stop=%.2fm hard_stop=%.2fm footprint=%.2fm x %.2fm cmd_publish=%.1fHz path_tracking_only=%s crawl=%.2f/%.2f heading_filter=%.2f",
            self.pose_topic,
            self.global_path_topic,
            self.local_path_topic,
            self.avoidance_path_topic,
            self.active_path_topic,
            "on" if self.use_muxed_active_path else "off",
            self.behavior_cmd_topic,
            "on" if self.use_drivable_grid else "off",
            "on" if self.use_dynamic_risk_grid else "off",
            self.emergency_stop_distance,
            self.avoidance_hard_stop_distance,
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

            # emergency stop only (avoidance can proceed unless obstacle is critically close)
            avoidance_can_continue = (
                self.active_path_source == "avoidance"
                and self.front_obstacle_clearance > self.avoidance_hard_stop_distance
            )
            if self.emergency_blocked and not self._rot_mode and not avoidance_can_continue:
                self._log_nav_reason("stop_emergency", "front obstacle stop active", warn=True)
                self.publish_drive([0.0, 0.0])
                rate.sleep()
                continue

            # behavior-layer hard stop
            if self.behavior_stop and not self._rot_mode:
                self._log_nav_reason(
                    "stop_behavior",
                    "reason=%s speed_limit=%.2f" % (self.behavior_reason, self.behavior_speed_limit),
                    warn=True,
                )
                self.publish_drive([0.0, 0.0])
                rate.sleep()
                continue

            if not self.path_pts:
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
