#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import threading
import json
import os
import rospy
from collections import deque

from geometry_msgs.msg import Point, PointStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from visualization_msgs.msg import Marker
from std_msgs.msg import Empty, String


class DrivableAreaBuilder:
    def __init__(self):
        # Core params
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.odom_topic = rospy.get_param("~odom_topic", "/lio_sam/mapping/odometry")

        # Grid/painting
        self.grid_resolution_m = float(rospy.get_param("~grid_resolution_m", 0.20))
        self.auto_seed_from_odom = bool(rospy.get_param("~auto_seed_from_odom", True))
        self.seed_radius_m = float(rospy.get_param("~seed_radius_m", 1.20))
        self.seed_step_m = float(rospy.get_param("~seed_step_m", 0.30))
        # Always keep a narrow guaranteed drivable trail on the actually traversed path.
        self.force_trail_from_odom = bool(rospy.get_param("~force_trail_from_odom", True))
        self.force_trail_radius_m = max(0.05, float(rospy.get_param("~force_trail_radius_m", 0.35)))
        self.robot_width_m = max(0.05, float(rospy.get_param("~robot_width_m", 0.58)))
        self.robot_length_m = max(0.05, float(rospy.get_param("~robot_length_m", 0.612)))
        self.footprint_padding_m = max(0.0, float(rospy.get_param("~footprint_padding_m", 0.05)))
        self.paint_robot_footprint_from_odom = bool(
            rospy.get_param("~paint_robot_footprint_from_odom", True)
        )
        self.footprint_trail_step_m = max(0.02, float(rospy.get_param("~footprint_trail_step_m", 0.10)))
        self.footprint_trail_extra_width_m = max(
            0.0, float(rospy.get_param("~footprint_trail_extra_width_m", 0.00))
        )
        self.footprint_trail_extra_length_m = max(
            0.0, float(rospy.get_param("~footprint_trail_extra_length_m", 0.00))
        )
        self.footprint_trail_enforce_ground_filter = bool(
            rospy.get_param("~footprint_trail_enforce_ground_filter", True)
        )
        self.same_level_fill_from_odom = bool(rospy.get_param("~same_level_fill_from_odom", True))
        self.same_level_fill_lateral_margin_m = max(
            0.0, float(rospy.get_param("~same_level_fill_lateral_margin_m", 0.80))
        )
        self.same_level_fill_longitudinal_margin_m = max(
            0.0, float(rospy.get_param("~same_level_fill_longitudinal_margin_m", 0.30))
        )
        self.same_level_fill_height_tolerance_m = max(
            0.01, float(rospy.get_param("~same_level_fill_height_tolerance_m", 0.10))
        )
        self.same_level_fill_max_cells = max(
            50, int(rospy.get_param("~same_level_fill_max_cells", 1200))
        )
        self.same_level_fill_use_eight_connected = bool(
            rospy.get_param("~same_level_fill_use_eight_connected", True)
        )
        self.edit_brush_radius_m = float(rospy.get_param("~edit_brush_radius_m", 1.00))
        self.height_update_alpha = float(rospy.get_param("~height_update_alpha", 0.35))
        self.marker_z_offset = float(rospy.get_param("~marker_z_offset", 0.03))
        self.marker_height = float(rospy.get_param("~marker_height", 0.08))
        self.marker_color_r = min(1.0, max(0.0, float(rospy.get_param("~marker_color_r", 0.95))))
        self.marker_color_g = min(1.0, max(0.0, float(rospy.get_param("~marker_color_g", 0.20))))
        self.marker_color_b = min(1.0, max(0.0, float(rospy.get_param("~marker_color_b", 0.20))))
        self.marker_color_a = min(1.0, max(0.0, float(rospy.get_param("~marker_color_a", 0.45))))
        # If ground reference temporarily breaks, keep seeding in degraded mode instead of fully stopping.
        self.allow_degraded_seeding_when_ref_missing = bool(
            rospy.get_param("~allow_degraded_seeding_when_ref_missing", True)
        )
        self.manual_edit_enforce_ground_filter = bool(
            rospy.get_param("~manual_edit_enforce_ground_filter", False)
        )

        # Ground risk filter (drop/curb aware)
        self.use_ground_filter = bool(rospy.get_param("~use_ground_filter", True))
        self.ground_cloud_topic = rospy.get_param("~ground_cloud_topic", "/lio_sam/mapping/cloud_registered")
        self.ground_cloud_downsample = max(1, int(rospy.get_param("~ground_cloud_downsample", 4)))
        self.ground_min_points_per_cell = max(1, int(rospy.get_param("~ground_min_points_per_cell", 3)))
        self.ground_window_radius_m = max(0.0, float(rospy.get_param("~ground_window_radius_m", 0.40)))
        self.ground_cell_ttl_s = max(0.2, float(rospy.get_param("~ground_cell_ttl_s", 3.0)))
        # Reject cells with strong high obstacles (e.g., tree trunks, poles) from drivable fill.
        self.use_obstacle_rejection = bool(rospy.get_param("~use_obstacle_rejection", True))
        self.obstacle_height_m = max(0.1, float(rospy.get_param("~obstacle_height_m", 0.45)))
        self.obstacle_min_points_per_cell = max(1, int(rospy.get_param("~obstacle_min_points_per_cell", 3)))
        # Robust per-cell ground height update (avoid "always min-z" bias on ramps/noise)
        self.ground_cell_z_alpha = min(1.0, max(0.01, float(rospy.get_param("~ground_cell_z_alpha", 0.25))))
        self.ground_cell_outlier_down_m = max(
            0.0, float(rospy.get_param("~ground_cell_outlier_down_m", 0.40))
        )
        self.ground_cell_outlier_up_m = max(
            0.0, float(rospy.get_param("~ground_cell_outlier_up_m", 0.60))
        )
        self.ground_min_z_offset = float(rospy.get_param("~ground_min_z_offset", -2.0))
        self.ground_max_z_offset = float(rospy.get_param("~ground_max_z_offset", 0.8))
        self.use_adaptive_ground_reference = bool(rospy.get_param("~use_adaptive_ground_reference", True))
        self.adaptive_reference_strict = bool(rospy.get_param("~adaptive_reference_strict", True))
        self.ground_reference_radius_m = max(0.2, float(rospy.get_param("~ground_reference_radius_m", 1.2)))
        self.ground_reference_percentile = min(
            50.0, max(1.0, float(rospy.get_param("~ground_reference_percentile", 20.0)))
        )
        self.ground_reference_min_points = max(5, int(rospy.get_param("~ground_reference_min_points", 30)))
        self.ground_reference_alpha = min(1.0, max(0.01, float(rospy.get_param("~ground_reference_alpha", 0.25))))
        self.ground_reference_max_jump_m = max(
            0.01, float(rospy.get_param("~ground_reference_max_jump_m", 0.10))
        )
        # LIO-SAM cloud compatible local ground tracker (linefit-style: low-z seed + robust plane fit).
        self.use_linefit_ground_tracker = bool(rospy.get_param("~use_linefit_ground_tracker", True))
        self.linefit_local_radius_m = max(1.0, float(rospy.get_param("~linefit_local_radius_m", 8.0)))
        self.linefit_seed_percentile = min(70.0, max(5.0, float(rospy.get_param("~linefit_seed_percentile", 30.0))))
        self.linefit_inlier_threshold_m = max(0.03, float(rospy.get_param("~linefit_inlier_threshold_m", 0.18)))
        self.linefit_min_points = max(30, int(rospy.get_param("~linefit_min_points", 120)))
        self.linefit_max_tilt_deg = max(1.0, float(rospy.get_param("~linefit_max_tilt_deg", 35.0)))
        self.linefit_smooth_alpha = min(1.0, max(0.01, float(rospy.get_param("~linefit_smooth_alpha", 0.35))))
        self.linefit_hold_sec = max(0.0, float(rospy.get_param("~linefit_hold_sec", 2.5)))
        # Local slope model from tracked ground reference history (for ramps/downhills)
        self.use_local_slope_model = bool(rospy.get_param("~use_local_slope_model", True))
        self.slope_model_history_size = max(10, int(rospy.get_param("~slope_model_history_size", 80)))
        self.slope_model_min_samples = max(6, int(rospy.get_param("~slope_model_min_samples", 12)))
        self.slope_model_max_tilt_deg = max(1.0, float(rospy.get_param("~slope_model_max_tilt_deg", 30.0)))
        # Optional prior: lidar is mounted above ground (helps reject wrong-floor/outlier ground picks)
        self.use_lidar_height_prior = bool(rospy.get_param("~use_lidar_height_prior", True))
        self.lidar_height_m = max(0.0, float(rospy.get_param("~lidar_height_m", 0.525)))
        self.lidar_height_tolerance_m = max(0.01, float(rospy.get_param("~lidar_height_tolerance_m", 0.20)))
        self.lidar_prior_mode = str(rospy.get_param("~lidar_prior_mode", "clamp")).strip().lower()
        if self.lidar_prior_mode not in ("clamp", "reject"):
            rospy.logwarn("Unknown lidar_prior_mode='%s', fallback to 'clamp'", self.lidar_prior_mode)
            self.lidar_prior_mode = "clamp"
        self.max_drop_m = max(0.0, float(rospy.get_param("~max_drop_m", 0.22)))
        self.max_step_up_m = max(0.0, float(rospy.get_param("~max_step_up_m", 0.16)))
        self.max_neighbor_height_diff_m = max(
            0.0, float(rospy.get_param("~max_neighbor_height_diff_m", 0.20))
        )
        self.allow_unknown_ground = bool(rospy.get_param("~allow_unknown_ground", False))
        # If true, mark risk only for confirmed hazards (drop/step/neighbor jump),
        # not for unknown/insufficient-support cells.
        self.risk_require_confirmed_hazard = bool(rospy.get_param("~risk_require_confirmed_hazard", True))
        # If false, keep using risk logic internally but do not accumulate forbidden/risk cells.
        self.track_risk_cells = bool(rospy.get_param("~track_risk_cells", False))
        self.risk_marker_topic = rospy.get_param("~risk_marker_topic", "/lio_sam/drivable_area/risk_marker")
        self.enable_risk_marker = bool(rospy.get_param("~publish_risk_marker", True))
        self.risk_max_cells = max(1000, int(rospy.get_param("~risk_max_cells", 20000)))

        # Editing topics
        self.add_point_topic = rospy.get_param("~add_point_topic", "/lio_sam/drivable_area/add_point")
        self.erase_point_topic = rospy.get_param("~erase_point_topic", "/lio_sam/drivable_area/erase_point")
        self.clear_topic = rospy.get_param("~clear_topic", "/lio_sam/drivable_area/clear")
        self.undo_topic = rospy.get_param("~undo_topic", "/lio_sam/drivable_area/undo")
        self.mode_topic = rospy.get_param("~mode_topic", "/lio_sam/drivable_area/mode")
        self.use_clicked_point = bool(rospy.get_param("~use_clicked_point", False))
        self.clicked_point_topic = rospy.get_param("~clicked_point_topic", "/clicked_point")
        self.clicked_point_mode = str(rospy.get_param("~clicked_point_mode", "erase")).strip().lower()
        if self.clicked_point_mode not in ("add", "erase", "toggle"):
            rospy.logwarn("Unknown clicked_point_mode='%s', fallback to 'erase'", self.clicked_point_mode)
            self.clicked_point_mode = "erase"
        self.max_history = max(1, int(rospy.get_param("~max_history", 30)))
        self.save_topic = rospy.get_param("~save_topic", "/lio_sam/drivable_area/save")
        self.load_topic = rospy.get_param("~load_topic", "/lio_sam/drivable_area/load")
        self.read_only_mode = bool(rospy.get_param("~read_only_mode", False))

        # Persistence
        self.state_file_path = os.path.expanduser(
            rospy.get_param("~state_file_path", "~/.ros/lio_sam_drivable_area_state.json")
        )
        self.auto_load_state = bool(rospy.get_param("~auto_load_state", True))
        self.auto_save_state = bool(rospy.get_param("~auto_save_state", True))
        self.save_on_shutdown = bool(rospy.get_param("~save_on_shutdown", True))
        self.auto_save_period_s = max(0.2, float(rospy.get_param("~auto_save_period_s", 2.0)))
        self.allow_resolution_mismatch = bool(rospy.get_param("~allow_resolution_mismatch", False))
        self.persist_on_risk_only_changes = bool(rospy.get_param("~persist_on_risk_only_changes", False))

        # Visualization / publishing
        self.marker_topic = rospy.get_param("~marker_topic", "/lio_sam/drivable_area/marker")
        self.grid_topic = rospy.get_param("~grid_topic", "/lio_sam/drivable_area/grid")
        self.marker_max_cells = max(1000, int(rospy.get_param("~marker_max_cells", 40000)))
        self.publish_period_s = max(0.1, float(rospy.get_param("~publish_period_s", 0.5)))

        self._cells = set()
        self._cell_z = {}
        self._last_seed_xy = None
        self._last_odom_xy = None
        self._last_odom_trail_pose = None
        self._last_odom_z = 0.0
        self._adaptive_ground_z = 0.0
        self._adaptive_ground_valid = False
        self._linefit_plane_abc = None
        self._linefit_last_update_sec = 0.0
        self._ground_track = []
        self._slope_plane_abc = None
        self._dirty = True
        self._history = []
        self._risk_cells = set()
        self._ground_min_z = {}  # Stores robust ground z estimate per cell (kept name for compatibility)
        self._ground_count = {}
        self._ground_stamp = {}
        self._obstacle_count = {}
        self._obstacle_stamp = {}
        self._lock = threading.RLock()
        self._changed_since_persist = False

        self.pub_marker = rospy.Publisher(self.marker_topic, Marker, queue_size=1, latch=True)
        self.pub_grid = rospy.Publisher(self.grid_topic, OccupancyGrid, queue_size=1, latch=True)
        self.pub_risk_marker = None
        if self.enable_risk_marker:
            self.pub_risk_marker = rospy.Publisher(self.risk_marker_topic, Marker, queue_size=1, latch=True)

        self.sub_odom = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=50)
        self.sub_ground_cloud = None
        if self.use_ground_filter:
            self.sub_ground_cloud = rospy.Subscriber(
                self.ground_cloud_topic, PointCloud2, self.ground_cloud_callback, queue_size=1
            )
        self.sub_add = rospy.Subscriber(self.add_point_topic, PointStamped, self.add_point_callback, queue_size=20)
        self.sub_erase = rospy.Subscriber(self.erase_point_topic, PointStamped, self.erase_point_callback, queue_size=20)
        self.sub_clear = rospy.Subscriber(self.clear_topic, Empty, self.clear_callback, queue_size=2)
        self.sub_undo = rospy.Subscriber(self.undo_topic, Empty, self.undo_callback, queue_size=2)
        self.sub_save = rospy.Subscriber(self.save_topic, Empty, self.save_callback, queue_size=2)
        self.sub_load = rospy.Subscriber(self.load_topic, Empty, self.load_callback, queue_size=2)
        self.sub_mode = rospy.Subscriber(self.mode_topic, String, self.mode_callback, queue_size=5)
        self.sub_click = None
        if self.use_clicked_point:
            self.sub_click = rospy.Subscriber(self.clicked_point_topic, PointStamped, self.clicked_point_callback, queue_size=20)

        rospy.Timer(rospy.Duration(self.publish_period_s), self.on_timer)
        self.auto_save_timer = None
        if self.auto_save_state:
            self.auto_save_timer = rospy.Timer(rospy.Duration(self.auto_save_period_s), self.on_auto_save_timer)
        rospy.on_shutdown(self.on_shutdown)

        if self.auto_load_state:
            self.load_state_from_file(log_prefix="startup")

        rospy.loginfo(
            "drivable_area_builder started | odom=%s, mode=%s, auto_seed=%s, ground_filter=%s, adaptive_ref=%s, grid=%.2fm, state=%s",
            self.odom_topic,
            "read_only" if self.read_only_mode else "editable",
            "on" if self.auto_seed_from_odom else "off",
            "on" if self.use_ground_filter else "off",
            "on" if self.use_adaptive_ground_reference else "off",
            self.grid_resolution_m,
            self.state_file_path,
        )
        rospy.loginfo(
            "drivable_area_builder slope model | enabled=%s, history=%d, min_samples=%d, max_tilt=%.1fdeg",
            "on" if self.use_local_slope_model else "off",
            self.slope_model_history_size,
            self.slope_model_min_samples,
            self.slope_model_max_tilt_deg,
        )
        rospy.loginfo(
            "drivable_area_builder linefit tracker | enabled=%s, radius=%.1fm, seed_p=%.1f, inlier=%.2fm",
            "on" if self.use_linefit_ground_tracker else "off",
            self.linefit_local_radius_m,
            self.linefit_seed_percentile,
            self.linefit_inlier_threshold_m,
        )
        rospy.loginfo(
            "drivable_area_builder height prior | enabled=%s, lidar_height=%.3fm, tol=%.3fm, mode=%s",
            "on" if self.use_lidar_height_prior else "off",
            self.lidar_height_m,
            self.lidar_height_tolerance_m,
            self.lidar_prior_mode,
        )
        rospy.loginfo(
            "drivable_area_builder footprint trail | enabled=%s, footprint=%.2fm x %.2fm, padding=%.2fm, step=%.2fm, safe_only=%s",
            "on" if self.paint_robot_footprint_from_odom else "off",
            self.robot_length_m,
            self.robot_width_m,
            self.footprint_padding_m,
            self.footprint_trail_step_m,
            "on" if self.footprint_trail_enforce_ground_filter else "off",
        )
        rospy.loginfo(
            "drivable_area_builder same-level fill | enabled=%s, lateral=%.2fm, longitudinal=%.2fm, dz=%.2fm, max_cells=%d",
            "on" if self.same_level_fill_from_odom else "off",
            self.same_level_fill_lateral_margin_m,
            self.same_level_fill_longitudinal_margin_m,
            self.same_level_fill_height_tolerance_m,
            self.same_level_fill_max_cells,
        )

    def xy_to_key(self, x, y):
        r = self.grid_resolution_m
        return int(math.floor(x / r)), int(math.floor(y / r))

    def key_to_center(self, ix, iy):
        r = self.grid_resolution_m
        return (ix + 0.5) * r, (iy + 0.5) * r

    def _mutations_allowed(self):
        return not self.read_only_mode

    def _prune_ground_cache(self, now_sec):
        ttl = self.ground_cell_ttl_s
        stale = [k for k, ts in list(self._ground_stamp.items()) if (now_sec - ts) > ttl]
        for k in stale:
            self._ground_stamp.pop(k, None)
            self._ground_min_z.pop(k, None)
            self._ground_count.pop(k, None)
        stale_obs = [k for k, ts in list(self._obstacle_stamp.items()) if (now_sec - ts) > ttl]
        for k in stale_obs:
            self._obstacle_stamp.pop(k, None)
            self._obstacle_count.pop(k, None)

    @staticmethod
    def _percentile(values, p):
        if not values:
            return None
        values_sorted = sorted(values)
        if len(values_sorted) == 1:
            return values_sorted[0]
        rank = (p / 100.0) * (len(values_sorted) - 1)
        i0 = int(math.floor(rank))
        i1 = min(len(values_sorted) - 1, i0 + 1)
        t = rank - float(i0)
        return (1.0 - t) * values_sorted[i0] + t * values_sorted[i1]

    def _get_ground_reference_z_locked(self, fallback_z):
        if self.use_adaptive_ground_reference and self._adaptive_ground_valid:
            return self._adaptive_ground_z
        if self.use_adaptive_ground_reference and self.adaptive_reference_strict:
            return None
        return fallback_z

    def _seed_reference_ground_z_locked(self, fallback_z):
        ref_z = self._get_ground_reference_z_locked(fallback_z)
        if ref_z is None:
            return None
        if self.use_adaptive_ground_reference and self._adaptive_ground_valid:
            return ref_z
        return self._apply_lidar_height_prior_locked(ref_z)

    def _apply_lidar_height_prior_locked(self, ground_z):
        if ground_z is None:
            return None
        if not self.use_lidar_height_prior:
            return float(ground_z)
        # Prefer continuity from tracked ground reference over raw odom z.
        # This avoids getting stuck on ramps when odom z is biased/drifting.
        if self._adaptive_ground_valid:
            expected_ground_z = self._adaptive_ground_z
        else:
            expected_ground_z = self._last_odom_z - self.lidar_height_m
        err = float(ground_z) - expected_ground_z
        if abs(err) <= self.lidar_height_tolerance_m:
            return float(ground_z)
        if self.lidar_prior_mode == "reject":
            return None
        return expected_ground_z + math.copysign(self.lidar_height_tolerance_m, err)

    @staticmethod
    def _solve_3x3(a, b):
        # Small dense solve for plane fitting; returns None if singular.
        m = [
            [float(a[0][0]), float(a[0][1]), float(a[0][2]), float(b[0])],
            [float(a[1][0]), float(a[1][1]), float(a[1][2]), float(b[1])],
            [float(a[2][0]), float(a[2][1]), float(a[2][2]), float(b[2])],
        ]
        for i in range(3):
            piv = i
            for r in range(i + 1, 3):
                if abs(m[r][i]) > abs(m[piv][i]):
                    piv = r
            if abs(m[piv][i]) < 1e-9:
                return None
            if piv != i:
                m[i], m[piv] = m[piv], m[i]
            div = m[i][i]
            for c in range(i, 4):
                m[i][c] /= div
            for r in range(3):
                if r == i:
                    continue
                f = m[r][i]
                if abs(f) < 1e-12:
                    continue
                for c in range(i, 4):
                    m[r][c] -= f * m[i][c]
        return (m[0][3], m[1][3], m[2][3])

    def _fit_plane_least_squares(self, points):
        if len(points) < 3:
            return None
        sx = sy = sz = sxx = syy = sxy = sxz = syz = 0.0
        n = float(len(points))
        for x, y, z in points:
            sx += x
            sy += y
            sz += z
            sxx += x * x
            syy += y * y
            sxy += x * y
            sxz += x * z
            syz += y * z
        return self._solve_3x3(
            (
                (sxx, sxy, sx),
                (sxy, syy, sy),
                (sx, sy, n),
            ),
            (sxz, syz, sz),
        )

    def _linefit_ground_update_locked(self, sampled_points, now_sec):
        if (not self.use_linefit_ground_tracker) or self._last_odom_xy is None:
            return False
        rx, ry = self._last_odom_xy
        rr = self.linefit_local_radius_m * self.linefit_local_radius_m
        local_points = []
        z_vals = []
        for x, y, z in sampled_points:
            dx = x - rx
            dy = y - ry
            if (dx * dx + dy * dy) <= rr:
                local_points.append((x, y, z))
                z_vals.append(z)
        if len(local_points) < self.linefit_min_points:
            return False

        z_th = self._percentile(z_vals, self.linefit_seed_percentile)
        if z_th is None:
            return False
        seeds = [p for p in local_points if p[2] <= z_th]
        if len(seeds) < max(12, int(0.2 * self.linefit_min_points)):
            return False

        coeff = self._fit_plane_least_squares(seeds)
        if coeff is None:
            return False
        a, b, c = coeff
        tilt_deg = math.degrees(math.atan(math.hypot(a, b)))
        if tilt_deg > self.linefit_max_tilt_deg:
            return False

        inliers = []
        th = self.linefit_inlier_threshold_m
        for x, y, z in local_points:
            z_est = (a * x) + (b * y) + c
            if abs(z - z_est) <= th:
                inliers.append((x, y, z))
        if len(inliers) < max(20, int(0.25 * self.linefit_min_points)):
            return False

        coeff2 = self._fit_plane_least_squares(inliers)
        if coeff2 is None:
            return False
        a2, b2, c2 = coeff2
        tilt2_deg = math.degrees(math.atan(math.hypot(a2, b2)))
        if tilt2_deg > self.linefit_max_tilt_deg:
            return False

        if self._linefit_plane_abc is not None:
            pa, pb, pc = self._linefit_plane_abc
            alpha = self.linefit_smooth_alpha
            a2 = (1.0 - alpha) * pa + alpha * a2
            b2 = (1.0 - alpha) * pb + alpha * b2
            c2 = (1.0 - alpha) * pc + alpha * c2

        self._linefit_plane_abc = (float(a2), float(b2), float(c2))
        self._linefit_last_update_sec = float(now_sec)
        self._slope_plane_abc = self._linefit_plane_abc
        self._adaptive_ground_z = (a2 * rx) + (b2 * ry) + c2
        self._adaptive_ground_valid = True
        return True

    def _update_slope_model_locked(self, ref_z):
        if (not self.use_local_slope_model) or self._last_odom_xy is None or ref_z is None:
            return
        x, y = self._last_odom_xy
        self._ground_track.append((float(x), float(y), float(ref_z)))
        if len(self._ground_track) > self.slope_model_history_size:
            self._ground_track.pop(0)
        if len(self._ground_track) < self.slope_model_min_samples:
            return

        n = float(len(self._ground_track))
        sx = sy = sz = sxx = syy = sxy = sxz = syz = 0.0
        for gx, gy, gz in self._ground_track:
            sx += gx
            sy += gy
            sz += gz
            sxx += gx * gx
            syy += gy * gy
            sxy += gx * gy
            sxz += gx * gz
            syz += gy * gz
        coeff = self._solve_3x3(
            (
                (sxx, sxy, sx),
                (sxy, syy, sy),
                (sx, sy, n),
            ),
            (sxz, syz, sz),
        )
        if coeff is None:
            return
        a, b, c = coeff
        tilt_deg = math.degrees(math.atan(math.hypot(a, b)))
        if tilt_deg > self.slope_model_max_tilt_deg:
            return
        self._slope_plane_abc = (float(a), float(b), float(c))

    def _reference_z_at_xy_locked(self, x, y, fallback_ref_z):
        if self._linefit_plane_abc is not None:
            if self.linefit_hold_sec <= 0.0:
                a, b, c = self._linefit_plane_abc
                return (a * float(x)) + (b * float(y)) + c
            age = rospy.Time.now().to_sec() - self._linefit_last_update_sec
            if age <= self.linefit_hold_sec:
                a, b, c = self._linefit_plane_abc
                return (a * float(x)) + (b * float(y)) + c
        if self._slope_plane_abc is None:
            return fallback_ref_z
        a, b, c = self._slope_plane_abc
        return (a * float(x)) + (b * float(y)) + c

    def _fallback_ground_z_at_xy_locked(self, x, y, odom_z):
        ref_z = self._get_ground_reference_z_locked(odom_z)
        if ref_z is None:
            if self.use_lidar_height_prior:
                ref_z = float(odom_z) - self.lidar_height_m
            else:
                ref_z = float(odom_z)
        return self._reference_z_at_xy_locked(x, y, ref_z)

    def _update_adaptive_ground_reference_locked(self, sampled_points):
        if not self.use_adaptive_ground_reference:
            return
        if self._last_odom_xy is None:
            return

        rx, ry = self._last_odom_xy
        rr = self.ground_reference_radius_m * self.ground_reference_radius_m
        near_z = []
        for x, y, z in sampled_points:
            dx = x - rx
            dy = y - ry
            if (dx * dx + dy * dy) <= rr:
                near_z.append(z)

        if len(near_z) < self.ground_reference_min_points:
            return

        cand = self._percentile(near_z, self.ground_reference_percentile)
        if cand is None:
            return
        cand = self._apply_lidar_height_prior_locked(cand)
        if cand is None:
            return

        if not self._adaptive_ground_valid:
            self._adaptive_ground_z = float(cand)
            self._adaptive_ground_valid = True
            return

        prev = self._adaptive_ground_z
        delta = float(cand) - prev
        if abs(delta) > self.ground_reference_max_jump_m:
            cand = prev + math.copysign(self.ground_reference_max_jump_m, delta)
        self._adaptive_ground_z = (1.0 - self.ground_reference_alpha) * prev + self.ground_reference_alpha * float(cand)
        self._adaptive_ground_valid = True

    def _bootstrap_adaptive_ground_reference_locked(self, sampled_points):
        if not self.use_adaptive_ground_reference:
            return
        if self._adaptive_ground_valid:
            return
        if len(sampled_points) < self.ground_reference_min_points:
            return
        z_vals = [p[2] for p in sampled_points]
        cand = self._percentile(z_vals, self.ground_reference_percentile)
        if cand is None:
            return
        cand = self._apply_lidar_height_prior_locked(cand)
        if cand is None:
            return
        self._adaptive_ground_z = float(cand)
        self._adaptive_ground_valid = True

    def ground_cloud_callback(self, msg):
        if not self.use_ground_filter:
            return
        try:
            with self._lock:
                now_sec = rospy.Time.now().to_sec()
                self._prune_ground_cache(now_sec)
                i = 0
                sampled_points = []
                for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                    i += 1
                    if self.ground_cloud_downsample > 1 and (i % self.ground_cloud_downsample != 0):
                        continue
                    x, y, z = float(p[0]), float(p[1]), float(p[2])
                    sampled_points.append((x, y, z))

                linefit_ok = self._linefit_ground_update_locked(sampled_points, now_sec)
                if not linefit_ok:
                    self._update_adaptive_ground_reference_locked(sampled_points)
                    self._bootstrap_adaptive_ground_reference_locked(sampled_points)
                ref_z = self._get_ground_reference_z_locked(self._last_odom_z)
                if ref_z is None:
                    # Keep cache update alive with degraded reference rather than hard stop.
                    ref_z = self._last_odom_z - self.lidar_height_m
                if not linefit_ok:
                    self._update_slope_model_locked(ref_z)

                for x, y, z in sampled_points:
                    local_ref_z = self._reference_z_at_xy_locked(x, y, ref_z)
                    key = self.xy_to_key(x, y)
                    if self.use_obstacle_rejection and z > (local_ref_z + self.obstacle_height_m):
                        self._obstacle_count[key] = min(1000, self._obstacle_count.get(key, 0) + 1)
                        self._obstacle_stamp[key] = now_sec
                        continue
                    min_z = local_ref_z + self.ground_min_z_offset
                    max_z = local_ref_z + self.ground_max_z_offset
                    if z < min_z or z > max_z:
                        continue
                    prev_z = self._ground_min_z.get(key, None)
                    if prev_z is None:
                        self._ground_min_z[key] = z
                    else:
                        dz = z - prev_z
                        if dz < -self.ground_cell_outlier_down_m or dz > self.ground_cell_outlier_up_m:
                            # Reject extreme jumps to avoid false curb/drop marks from outliers.
                            continue
                        self._ground_min_z[key] = (1.0 - self.ground_cell_z_alpha) * prev_z + self.ground_cell_z_alpha * z
                    self._ground_count[key] = min(1000, self._ground_count.get(key, 0) + 1)
                    self._ground_stamp[key] = now_sec
        except Exception as e:
            rospy.logwarn_throttle(1.0, "ground_cloud_callback error: %s", str(e))

    def _estimate_ground(self, key):
        if not self.use_ground_filter:
            return True, self._last_odom_z, 1

        with self._lock:
            now_sec = rospy.Time.now().to_sec()
            self._prune_ground_cache(now_sec)

            ix, iy = key
            n = int(math.ceil(self.ground_window_radius_m / self.grid_resolution_m))
            if n < 0:
                n = 0

            z_wsum = 0.0
            wsum = 0.0
            cnt = 0
            for di in range(-n, n + 1):
                for dj in range(-n, n + 1):
                    nk = (ix + di, iy + dj)
                    z = self._ground_min_z.get(nk, None)
                    if z is None:
                        continue
                    c = self._ground_count.get(nk, 0)
                    if c <= 0:
                        continue
                    d = math.hypot(float(di), float(dj))
                    w = 1.0 / (1.0 + d)
                    z_wsum += w * z
                    wsum += w
                    cnt += c
            if wsum <= 0.0:
                return False, None, 0
            return True, (z_wsum / wsum), cnt

    def _is_key_ground_safe(self, key, ref_z):
        ok, z_ground, support = self._estimate_ground(key)
        if not ok:
            if self.allow_unknown_ground:
                return True, None, "unknown_allowed"
            return False, None, "unknown"
        if support < self.ground_min_points_per_cell:
            return False, z_ground, "low_support"
        if self.use_obstacle_rejection:
            with self._lock:
                if self._obstacle_count.get(key, 0) >= self.obstacle_min_points_per_cell:
                    return False, z_ground, "obstacle_high"

        x, y = self.key_to_center(key[0], key[1])
        local_ref_z = self._reference_z_at_xy_locked(x, y, ref_z)
        dz = z_ground - local_ref_z
        if dz < -self.max_drop_m:
            return False, z_ground, "drop"
        if dz > self.max_step_up_m:
            return False, z_ground, "step"

        if self.max_neighbor_height_diff_m > 0.0:
            with self._lock:
                for nk in ((key[0] + 1, key[1]), (key[0] - 1, key[1]), (key[0], key[1] + 1), (key[0], key[1] - 1)):
                    nz = self._ground_min_z.get(nk, None)
                    if nz is None:
                        continue
                    if abs(z_ground - nz) > self.max_neighbor_height_diff_m:
                        return False, z_ground, "neighbor_jump"
        return True, z_ground, "ok"

    def _push_history(self, item):
        self._history.append(item)
        if len(self._history) > self.max_history:
            self._history.pop(0)

    def _record_delta(self, changes):
        if changes:
            self._push_history(("delta", changes))

    @staticmethod
    def _yaw_from_quaternion(q):
        return math.atan2(
            2.0 * ((q.w * q.z) + (q.x * q.y)),
            1.0 - (2.0 * ((q.y * q.y) + (q.z * q.z))),
        )

    @staticmethod
    def _interp_angle(a0, a1, t):
        da = math.atan2(math.sin(a1 - a0), math.cos(a1 - a0))
        return a0 + (float(t) * da)

    def _paint_key(self, key, z, allow):
        # Caller must hold self._lock.
        before_exists = key in self._cells
        before_z = self._cell_z.get(key, 0.0)

        changed = False
        if allow:
            if not before_exists:
                self._cells.add(key)
                self._cell_z[key] = z
                changed = True
            else:
                # Keep surface smooth while still following slope changes.
                new_z = (1.0 - self.height_update_alpha) * before_z + self.height_update_alpha * z
                if abs(new_z - before_z) > 1e-4:
                    self._cell_z[key] = new_z
                    changed = True
        else:
            if before_exists:
                self._cells.remove(key)
                self._cell_z.pop(key, None)
                changed = True

        if not changed:
            return None

        after_exists = key in self._cells
        after_z = self._cell_z.get(key, 0.0)
        if after_exists:
            self._risk_cells.discard(key)
        return (key, before_exists, before_z, after_exists, after_z)

    def _paint_keys(self, keys, cz, allow=True, record_history=False, enforce_ground_filter=True):
        key_list = sorted(set(keys))
        if not key_list:
            return False

        with self._lock:
            changed = False
            risk_changed = False
            changes = []
            ref_z = self._seed_reference_ground_z_locked(cz)
            use_ground_filter = (enforce_ground_filter and self.use_ground_filter)
            if allow and use_ground_filter and ref_z is None:
                if self.allow_degraded_seeding_when_ref_missing:
                    # Degraded fallback: keep area growth alive until ground reference recovers.
                    use_ground_filter = False
                else:
                    return False

            for key in key_list:
                ix, iy = key
                x, y = self.key_to_center(ix, iy)
                if allow and use_ground_filter:
                    safe, gz, reason = self._is_key_ground_safe(key, ref_z)
                    if not safe:
                        if self.track_risk_cells:
                            should_mark_risk = True
                            if self.risk_require_confirmed_hazard:
                                should_mark_risk = reason in ("drop", "step", "neighbor_jump")
                            if should_mark_risk:
                                before_risk = key in self._risk_cells
                                self._risk_cells.add(key)
                                if not before_risk:
                                    risk_changed = True
                            else:
                                if key in self._risk_cells:
                                    self._risk_cells.discard(key)
                                    risk_changed = True
                        elif key in self._risk_cells:
                            self._risk_cells.discard(key)
                            risk_changed = True
                        continue
                    if key in self._risk_cells:
                        self._risk_cells.discard(key)
                        risk_changed = True
                    if gz is not None:
                        z = gz
                    else:
                        z = ref_z
                else:
                    if allow:
                        z = self._fallback_ground_z_at_xy_locked(x, y, cz)
                    else:
                        z = cz
                rec = self._paint_key(key, z, allow)
                if rec is not None:
                    changed = True
                    if record_history:
                        changes.append(rec)

            if changed or risk_changed:
                self._dirty = True
                if changed or (risk_changed and self.persist_on_risk_only_changes):
                    self._changed_since_persist = True
                if record_history and changed:
                    self._record_delta(changes)
            return changed

    def paint_circle(self, cx, cy, cz, radius_m, allow=True, record_history=False, enforce_ground_filter=True):
        if radius_m <= 0.0:
            return False
        r = self.grid_resolution_m
        ic, jc = self.xy_to_key(cx, cy)
        n = int(math.ceil(radius_m / r))
        rr = radius_m * radius_m
        keys = []
        for di in range(-n, n + 1):
            for dj in range(-n, n + 1):
                ix = ic + di
                iy = jc + dj
                x, y = self.key_to_center(ix, iy)
                if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= rr:
                    keys.append((ix, iy))
        return self._paint_keys(
            keys,
            cz,
            allow=allow,
            record_history=record_history,
            enforce_ground_filter=enforce_ground_filter,
        )

    def _footprint_keys(self, cx, cy, yaw, half_length_m, half_width_m):
        if half_length_m <= 0.0 or half_width_m <= 0.0:
            return []
        raster_margin = 0.5 * self.grid_resolution_m
        half_length = half_length_m + raster_margin
        half_width = half_width_m + raster_margin
        c = math.cos(yaw)
        s = math.sin(yaw)
        bbox_x = abs(c) * half_length + abs(s) * half_width
        bbox_y = abs(s) * half_length + abs(c) * half_width
        min_ix, min_iy = self.xy_to_key(cx - bbox_x, cy - bbox_y)
        max_ix, max_iy = self.xy_to_key(cx + bbox_x, cy + bbox_y)
        keys = []
        for ix in range(min_ix, max_ix + 1):
            for iy in range(min_iy, max_iy + 1):
                x, y = self.key_to_center(ix, iy)
                dx = x - cx
                dy = y - cy
                local_x = c * dx + s * dy
                local_y = (-s * dx) + (c * dy)
                if abs(local_x) <= half_length and abs(local_y) <= half_width:
                    keys.append((ix, iy))
        return keys

    def _key_within_oriented_box(self, key, cx, cy, yaw, half_length_m, half_width_m):
        x, y = self.key_to_center(key[0], key[1])
        dx = x - cx
        dy = y - cy
        c = math.cos(yaw)
        s = math.sin(yaw)
        local_x = c * dx + s * dy
        local_y = (-s * dx) + (c * dy)
        return abs(local_x) <= half_length_m and abs(local_y) <= half_width_m

    def paint_footprint(
        self,
        cx,
        cy,
        cz,
        yaw,
        allow=True,
        record_history=False,
        enforce_ground_filter=True,
        half_length_m=None,
        half_width_m=None,
    ):
        if half_length_m is None:
            half_length_m = 0.5 * self.robot_length_m + self.footprint_padding_m + self.footprint_trail_extra_length_m
        if half_width_m is None:
            half_width_m = 0.5 * self.robot_width_m + self.footprint_padding_m + self.footprint_trail_extra_width_m
        keys = self._footprint_keys(cx, cy, yaw, half_length_m, half_width_m)
        return self._paint_keys(
            keys,
            cz,
            allow=allow,
            record_history=record_history,
            enforce_ground_filter=enforce_ground_filter,
        )

    def paint_swept_footprint(self, start_pose, end_pose, allow=True, record_history=False, enforce_ground_filter=True):
        if start_pose is None or end_pose is None:
            return False
        sx, sy, sz, syaw = start_pose
        ex, ey, ez, eyaw = end_pose
        dist = math.hypot(ex - sx, ey - sy)
        steps = max(1, int(math.ceil(dist / self.footprint_trail_step_m)))
        changed = False
        for idx in range(steps + 1):
            t = float(idx) / float(steps)
            px = sx + ((ex - sx) * t)
            py = sy + ((ey - sy) * t)
            pz = sz + ((ez - sz) * t)
            pyaw = self._interp_angle(syaw, eyaw, t)
            changed = self.paint_footprint(
                px,
                py,
                pz,
                pyaw,
                allow=allow,
                record_history=record_history,
                enforce_ground_filter=enforce_ground_filter,
            ) or changed
        return changed

    def paint_same_level_local_fill(self, cx, cy, cz, yaw):
        if (not self.same_level_fill_from_odom) or (not self.use_ground_filter):
            return False

        with self._lock:
            ref_z = self._seed_reference_ground_z_locked(cz)
        if ref_z is None:
            return False

        seed_half_length = 0.5 * self.robot_length_m + self.footprint_padding_m + self.footprint_trail_extra_length_m
        seed_half_width = 0.5 * self.robot_width_m + self.footprint_padding_m + self.footprint_trail_extra_width_m
        envelope_half_length = seed_half_length + self.same_level_fill_longitudinal_margin_m
        envelope_half_width = seed_half_width + self.same_level_fill_lateral_margin_m
        seed_keys = self._footprint_keys(cx, cy, yaw, seed_half_length, seed_half_width)
        if not seed_keys:
            return False

        target_z_samples = []
        accepted = set()
        q = deque()
        for key in seed_keys:
            if key in accepted:
                continue
            safe, gz, _reason = self._is_key_ground_safe(key, ref_z)
            if not safe:
                continue
            accepted.add(key)
            q.append(key)
            if gz is not None:
                target_z_samples.append(float(gz))

        if not accepted:
            return False

        target_z = self._percentile(target_z_samples, 50.0)
        if target_z is None:
            with self._lock:
                target_z = self._reference_z_at_xy_locked(cx, cy, ref_z)
        if target_z is None:
            return False

        visited = set(accepted)
        if self.same_level_fill_use_eight_connected:
            neighbor_offsets = (
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1),
            )
        else:
            neighbor_offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))

        while q and len(accepted) < self.same_level_fill_max_cells:
            key = q.popleft()
            for dx, dy in neighbor_offsets:
                nk = (key[0] + dx, key[1] + dy)
                if nk in visited:
                    continue
                visited.add(nk)
                if not self._key_within_oriented_box(
                    nk, cx, cy, yaw, envelope_half_length, envelope_half_width
                ):
                    continue
                safe, gz, _reason = self._is_key_ground_safe(nk, ref_z)
                if (not safe) or (gz is None):
                    continue
                if abs(float(gz) - float(target_z)) > self.same_level_fill_height_tolerance_m:
                    continue
                accepted.add(nk)
                q.append(nk)
                if len(accepted) >= self.same_level_fill_max_cells:
                    break

        return self._paint_keys(
            accepted,
            cz,
            allow=True,
            record_history=False,
            enforce_ground_filter=True,
        )

    def odom_callback(self, msg):
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        z = float(msg.pose.pose.position.z)
        yaw = self._yaw_from_quaternion(msg.pose.pose.orientation)

        with self._lock:
            prev_trail_pose = self._last_odom_trail_pose
            self._last_odom_z = z
            self._last_odom_xy = (x, y)
            last_seed_xy = self._last_seed_xy
            trail_z = self._fallback_ground_z_at_xy_locked(x, y, z)
            self._last_odom_trail_pose = (x, y, trail_z, yaw)

        if self.read_only_mode:
            return
        if not self.auto_seed_from_odom:
            return

        if self.force_trail_from_odom:
            # Guaranteed narrow drivable trail along the confirmed traversed path.
            self.paint_circle(
                x, y, trail_z, self.force_trail_radius_m, allow=True, record_history=False, enforce_ground_filter=False
            )
        if self.paint_robot_footprint_from_odom:
            if prev_trail_pose is None:
                self.paint_footprint(
                    x,
                    y,
                    trail_z,
                    yaw,
                    allow=True,
                    record_history=False,
                    enforce_ground_filter=self.footprint_trail_enforce_ground_filter,
                )
            else:
                self.paint_swept_footprint(
                    prev_trail_pose,
                    (x, y, trail_z, yaw),
                    allow=True,
                    record_history=False,
                    enforce_ground_filter=self.footprint_trail_enforce_ground_filter,
                )
        if self.same_level_fill_from_odom:
            self.paint_same_level_local_fill(x, y, trail_z, yaw)

        if last_seed_xy is None:
            self.paint_circle(
                x, y, trail_z, self.seed_radius_m, allow=True, record_history=False, enforce_ground_filter=True
            )
            with self._lock:
                self._last_seed_xy = (x, y)
            return

        dist = math.hypot(x - last_seed_xy[0], y - last_seed_xy[1])
        if dist >= self.seed_step_m:
            self.paint_circle(
                x, y, trail_z, self.seed_radius_m, allow=True, record_history=False, enforce_ground_filter=True
            )
            with self._lock:
                self._last_seed_xy = (x, y)

    def apply_edit(self, x, y, allow, source):
        if not self._mutations_allowed():
            rospy.logwarn_throttle(1.0, "drivable_area %s ignored: builder is in read-only mode", source)
            return
        with self._lock:
            z = self._last_odom_z
        changed = self.paint_circle(
            x,
            y,
            z,
            self.edit_brush_radius_m,
            allow=allow,
            record_history=True,
            enforce_ground_filter=(self.manual_edit_enforce_ground_filter and allow),
        )
        if changed:
            act = "ADD" if allow else "ERASE"
            rospy.loginfo("drivable_area %s from %s at (%.2f, %.2f)", act, source, x, y)

    def add_point_callback(self, msg):
        self.apply_edit(float(msg.point.x), float(msg.point.y), True, "add_point")

    def erase_point_callback(self, msg):
        self.apply_edit(float(msg.point.x), float(msg.point.y), False, "erase_point")

    def clicked_point_callback(self, msg):
        if self.clicked_point_mode == "toggle":
            key = self.xy_to_key(float(msg.point.x), float(msg.point.y))
            allow = key not in self._cells
        else:
            allow = (self.clicked_point_mode == "add")
        self.apply_edit(float(msg.point.x), float(msg.point.y), allow, "clicked_point")

    def mode_callback(self, msg):
        mode = str(msg.data).strip().lower()
        if mode not in ("add", "erase", "toggle"):
            rospy.logwarn("drivable_area mode ignored: '%s' (use add|erase|toggle)", mode)
            return
        if mode == self.clicked_point_mode:
            return
        self.clicked_point_mode = mode
        rospy.loginfo("drivable_area clicked_point_mode set to '%s'", self.clicked_point_mode)

    def clear_callback(self, _msg):
        if not self._mutations_allowed():
            rospy.logwarn_throttle(1.0, "drivable_area clear ignored: builder is in read-only mode")
            return
        with self._lock:
            if not self._cells:
                return
            snapshot = (
                "snapshot",
                set(self._cells),
                dict(self._cell_z),
                set(self._risk_cells),
                self._last_seed_xy,
                self._last_odom_z,
            )
            self._push_history(snapshot)
            self._cells.clear()
            self._cell_z.clear()
            self._risk_cells.clear()
            self._dirty = True
            self._changed_since_persist = True
        rospy.loginfo("drivable_area cleared")

    def undo_callback(self, _msg):
        if not self._mutations_allowed():
            rospy.logwarn_throttle(1.0, "drivable_area undo ignored: builder is in read-only mode")
            return
        with self._lock:
            if not self._history:
                rospy.loginfo("drivable_area undo: no history")
                return
            item = self._history.pop()
            mode = item[0]
            if mode == "snapshot":
                self._cells = set(item[1])
                self._cell_z = dict(item[2])
                self._risk_cells = set(item[3])
                self._last_seed_xy = item[4]
                self._last_odom_z = item[5]
                self._dirty = True
                self._changed_since_persist = True
                rospy.loginfo("drivable_area undo: restored snapshot")
                return
            if mode == "delta":
                changes = item[1]
                for key, before_exists, before_z, _after_exists, _after_z in reversed(changes):
                    if before_exists:
                        self._cells.add(key)
                        self._cell_z[key] = before_z
                    else:
                        self._cells.discard(key)
                        self._cell_z.pop(key, None)
                self._dirty = True
                self._changed_since_persist = True
                rospy.loginfo("drivable_area undo: reverted last edit")

    def _build_state_snapshot(self):
        with self._lock:
            cells = [[ix, iy, float(self._cell_z.get((ix, iy), self._last_odom_z))] for ix, iy in sorted(self._cells)]
            risk_cells = [[ix, iy] for ix, iy in sorted(self._risk_cells)]
            payload = {
                "version": 1,
                "grid_resolution_m": float(self.grid_resolution_m),
                "cells": cells,
                "risk_cells": risk_cells,
                "last_seed_xy": list(self._last_seed_xy) if self._last_seed_xy is not None else None,
                "last_odom_z": float(self._last_odom_z),
                "saved_at": float(rospy.Time.now().to_sec()),
            }
            return payload

    def save_state_to_file(self, log_prefix="manual"):
        payload = self._build_state_snapshot()
        out = self.state_file_path
        out_dir = os.path.dirname(out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        tmp = out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, out)

        with self._lock:
            self._changed_since_persist = False
        rospy.loginfo("drivable_area state saved (%s): %s (cells=%d, risk=%d)", log_prefix, out, len(payload["cells"]), len(payload["risk_cells"]))

    def load_state_from_file(self, log_prefix="manual"):
        path = self.state_file_path
        if not os.path.exists(path):
            rospy.logwarn("drivable_area state file not found (%s): %s", log_prefix, path)
            return False

        with open(path, "r") as f:
            payload = json.load(f)

        saved_res = float(payload.get("grid_resolution_m", self.grid_resolution_m))
        if (not self.allow_resolution_mismatch) and abs(saved_res - self.grid_resolution_m) > 1e-6:
            rospy.logwarn(
                "drivable_area load skipped (%s): resolution mismatch saved=%.4f current=%.4f",
                log_prefix,
                saved_res,
                self.grid_resolution_m,
            )
            return False

        cells = payload.get("cells", [])
        risk_cells = payload.get("risk_cells", [])
        last_seed = payload.get("last_seed_xy", None)
        last_odom_z = float(payload.get("last_odom_z", self._last_odom_z))

        with self._lock:
            self._cells.clear()
            self._cell_z.clear()
            self._risk_cells.clear()
            for row in cells:
                if len(row) < 3:
                    continue
                ix = int(row[0])
                iy = int(row[1])
                z = float(row[2])
                key = (ix, iy)
                self._cells.add(key)
                self._cell_z[key] = z
            for row in risk_cells:
                if len(row) < 2:
                    continue
                self._risk_cells.add((int(row[0]), int(row[1])))
            if last_seed is not None and len(last_seed) >= 2:
                self._last_seed_xy = (float(last_seed[0]), float(last_seed[1]))
            else:
                self._last_seed_xy = None
            self._last_odom_z = last_odom_z
            self._dirty = True
            self._changed_since_persist = False

        rospy.loginfo(
            "drivable_area state loaded (%s): %s (cells=%d, risk=%d)",
            log_prefix,
            path,
            len(cells),
            len(risk_cells),
        )
        return True

    def save_callback(self, _msg):
        if not self._mutations_allowed():
            rospy.logwarn_throttle(1.0, "drivable_area save ignored: builder is in read-only mode")
            return
        try:
            self.save_state_to_file(log_prefix="topic")
        except Exception as e:
            rospy.logwarn("drivable_area save failed: %s", str(e))

    def load_callback(self, _msg):
        if not self._mutations_allowed():
            rospy.logwarn_throttle(1.0, "drivable_area load ignored: builder is in read-only mode")
            return
        try:
            self.load_state_from_file(log_prefix="topic")
        except Exception as e:
            rospy.logwarn("drivable_area load failed: %s", str(e))

    def on_auto_save_timer(self, _event):
        if not self._mutations_allowed():
            return
        with self._lock:
            changed = self._changed_since_persist
        if not changed:
            return
        try:
            self.save_state_to_file(log_prefix="auto")
        except Exception as e:
            rospy.logwarn_throttle(1.0, "drivable_area auto-save failed: %s", str(e))

    def on_shutdown(self):
        if not self._mutations_allowed():
            return
        if not self.save_on_shutdown:
            return
        with self._lock:
            changed = self._changed_since_persist
        if not changed and os.path.exists(self.state_file_path):
            return
        try:
            self.save_state_to_file(log_prefix="shutdown")
        except Exception as e:
            rospy.logwarn("drivable_area shutdown save failed: %s", str(e))

    def publish_marker(self):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "drivable_area"
        marker.id = 1

        with self._lock:
            cells = sorted(self._cells)
            cell_z = dict(self._cell_z)
            last_odom_z = self._last_odom_z

        if not cells:
            marker.action = Marker.DELETE
            self.pub_marker.publish(marker)
            return

        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.scale.x = self.grid_resolution_m
        marker.scale.y = self.grid_resolution_m
        marker.scale.z = self.marker_height
        marker.color.a = self.marker_color_a
        marker.color.r = self.marker_color_r
        marker.color.g = self.marker_color_g
        marker.color.b = self.marker_color_b

        step = max(1, int(math.ceil(float(len(cells)) / float(self.marker_max_cells))))
        for ix, iy in cells[::step]:
            p = Point()
            p.x, p.y = self.key_to_center(ix, iy)
            p.z = cell_z.get((ix, iy), last_odom_z) + self.marker_z_offset
            marker.points.append(p)

        self.pub_marker.publish(marker)

    def publish_risk_marker(self):
        if not self.enable_risk_marker or self.pub_risk_marker is None:
            return

        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "drivable_area_risk"
        marker.id = 2

        with self._lock:
            risk_cells = sorted(self._risk_cells)
            ground_min_z = dict(self._ground_min_z)
            last_odom_z = self._last_odom_z

        if not risk_cells:
            marker.action = Marker.DELETE
            self.pub_risk_marker.publish(marker)
            return

        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.scale.x = self.grid_resolution_m
        marker.scale.y = self.grid_resolution_m
        marker.scale.z = self.marker_height * 0.8
        marker.color.a = 0.60
        marker.color.r = 0.95
        marker.color.g = 0.15
        marker.color.b = 0.12

        step = max(1, int(math.ceil(float(len(risk_cells)) / float(self.risk_max_cells))))
        for ix, iy in risk_cells[::step]:
            p = Point()
            p.x, p.y = self.key_to_center(ix, iy)
            p.z = ground_min_z.get((ix, iy), last_odom_z) + self.marker_z_offset * 1.2
            marker.points.append(p)

        self.pub_risk_marker.publish(marker)

    def publish_grid(self):
        grid = OccupancyGrid()
        grid.header.frame_id = self.frame_id
        grid.header.stamp = rospy.Time.now()

        with self._lock:
            cells = sorted(self._cells)

        if not cells:
            grid.info.resolution = self.grid_resolution_m
            grid.info.width = 1
            grid.info.height = 1
            grid.info.origin.position.x = 0.0
            grid.info.origin.position.y = 0.0
            grid.info.origin.orientation.w = 1.0
            grid.data = [-1]
            self.pub_grid.publish(grid)
            return

        min_ix = min(ix for ix, _ in cells)
        max_ix = max(ix for ix, _ in cells)
        min_iy = min(iy for _, iy in cells)
        max_iy = max(iy for _, iy in cells)

        width = max_ix - min_ix + 1
        height = max_iy - min_iy + 1

        grid.info.resolution = self.grid_resolution_m
        grid.info.width = width
        grid.info.height = height
        grid.info.origin.position.x = min_ix * self.grid_resolution_m
        grid.info.origin.position.y = min_iy * self.grid_resolution_m
        grid.info.origin.orientation.w = 1.0

        data = [-1] * (width * height)
        for ix, iy in cells:
            gx = ix - min_ix
            gy = iy - min_iy
            idx = gy * width + gx
            data[idx] = 0  # drivable area

        grid.data = data
        self.pub_grid.publish(grid)

    def on_timer(self, _event):
        with self._lock:
            dirty = self._dirty
            if dirty:
                self._dirty = False
        if not dirty:
            return
        self.publish_marker()
        self.publish_risk_marker()
        self.publish_grid()


def main():
    rospy.init_node("lio_sam_drivable_area_builder", anonymous=False)
    DrivableAreaBuilder()
    rospy.spin()


if __name__ == "__main__":
    main()
