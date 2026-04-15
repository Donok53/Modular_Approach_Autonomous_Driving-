#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified A* path planner (OSM) with AUTO indoor/outdoor mode via ref.csv

Auto mode selection by CSV (2nd row):
- If abs(ref_lat)+abs(ref_lon) < 1e-8  --> INDOOR mode (LOCAL/ENU)
- Else                                 --> OUTDOOR mode (UTM-REL, origin = ref_east/ref_north)

Shared features:
  • Edge-orthogonal projection start snapping with forward hysteresis
  • Publish /astar/path only when changed (optional slow repub)
  • Optional Jump-Guard to blacklist abnormal long edges and replan
  • Auto-fallback: If UTM edges look insane (e.g., false-northing mix), rebuild in ENU

ROS Params (selected):
  ~osm_file (str)                 : OSM XML path
  ~ref_file (str)                 : CSV path (header includes ref_lat, ref_lon, ref_east, ref_north, ...)
  ~origin_yaw_deg (float)         : yaw-only rotation from local XY to map (deg). default 0.0
  ~jump_guard_enable (bool)       : default false
  ~jump_guard_max_step_m (float)  : default 20.0
  ~jump_guard_max_attempts (int)  : default 3
  ~debug_log_enable (bool)        : default true
  ~path_repub_period (float)      : default 0.0 (no periodic republish)
  ~snap_progress_min_step_m (float): default 3.0
  ~snap_back_allow_m (float)      : default 0.3
  ~mode_override (str)            : "AUTO"|"ENU"|"UTM". default "AUTO"

Topics:
  pub  /astar/graph_markers        (visualization_msgs/Marker)
  pub  /astar/path                 (nav_msgs/Path)
  pub  /astar/path_wgs84           (nav_msgs/Path; x=lat, y=lon just for debug)
  pub  /astar/path_node_id_list    (std_msgs/Int32MultiArray)
  pub  /astar/server_dst_node_list (sensor_msgs/PointCloud2)

  sub  /initialpose                                (geometry_msgs/PoseWithCovarianceStamped)
  sub  ~pose_topic (default: lio_localizer/odometry/optimization) (nav_msgs/Odometry)
  sub  /move_base_simple/goal                      (geometry_msgs/PoseStamped)
  sub  /server_to_robot_topic                      (astar_map/server_to_robot)
"""

import rospy, math, sys, time, csv, colorsys, struct, heapq, xml.etree.ElementTree as ET
from geometry_msgs.msg import Point, PoseStamped, Quaternion, PoseWithCovarianceStamped
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header, Int32MultiArray, Empty, Bool
from sensor_msgs.msg import PointCloud2, PointField
from astar_map.msg import server_to_robot
import utm

# -------------------- Data --------------------
class Node:
    def __init__(self, id_, east, north, lat, lon, name=None):
        self.id = id_
        self.east = east    # local XY before yaw (ENU-relative or UTM-relative)
        self.north = north
        self.lat = lat
        self.lon = lon
        self.name = name
        self.parent = None
        self.cost = sys.maxsize
    def __eq__(self, o): return self.id == o.id

class Edge:
    def __init__(self, src, dst):
        self.src = src; self.dst = dst

# -------------------- Planner --------------------
class AStarPlanner:
    def __init__(self):
        # Graph
        self.node_list = []
        self.edge_list = []

        # Mode
        self.mode = "AUTO"  # AUTO | ENU | UTM
        self._mode_param = rospy.get_param("~mode_override", "AUTO").upper()
        self._mode_locked = False

        # Yaw (local->map)
        self._yaw_deg = float(rospy.get_param("~origin_yaw_deg", 0.0))
        self._cos = math.cos(math.radians(self._yaw_deg))
        self._sin = math.sin(math.radians(self._yaw_deg))

        # ENU reference (when mode ENU)
        self._ref_lat = None
        self._ref_lon = None
        self._ecef_ref = None
        self._enu_R = None  # 3x3

        # UTM reference from CSV (when mode UTM)
        self._utm_ref_e = None
        self._utm_ref_n = None

        # Start/Goal
        self.start_id = None
        self.goal_id = None
        self.start_init_flag = False
        self.new_goal_flag = False
        self.server_dst_node_list = []
        self.pose_topic = rospy.get_param("~pose_topic", "lio_localizer/odometry/optimization")
        self.use_drivable_grid_global = bool(rospy.get_param("~use_drivable_grid_global", True))
        self.drivable_grid_topic = rospy.get_param("~drivable_grid_topic", "/lio_sam/drivable_area/grid")
        self.use_dynamic_risk_grid_global = bool(
            rospy.get_param("~use_dynamic_risk_grid_global", False)
        )
        self.dynamic_risk_grid_topic = rospy.get_param(
            "~dynamic_risk_grid_topic", "/planning/dynamic_risk_grid"
        )
        self.dynamic_risk_occupied_threshold = max(
            1, min(100, int(rospy.get_param("~dynamic_risk_occupied_threshold", 45)))
        )
        self.use_global_obstacle_overlay = bool(
            rospy.get_param("~use_global_obstacle_overlay", False)
        )
        self.global_obstacle_overlay_topic = rospy.get_param(
            "~global_obstacle_overlay_topic", "/planning/global_obstacle_overlay"
        )
        self.global_obstacle_overlay_threshold = max(
            1, min(100, int(rospy.get_param("~global_obstacle_overlay_threshold", 50)))
        )
        self.grid_unknown_is_occupied = bool(rospy.get_param("~grid_unknown_is_occupied", True))
        self.grid_snap_search_radius_cells = max(
            1, int(rospy.get_param("~grid_snap_search_radius_cells", 30))
        )
        self.robot_width_m = max(0.0, float(rospy.get_param("~robot_width_m", 0.58)))
        self.robot_length_m = max(0.0, float(rospy.get_param("~robot_length_m", 0.612)))
        self.footprint_padding_m = max(0.0, float(rospy.get_param("~footprint_padding_m", 0.0)))
        legacy_any_angle = bool(rospy.get_param("~global_path_use_any_angle", True))
        planner_param = str(rospy.get_param("~global_path_grid_planner", "")).strip().lower()
        self.global_path_grid_planner = self._normalize_global_grid_planner(
            planner_param, legacy_any_angle
        )
        self.global_path_use_any_angle = self.global_path_grid_planner == "theta"
        self.global_path_clearance_m = max(
            0.0, float(rospy.get_param("~global_path_clearance_m", 0.02))
        )
        self.global_path_clearance_model = str(
            rospy.get_param("~global_path_clearance_model", "width")
        ).strip().lower()
        if self.global_path_clearance_model not in ("width", "circumscribed"):
            rospy.logwarn(
                "[astar] unsupported global_path_clearance_model=%s, falling back to width",
                self.global_path_clearance_model,
            )
            self.global_path_clearance_model = "width"
        self.global_path_drivable_area_is_center_safe = bool(
            rospy.get_param("~global_path_drivable_area_is_center_safe", False)
        )
        self.global_path_goal_tail_clearance_radius_m = float(
            rospy.get_param("~global_path_goal_tail_clearance_radius_m", -1.0)
        )
        self.drivable_grid_graph_fallback_max_goal_gap_m = max(
            0.0,
            float(rospy.get_param("~drivable_grid_graph_fallback_max_goal_gap_m", 1.0)),
        )
        self.enable_graph_fallback = bool(
            rospy.get_param("~enable_graph_fallback", False)
        )
        self.drivable_grid_goal_extension_max_gap_m = max(
            0.0,
            float(rospy.get_param("~drivable_grid_goal_extension_max_gap_m", 0.60)),
        )
        self.preserve_user_goal_on_drivable_grid = bool(
            rospy.get_param("~preserve_user_goal_on_drivable_grid", False)
        )
        self.continuous_replan = bool(rospy.get_param("~continuous_replan", True))
        self.replan_min_start_shift_m = max(
            0.0, float(rospy.get_param("~replan_min_start_shift_m", 0.15))
        )
        self.replan_min_interval_s = max(
            0.0, float(rospy.get_param("~replan_min_interval_s", 0.40))
        )
        self.replan_path_validation_lookahead_m = max(
            0.5, float(rospy.get_param("~replan_path_validation_lookahead_m", 4.0))
        )
        self.replan_path_start_ignore_m = max(
            0.0, float(rospy.get_param("~replan_path_start_ignore_m", 0.35))
        )
        self.replan_path_start_deviation_m = max(
            self.replan_min_start_shift_m,
            float(
                rospy.get_param(
                    "~replan_path_start_deviation_m",
                    0.80,
                )
            ),
        )
        self.keep_last_path_on_replan_failure = bool(
            rospy.get_param("~keep_last_path_on_replan_failure", True)
        )
        self.keep_last_path_goal_tolerance_m = max(
            0.0, float(rospy.get_param("~keep_last_path_goal_tolerance_m", 0.05))
        )
        self.goal_reached_replan_freeze_distance_m = max(
            0.0, float(rospy.get_param("~goal_reached_replan_freeze_distance_m", 0.60))
        )
        self.keep_last_path_max_start_path_deviation_m = max(
            0.0,
            float(rospy.get_param("~keep_last_path_max_start_path_deviation_m", 0.80)),
        )
        self.planner_loop_hz = max(2.0, float(rospy.get_param("~planner_loop_hz", 8.0)))
        self.global_path_candidate_count = max(
            1, int(rospy.get_param("~global_path_candidate_count", 5))
        )
        self.global_path_candidate_penalty_radius_m = max(
            0.1,
            float(
                rospy.get_param(
                    "~global_path_candidate_penalty_radius_m",
                    max(0.60, self.robot_width_m + self.footprint_padding_m),
                )
            ),
        )
        self.global_path_candidate_penalty_cost = max(
            0.0, float(rospy.get_param("~global_path_candidate_penalty_cost", 6.0))
        )
        self.global_path_candidate_max_similarity = min(
            0.999,
            max(
                0.0, float(rospy.get_param("~global_path_candidate_max_similarity", 0.85))
            ),
        )

        # Jump-guard & debug
        self.jump_guard_enable = rospy.get_param("~jump_guard_enable", False)
        self.jump_guard_max_step_m = float(rospy.get_param("~jump_guard_max_step_m", 20.0))
        self.jump_guard_max_attempts = int(rospy.get_param("~jump_guard_max_attempts", 3))
        self.bad_edges = set()
        self.debug_log_enable = rospy.get_param("~debug_log_enable", True)
        self.path_repub_period = float(rospy.get_param("~path_repub_period", 0.0))
        self._last_path_nodes = None
        self._last_path_pub_t = 0.0
        self._last_candidate_paths_pub_t = 0.0
        self.publish_smoothed_path = bool(rospy.get_param("~publish_smoothed_path", True))
        self.path_simplify_epsilon_m = max(
            0.0, float(rospy.get_param("~path_simplify_epsilon_m", 0.20))
        )
        self.published_path_spacing_m = max(
            0.05, float(rospy.get_param("~published_path_spacing_m", 0.25))
        )

        # Snapping state
        self._snap_edge = None
        self._snap_t = None
        self._snap_last_update_s = 0.0
        self._snap_progress_min_step_m = float(rospy.get_param("~snap_progress_min_step_m", 3.0))
        self._snap_back_allow_m = float(rospy.get_param("~snap_back_allow_m", 0.3))
        self.enable_map_reload = bool(rospy.get_param("~enable_map_reload", True))
        self.reload_topic = rospy.get_param("~reload_topic", "/astar/reload_map")
        self._reload_requested = False
        self._osm_file = ""
        self._ref_file = ""
        self._goal_display_xy = None
        self._display_start_xy = None
        self._display_start_yaw = None
        self.drivable_grid = None
        self._last_world_path_signature = None
        self._last_world_path = None
        self._last_candidate_world_paths_signature = None
        self._last_snapped_goal_xy = None
        self._last_graph_goal_snap_xy = None
        self._goal_marker_xy = None
        self._last_published_start_xy = None
        self._last_published_goal_xy = None
        self._last_published_start_id = None
        self._last_published_goal_id = None
        self._last_published_start_reset_seq = 0
        self._manual_start_reset_seq = 0
        self._path_is_fallback = False
        self._last_planned_start_xy = None
        self._last_planned_goal_xy = None
        self._last_planned_start_id = None
        self._last_planned_goal_id = None
        self._last_plan_stamp_s = 0.0
        self._last_plan_success = False
        self.dynamic_risk_grid = None
        self.global_obstacle_overlay = None
        self._dynamic_risk_grid_change_seq = 0
        self._global_obstacle_overlay_change_seq = 0
        self._last_planned_dynamic_risk_grid_change_seq = 0
        self._last_planned_global_obstacle_overlay_change_seq = 0

        # Pubs/Subs
        self.pub_marker = rospy.Publisher('/astar/graph_markers', Marker, queue_size=10)
        self.pub_goal_marker = rospy.Publisher('/astar/clicked_goal_marker', Marker, queue_size=10)
        self.pub_path = rospy.Publisher('/astar/path', Path, queue_size=10, latch=True)
        self.pub_path_display = rospy.Publisher('/astar/path_display', Path, queue_size=10, latch=True)
        self.pub_path_wgs84 = rospy.Publisher('/astar/path_wgs84', Path, queue_size=10, latch=True)
        self.pub_path_node_id_list = rospy.Publisher('/astar/path_node_id_list', Int32MultiArray, queue_size=10, latch=True)
        self.pub_path_is_fallback = rospy.Publisher('/astar/path_is_fallback', Bool, queue_size=10, latch=True)
        self.pub_candidate_paths = rospy.Publisher('/astar/candidate_paths', MarkerArray, queue_size=5, latch=True)
        self.pub_server_dst_list = rospy.Publisher('/astar/server_dst_node_list', PointCloud2, queue_size=10)

        self.sub_start_from_rviz = rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.callback_start)
        self.sub_start_from_pose = rospy.Subscriber(self.pose_topic, Odometry, self.pose_callback)
        self.sub_goal_from_rviz = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.callback_goal_from_rviz)
        self.sub_goal_from_server = rospy.Subscriber('/server_to_robot_topic', server_to_robot, self.callback_goal_from_server)
        self.sub_drivable_grid = None
        if self.use_drivable_grid_global:
            self.sub_drivable_grid = rospy.Subscriber(
                self.drivable_grid_topic, OccupancyGrid, self.drivable_grid_callback, queue_size=3
            )
        self.sub_dynamic_risk_grid = None
        if self.use_drivable_grid_global and self.use_dynamic_risk_grid_global:
            self.sub_dynamic_risk_grid = rospy.Subscriber(
                self.dynamic_risk_grid_topic,
                OccupancyGrid,
                self.dynamic_risk_grid_callback,
                queue_size=3,
            )
        self.sub_global_obstacle_overlay = None
        if self.use_drivable_grid_global and self.use_global_obstacle_overlay:
            self.sub_global_obstacle_overlay = rospy.Subscriber(
                self.global_obstacle_overlay_topic,
                OccupancyGrid,
                self.global_obstacle_overlay_callback,
                queue_size=3,
            )
        self.sub_reload = None
        if self.enable_map_reload:
            self.sub_reload = rospy.Subscriber(self.reload_topic, Empty, self.callback_reload_map, queue_size=2)
        rospy.loginfo("[astar] pose topic: %s", self.pose_topic)
        if self.use_drivable_grid_global:
            rospy.loginfo("[astar] global path source: drivable grid first (%s)", self.drivable_grid_topic)
            if self.use_dynamic_risk_grid_global:
                rospy.loginfo(
                    "[astar] global dynamic risk overlay: %s (threshold=%d)",
                    self.dynamic_risk_grid_topic,
                    self.dynamic_risk_occupied_threshold,
                )
            if self.use_global_obstacle_overlay:
                rospy.loginfo(
                    "[astar] global pointcloud obstacle overlay: %s (threshold=%d)",
                    self.global_obstacle_overlay_topic,
                    self.global_obstacle_overlay_threshold,
                )
            rospy.loginfo(
                "[astar] global grid planner: %s, candidates=%d, center_safe_radius=%.3f m, goal_tail_radius=%.3f m, mask_mode=%s",
                self._global_grid_planner_label(),
                self.global_path_candidate_count,
                self._global_path_clearance_radius_m(),
                self._global_path_goal_tail_clearance_radius(),
                (
                    "as-is"
                    if self.global_path_drivable_area_is_center_safe
                    else "shrink-by-%s" % self.global_path_clearance_model
                ),
            )
            rospy.loginfo(
                "[astar] preserve exact user goal on drivable-grid path: %s",
                "on" if self.preserve_user_goal_on_drivable_grid else "off",
            )
        elif self.use_dynamic_risk_grid_global:
            rospy.logwarn(
                "[astar] use_dynamic_risk_grid_global=true ignored because use_drivable_grid_global=false"
            )
        elif self.use_global_obstacle_overlay:
            rospy.logwarn(
                "[astar] use_global_obstacle_overlay=true ignored because use_drivable_grid_global=false"
            )
        if self.enable_map_reload:
            rospy.loginfo("[astar] map reload topic: %s", self.reload_topic)

    def set_map_sources(self, osm_file, ref_file):
        self._osm_file = osm_file
        self._ref_file = ref_file

    def callback_reload_map(self, _msg):
        self._reload_requested = True

    def drivable_grid_callback(self, msg):
        self.drivable_grid = msg

    def dynamic_risk_grid_callback(self, msg):
        if self._grid_message_differs(self.dynamic_risk_grid, msg):
            self._dynamic_risk_grid_change_seq += 1
        self.dynamic_risk_grid = msg

    def global_obstacle_overlay_callback(self, msg):
        if self._grid_message_differs(self.global_obstacle_overlay, msg):
            self._global_obstacle_overlay_change_seq += 1
        self.global_obstacle_overlay = msg

    def consume_reload_request(self):
        if not self._reload_requested:
            return False
        self._reload_requested = False
        return True

    def reload_map(self):
        if not self._osm_file:
            rospy.logwarn("[astar] map reload requested but osm_file is empty")
            return False
        try:
            self.bad_edges.clear()
            self.load_osm_data(self._osm_file, self._ref_file)
            self.start_init_flag = False
            self.new_goal_flag = False
            self._last_planned_start_xy = None
            self._last_planned_goal_xy = None
            self._last_planned_start_id = None
            self._last_planned_goal_id = None
            self._last_plan_stamp_s = 0.0
            self._last_plan_success = False
            self._last_planned_dynamic_risk_grid_change_seq = self._dynamic_risk_grid_change_seq
            self._last_planned_global_obstacle_overlay_change_seq = (
                self._global_obstacle_overlay_change_seq
            )
            rospy.loginfo("[astar] map reloaded from %s", self._osm_file)
            return True
        except Exception as e:
            rospy.logwarn("[astar] map reload failed: %s", str(e))
            return False

    @staticmethod
    def _xy_distance(a, b):
        if a is None or b is None:
            return float("inf")
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

    @staticmethod
    def _quat_to_yaw(q):
        return math.atan2(
            2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y)),
            1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z)),
        )

    def _has_active_goal_context(self):
        if not self.start_init_flag:
            return False
        if self.goal_id is not None:
            return True
        return (
            self.use_drivable_grid_global
            and self._display_start_xy is not None
            and self._goal_display_xy is not None
        )

    def _is_near_active_goal(self):
        if self.goal_reached_replan_freeze_distance_m <= 0.0:
            return False
        return (
            self._display_start_xy is not None
            and self._goal_display_xy is not None
            and self._xy_distance(self._display_start_xy, self._goal_display_xy)
            <= self.goal_reached_replan_freeze_distance_m
        )

    def _mark_plan_context(self, success):
        self._last_planned_start_xy = (
            tuple(self._display_start_xy) if self._display_start_xy is not None else None
        )
        self._last_planned_goal_xy = (
            tuple(self._goal_display_xy) if self._goal_display_xy is not None else None
        )
        self._last_planned_start_id = self.start_id
        self._last_planned_goal_id = self.goal_id
        self._last_plan_stamp_s = rospy.get_time()
        self._last_plan_success = bool(success)
        self._last_planned_dynamic_risk_grid_change_seq = self._dynamic_risk_grid_change_seq
        self._last_planned_global_obstacle_overlay_change_seq = (
            self._global_obstacle_overlay_change_seq
        )

    def _should_replan_active_goal(self):
        if not self._has_active_goal_context():
            return False
        if self.new_goal_flag or self._last_plan_stamp_s <= 0.0:
            return True
        if self._is_near_active_goal():
            if self.debug_log_enable:
                rospy.loginfo_throttle(
                    1.0,
                    "[astar] near active goal; freezing replans within %.2f m",
                    self.goal_reached_replan_freeze_distance_m,
                )
            return False
        if not self.continuous_replan:
            return False

        now = rospy.get_time()
        if (now - self._last_plan_stamp_s) < self.replan_min_interval_s:
            return False
        if not self._last_plan_success:
            return True
        grid_state_changed = False
        if (
            self.use_dynamic_risk_grid_global
            and self._dynamic_risk_grid_change_seq
            != self._last_planned_dynamic_risk_grid_change_seq
        ):
            grid_state_changed = True
        if (
            self.use_global_obstacle_overlay
            and self._global_obstacle_overlay_change_seq
            != self._last_planned_global_obstacle_overlay_change_seq
        ):
            grid_state_changed = True
        if grid_state_changed:
            if not self.use_drivable_grid_global:
                return True
            if self._current_published_path_conflicts_with_current_grid():
                return True
            if self.debug_log_enable:
                rospy.loginfo_throttle(
                    1.0,
                    "[astar] overlay/risk changed but keeping current global path: upcoming path remains valid",
                )
        if (not self.use_drivable_grid_global) and self.start_id != self._last_planned_start_id:
            return True
        if self.goal_id != self._last_planned_goal_id:
            return True
        if (self._display_start_xy is None) != (self._last_planned_start_xy is None):
            return True
        if (self._goal_display_xy is None) != (self._last_planned_goal_xy is None):
            return True
        if (
            self._display_start_xy is not None
            and self._last_planned_start_xy is not None
            and self._xy_distance(self._display_start_xy, self._last_planned_start_xy)
            >= self.replan_min_start_shift_m
        ):
            if not self.use_drivable_grid_global:
                return True
            if self._current_published_path_start_deviation_m() >= self.replan_path_start_deviation_m:
                return True
            if self.debug_log_enable:
                rospy.loginfo_throttle(
                    1.0,
                    "[astar] start moved but keeping current global path: start remains close to published path",
                )
        if (
            self._goal_display_xy is not None
            and self._last_planned_goal_xy is not None
            and self._xy_distance(self._goal_display_xy, self._last_planned_goal_xy)
            >= 0.05
        ):
            return True
        return False

    # ---------- ENU helpers ----------
    _A = 6378137.0
    _E2 = 6.69437999014e-3

    @staticmethod
    def _llh_to_ecef(lat_deg, lon_deg, h=0.0):
        lat = math.radians(lat_deg); lon = math.radians(lon_deg)
        a = AStarPlanner._A; e2 = AStarPlanner._E2
        sin_lat = math.sin(lat); cos_lat = math.cos(lat)
        sin_lon = math.sin(lon); cos_lon = math.cos(lon)
        N = a / math.sqrt(1.0 - e2 * sin_lat*sin_lat)
        x = (N + h) * cos_lat * cos_lon
        y = (N + h) * cos_lat * sin_lon
        z = (N * (1.0 - e2) + h) * sin_lat
        return x, y, z

    @staticmethod
    def _ecef_to_enu_matrix(ref_lat_deg, ref_lon_deg):
        lat = math.radians(ref_lat_deg); lon = math.radians(ref_lon_deg)
        sin_lat = math.sin(lat); cos_lat = math.cos(lat)
        sin_lon = math.sin(lon); cos_lon = math.cos(lon)
        R = [
            [-sin_lon,              cos_lon,               0.0],
            [-sin_lat*cos_lon,     -sin_lat*sin_lon,      cos_lat],
            [ cos_lat*cos_lon,      cos_lat*sin_lon,      sin_lat],
        ]
        return R

    @staticmethod
    def _mat3_mul_vec3(M, v):
        return (
            M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
            M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
            M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2],
        )

    def _ll_to_enu(self, lat_deg, lon_deg):
        x, y, z = self._llh_to_ecef(lat_deg, lon_deg, 0.0)
        dx, dy, dz = x - self._ecef_ref[0], y - self._ecef_ref[1], z - self._ecef_ref[2]
        e, n, _ = self._mat3_mul_vec3(self._enu_R, (dx, dy, dz))
        return e, n

    # ---------- local -> map (yaw only) ----------
    def _xy_to_map(self, x, y):
        mx =  self._cos * x + self._sin * y
        my = -self._sin * x + self._cos * y
        return mx, my

    # ---------- CSV reader & mode decide ----------
    @staticmethod
    def _read_ref_csv(path):
        try:
            with open(path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                row = next(reader, None)
                if header is None or row is None:
                    return None
                h2i = {h.strip(): i for i, h in enumerate(header)}
                def get(name, default=0.0):
                    i = h2i.get(name, None)
                    if i is None or i >= len(row): return default
                    try: return float(row[i])
                    except: return default
                ref = {
                    "ref_lat":   get("ref_lat", 0.0),
                    "ref_lon":   get("ref_lon", 0.0),
                    "ref_east":  get("ref_east", 0.0),
                    "ref_north": get("ref_north", 0.0),
                }
                return ref
        except Exception as e:
            rospy.logwarn(f"[astar] failed to read ref CSV: {e}")
            return None

    @staticmethod
    def _read_node_tag_float(node_el, key):
        tag = node_el.find(f'tag[@k="{key}"]')
        if tag is None:
            return None
        try:
            return float(tag.attrib["v"])
        except Exception:
            return None

    def _decide_mode_from_csv(self, refcsv):
        if self._mode_param in ("ENU", "UTM"):
            self.mode = "ENU" if self._mode_param == "ENU" else "UTM"
            self._mode_locked = True
            rospy.loginfo(f"[astar] mode_override='{self._mode_param}'")
            return

        if refcsv is None:
            self.mode = "ENU"
            rospy.logwarn("[astar] no CSV → default ENU (indoor-safe).")
            return

        lat = abs(refcsv.get("ref_lat", 0.0))
        lon = abs(refcsv.get("ref_lon", 0.0))
        if lat + lon < 1e-8:
            self.mode = "ENU"
            rospy.loginfo("[astar] CSV lat/lon == 0 → INDOOR → ENU mode.")
        else:
            self.mode = "UTM"
            rospy.loginfo("[astar] CSV lat/lon present → OUTDOOR → UTM-REL mode.")

    # -------------------- Loading --------------------
    def load_osm_data(self, osm_file, ref_file):
        root = ET.parse(osm_file).getroot()
        first_nd = root.find('.//node')
        if first_nd is None:
            rospy.logerr("[astar] OSM has no nodes."); return

        # CSV & mode
        refcsv = self._read_ref_csv(ref_file) if ref_file else None
        self._decide_mode_from_csv(refcsv)

        # ENU reference (for ENU OR for fallback)
        f_id = int(first_nd.attrib['id'])
        f_lat = float(first_nd.attrib['lat']); f_lon = float(first_nd.attrib['lon'])

        if self.mode == "ENU":
            self._ref_lat = f_lat if refcsv is None or abs(refcsv.get("ref_lat", 0.0))+abs(refcsv.get("ref_lon", 0.0))<1e-8 else refcsv["ref_lat"]
            self._ref_lon = f_lon if refcsv is None or abs(refcsv.get("ref_lat", 0.0))+abs(refcsv.get("ref_lon", 0.0))<1e-8 else refcsv["ref_lon"]
            rospy.loginfo(f"[astar] ENU origin lat={self._ref_lat:.7f}, lon={self._ref_lon:.7f}")
            self._ecef_ref = self._llh_to_ecef(self._ref_lat, self._ref_lon, 0.0)
            self._enu_R = self._ecef_to_enu_matrix(self._ref_lat, self._ref_lon)

            self.node_list = []
            local_xy_count = 0
            for nd in root.findall('.//node'):
                nid = int(nd.attrib['id'])
                lat = float(nd.attrib['lat']); lon = float(nd.attrib['lon'])
                tag = nd.find('tag[@k="name"]')
                name = tag.attrib['v'] if tag is not None else None
                local_x = self._read_node_tag_float(nd, "local_x")
                local_y = self._read_node_tag_float(nd, "local_y")
                if local_x is not None and local_y is not None:
                    e, n = local_x, local_y
                    local_xy_count += 1
                else:
                    e, n = self._ll_to_enu(lat, lon)
                self.node_list.append(Node(nid, e, n, lat, lon, name))
            if local_xy_count > 0:
                rospy.loginfo("[astar] using embedded local_x/local_y for %d OSM nodes", local_xy_count)

        else:  # UTM-REL
            self._utm_ref_e = refcsv.get("ref_east", 0.0) if refcsv else 0.0
            self._utm_ref_n = refcsv.get("ref_north", 0.0) if refcsv else 0.0
            rospy.loginfo(f"[astar] UTM-REL origin east={self._utm_ref_e:.3f}, north={self._utm_ref_n:.3f}")

            # Also keep ENU reference ready for possible fallback
            self._ref_lat = f_lat; self._ref_lon = f_lon
            self._ecef_ref = self._llh_to_ecef(self._ref_lat, self._ref_lon, 0.0)
            self._enu_R = self._ecef_to_enu_matrix(self._ref_lat, self._ref_lon)

            self.node_list = []
            big_jump = False
            for nd in root.findall('.//node'):
                nid = int(nd.attrib['id'])
                lat = float(nd.attrib['lat']); lon = float(nd.attrib['lon'])
                tag = nd.find('tag[@k="name"]')
                name = tag.attrib['v'] if tag is not None else None
                ue, un, znum, zlet = utm.from_latlon(lat, lon)
                x = ue - self._utm_ref_e
                y = un - self._utm_ref_n
                self.node_list.append(Node(nid, x, y, lat, lon, name))

            # quick sanity on edges (false-northing mix → 10,000 km jumps)
            mx = 0.0; cnt = 0
            for way in root.findall('.//way'):
                ids = way.findall('nd')
                for i in range(len(ids) - 1):
                    a = self.findNodeById(int(ids[i].attrib['ref']))
                    b = self.findNodeById(int(ids[i+1].attrib['ref']))
                    if a and b:
                        d = math.hypot(a.east - b.east, a.north - b.north)
                        mx = max(mx, d); cnt += 1
            if cnt > 0 and mx > 5_000.0 and not self._mode_locked:
                rospy.logwarn(f"[astar] abnormal UTM edge length max={mx:.1f} m → fallback to ENU.")
                # rebuild nodes in ENU
                self.mode = "ENU"
                self.node_list = []
                self._ecef_ref = self._llh_to_ecef(self._ref_lat, self._ref_lon, 0.0)
                self._enu_R = self._ecef_to_enu_matrix(self._ref_lat, self._ref_lon)
                for nd in root.findall('.//node'):
                    nid = int(nd.attrib['id'])
                    lat = float(nd.attrib['lat']); lon = float(nd.attrib['lon'])
                    tag = nd.find('tag[@k="name"]')
                    name = tag.attrib['v'] if tag is not None else None
                    e, n = self._ll_to_enu(lat, lon)
                    self.node_list.append(Node(nid, e, n, lat, lon, name))

        # Build edges (bidirectional)
        self.edge_list = []
        lens = []
        for way in root.findall('.//way'):
            ids = way.findall('nd')
            for i in range(len(ids) - 1):
                a = self.findNodeById(int(ids[i].attrib['ref']))
                b = self.findNodeById(int(ids[i+1].attrib['ref']))
                if a is None or b is None: continue
                d = math.hypot(a.east - b.east, a.north - b.north)
                lens.append(d)
                self.edge_list.append(Edge(a, b))
                self.edge_list.append(Edge(b, a))

        if lens:
            lens.sort()
            med = lens[len(lens)//2]; p90 = lens[int(0.9*len(lens))]; mx = lens[-1]
            rospy.loginfo(f"[astar] edges built ({self.mode}) len median={med:.2f} p90={p90:.2f} max={mx:.2f} m")

    # -------------------- Graph helpers --------------------
    def set_dst_node_list(self, lst): self.server_dst_node_list = lst

    def graph_setup(self):
        for n in self.node_list:
            n.parent = None; n.cost = sys.maxsize

    def findNodeById(self, id_):
        for n in self.node_list:
            if n.id == id_: return n
        return None

    # -------------------- A* --------------------
    def edges(self, node):
        return [e for e in self.edge_list if e.src == node and (e.src.id, e.dst.id) not in self.bad_edges]

    def distance(self, a, b):
        return math.hypot(a.east - b.east, a.north - b.north)

    def planning(self, start_id, goal_id):
        s = self.findNodeById(start_id); g = self.findNodeById(goal_id)
        if s is None or g is None: return []
        open_set = {s.id: s}; closed = {}
        s.cost = 0.0
        reached = (s == g)
        while open_set:
            c_id = min(open_set, key=lambda i: open_set[i].cost + self.distance(g, open_set[i]))
            cur = open_set[c_id]
            if cur == g:
                reached = True
                break
            del open_set[c_id]; closed[c_id] = cur
            for e in self.edges(cur):
                nid = e.dst.id
                if nid in closed: continue
                new_cost = e.src.cost + self.distance(e.src, e.dst)
                if e.dst.cost > new_cost:
                    e.dst.cost = new_cost; e.dst.parent = e.src
                if nid not in open_set: open_set[nid] = e.dst
        if not reached:
            return []
        return self._backtrack(g)

    def _backtrack(self, node):
        path = []; cur = node
        while cur and cur.parent is not None:
            path.append(cur.id); cur = cur.parent
        if cur: path.append(cur.id)
        path.reverse(); return path

    # -------------------- Jump Guard --------------------
    def _seg_len_map(self, u, v):
        ux, uy = self._xy_to_map(u.east, u.north)
        vx, vy = self._xy_to_map(v.east, v.north)
        return math.hypot(vx - ux, vy - uy)

    @staticmethod
    def _point_line_distance(pt, a, b):
        ax, ay = a
        bx, by = b
        px, py = pt
        vx = bx - ax
        vy = by - ay
        denom = vx * vx + vy * vy
        if denom <= 1e-12:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * vx + (py - ay) * vy) / denom
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        proj_x = ax + t * vx
        proj_y = ay + t * vy
        return math.hypot(px - proj_x, py - proj_y)

    def _distance_point_to_polyline(self, pt, points):
        if not points:
            return float("inf")
        if len(points) == 1:
            return math.hypot(float(pt[0]) - float(points[0][0]), float(pt[1]) - float(points[0][1]))
        best = float("inf")
        for idx in range(len(points) - 1):
            best = min(best, self._point_line_distance(pt, points[idx], points[idx + 1]))
        return best

    def _rdp(self, points, epsilon):
        if len(points) <= 2 or epsilon <= 0.0:
            return list(points)
        start = points[0]
        end = points[-1]
        max_dist = -1.0
        split_idx = -1
        for i in range(1, len(points) - 1):
            d = self._point_line_distance(points[i], start, end)
            if d > max_dist:
                max_dist = d
                split_idx = i
        if max_dist <= epsilon or split_idx < 0:
            return [start, end]
        left = self._rdp(points[: split_idx + 1], epsilon)
        right = self._rdp(points[split_idx:], epsilon)
        return left[:-1] + right

    def _resample_path(self, points, spacing):
        if len(points) <= 1:
            return list(points)
        out = [points[0]]
        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            seg_len = math.hypot(x1 - x0, y1 - y0)
            if seg_len <= 1e-9:
                continue
            steps = max(1, int(math.ceil(seg_len / spacing)))
            for step in range(1, steps + 1):
                t = float(step) / float(steps)
                out.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
        return out

    @staticmethod
    def _project_point_to_segment(point, seg_a, seg_b):
        px, py = float(point[0]), float(point[1])
        ax, ay = float(seg_a[0]), float(seg_a[1])
        bx, by = float(seg_b[0]), float(seg_b[1])
        vx = bx - ax
        vy = by - ay
        denom = vx * vx + vy * vy
        if denom <= 1e-9:
            return (ax, ay), 0.0, (px - ax) * (px - ax) + (py - ay) * (py - ay)
        t = ((px - ax) * vx + (py - ay) * vy) / denom
        t = max(0.0, min(1.0, t))
        proj = (ax + t * vx, ay + t * vy)
        d2 = (px - proj[0]) * (px - proj[0]) + (py - proj[1]) * (py - proj[1])
        return proj, t, d2

    def _trim_world_path_from_current_start(self, world_points):
        pts = self._dedupe_world_points(world_points)
        if len(pts) <= 1 or self._display_start_xy is None:
            return pts

        best_idx = 0
        best_t = 0.0
        best_proj = pts[0]
        best_d2 = self._xy_distance(self._display_start_xy, pts[0]) ** 2
        for idx in range(len(pts) - 1):
            proj, t, d2 = self._project_point_to_segment(
                self._display_start_xy, pts[idx], pts[idx + 1]
            )
            if d2 < best_d2:
                best_idx = idx
                best_t = t
                best_proj = proj
                best_d2 = d2

        trimmed = []
        if best_t <= 1e-3:
            trimmed.append(pts[best_idx])
            trimmed.extend(pts[best_idx + 1 :])
        elif best_t >= 1.0 - 1e-3:
            trimmed.append(pts[best_idx + 1])
            trimmed.extend(pts[best_idx + 2 :])
        else:
            trimmed.append(best_proj)
            trimmed.extend(pts[best_idx + 1 :])
        return self._dedupe_world_points(trimmed)

    def _prepare_display_path(self, world_points, simplify=True):
        if len(world_points) <= 1:
            return list(world_points)
        pts = self._trim_world_path_from_current_start(world_points)
        if simplify and self.publish_smoothed_path and self.path_simplify_epsilon_m > 0.0:
            pts = self._rdp(pts, self.path_simplify_epsilon_m)
        return self._resample_path(pts, self.published_path_spacing_m)

    def _prepare_visualization_path(self, world_points, simplify=True):
        pts = self._prepare_display_path(world_points, simplify=simplify)
        if self._display_start_xy is not None:
            if (not pts) or self._xy_distance(self._display_start_xy, pts[0]) > 0.05:
                pts = [tuple(self._display_start_xy)] + pts
        if self._goal_display_xy is not None:
            if (not pts) or self._xy_distance(self._goal_display_xy, pts[-1]) > 0.05:
                pts = list(pts) + [tuple(self._goal_display_xy)]
        return self._dedupe_world_points(pts)

    def _apply_start_yaw_hint(self, yaws):
        if (not yaws) or self._display_start_yaw is None:
            return yaws
        hinted = list(yaws)
        hinted[0] = float(self._display_start_yaw)
        return hinted

    def _prepend_current_start_to_path_points(self, points):
        pts = list(points)
        if self._display_start_xy is not None:
            if (not pts) or self._xy_distance(self._display_start_xy, pts[0]) > 0.05:
                pts = [tuple(self._display_start_xy)] + pts
        return self._dedupe_world_points(pts)

    @staticmethod
    def _set_pose_yaw(pose_stamped, yaw):
        pose_stamped.pose.orientation.x = 0.0
        pose_stamped.pose.orientation.y = 0.0
        pose_stamped.pose.orientation.z = math.sin(0.5 * yaw)
        pose_stamped.pose.orientation.w = math.cos(0.5 * yaw)

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
    def _world_path_signature(world_points):
        return tuple((round(float(x), 2), round(float(y), 2)) for x, y in world_points)

    def _candidate_world_paths_signature(self, world_paths):
        return tuple(self._world_path_signature(path) for path in world_paths if path)

    @staticmethod
    def _dedupe_world_points(world_points):
        out = []
        for x, y in world_points:
            pt = (float(x), float(y))
            if out and math.hypot(pt[0] - out[-1][0], pt[1] - out[-1][1]) <= 1e-6:
                continue
            out.append(pt)
        return out

    def _global_path_clearance_radius_m(self):
        # /drivable_area/grid is a drivable-region mask, so we derive a center-safe mask
        # at planning time instead of modifying the published map itself.
        if self.global_path_drivable_area_is_center_safe:
            base_radius = 0.0
        elif self.global_path_clearance_model == "circumscribed":
            # Legacy conservative mode: keep the robot corners clear on diagonal segments.
            base_radius = 0.5 * math.hypot(self.robot_length_m, self.robot_width_m)
        else:
            # Prefer a width-based center corridor for narrow passages when localization is stable.
            base_radius = 0.5 * self.robot_width_m
        return base_radius + self.footprint_padding_m + self.global_path_clearance_m

    def _global_path_goal_tail_clearance_radius(self):
        body_radius = self._global_path_clearance_radius_m()
        if self.global_path_goal_tail_clearance_radius_m < 0.0:
            return body_radius
        return min(body_radius, max(0.0, self.global_path_goal_tail_clearance_radius_m))

    def _publish_world_path_messages(self, world_points, stamp=None, simplify=True):
        if not world_points:
            return
        if stamp is None:
            stamp = rospy.Time.now()

        p = Path()
        p.header.frame_id = "map"
        p.header.stamp = stamp
        display_points = self._prepend_current_start_to_path_points(
            self._prepare_display_path(world_points, simplify=simplify)
        )
        display_yaws = self._apply_start_yaw_hint(self._path_yaws(display_points))
        for (x, y), yaw in zip(display_points, display_yaws):
            ps = PoseStamped()
            ps.header = p.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            self._set_pose_yaw(ps, yaw)
            p.poses.append(ps)

        pd = Path()
        pd.header.frame_id = "map"
        pd.header.stamp = stamp
        viz_points = self._prepare_visualization_path(world_points, simplify=simplify)
        viz_yaws = self._apply_start_yaw_hint(self._path_yaws(viz_points))
        for (x, y), yaw in zip(viz_points, viz_yaws):
            pds = PoseStamped()
            pds.header = pd.header
            pds.pose.position.x = float(x)
            pds.pose.position.y = float(y)
            pds.pose.position.z = 0.0
            self._set_pose_yaw(pds, yaw)
            pd.poses.append(pds)

        pw = Path()
        pw.header.frame_id = "map"
        pw.header.stamp = stamp

        self.pub_path.publish(p)
        self.pub_path_display.publish(pd)
        self.pub_path_wgs84.publish(pw)

        ids_msg = Int32MultiArray()
        ids_msg.data = []
        self.pub_path_node_id_list.publish(ids_msg)

    def _publish_candidate_world_paths_messages(self, world_paths, stamp=None):
        if stamp is None:
            stamp = rospy.Time.now()
        msg = MarkerArray()

        clear = Marker()
        clear.header.frame_id = "map"
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        msg.markers.append(clear)

        for idx, world_points in enumerate(world_paths):
            if not world_points:
                continue
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = stamp
            marker.ns = "astar_candidate_paths"
            marker.id = idx
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.06 if idx == 0 else 0.04
            if idx == 0:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 0.95
            else:
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 0.85
            for x, y in self._prepare_visualization_path(world_points, simplify=False):
                marker.points.append(Point(float(x), float(y), 0.03))
            msg.markers.append(marker)

        self.pub_candidate_paths.publish(msg)

    def publish_candidate_world_paths_if_changed(self, world_paths, stamp=None, force=False):
        signature = self._candidate_world_paths_signature(world_paths)
        if (not force) and signature == self._last_candidate_world_paths_signature:
            return
        self._last_candidate_world_paths_signature = signature
        self._last_candidate_paths_pub_t = time.monotonic()
        self._publish_candidate_world_paths_messages(world_paths, stamp=stamp)

    def publish_candidate_world_paths_if_needed(self, world_paths, stamp=None):
        signature = self._candidate_world_paths_signature(world_paths)
        changed = signature != self._last_candidate_world_paths_signature
        do_periodic = False
        if (not changed) and self.path_repub_period > 0.0:
            tnow = time.monotonic()
            if (tnow - self._last_candidate_paths_pub_t) >= self.path_repub_period:
                do_periodic = True
                self._last_candidate_paths_pub_t = tnow
        if not changed and not do_periodic:
            return
        self._last_candidate_world_paths_signature = signature
        if changed:
            self._last_candidate_paths_pub_t = time.monotonic()
        self._publish_candidate_world_paths_messages(world_paths, stamp=stamp)

    def clear_candidate_world_paths(self, stamp=None, force=False):
        if (not force) and self._last_candidate_world_paths_signature is None:
            return
        self._last_candidate_world_paths_signature = None
        self._last_candidate_paths_pub_t = 0.0
        self._publish_candidate_world_paths_messages([], stamp=stamp)

    def _publish_path_fallback_state(self, is_fallback, force=False):
        is_fallback = bool(is_fallback)
        if (not force) and self._path_is_fallback == is_fallback:
            return
        self._path_is_fallback = is_fallback
        self.pub_path_is_fallback.publish(Bool(data=is_fallback))

    def _capture_published_path_context(self):
        self._last_published_start_xy = (
            tuple(self._display_start_xy) if self._display_start_xy is not None else None
        )
        self._last_published_goal_xy = (
            tuple(self._goal_display_xy) if self._goal_display_xy is not None else None
        )
        self._last_published_start_id = self.start_id
        self._last_published_goal_id = self.goal_id
        self._last_published_start_reset_seq = self._manual_start_reset_seq

    def _clear_published_path_context(self):
        self._last_published_start_xy = None
        self._last_published_goal_xy = None
        self._last_published_start_id = None
        self._last_published_goal_id = None
        self._last_published_start_reset_seq = self._manual_start_reset_seq

    def _current_published_path_world_points(self):
        if self._last_world_path:
            return list(self._last_world_path)
        if not self._last_path_nodes:
            return []
        points = []
        for nid in self._last_path_nodes:
            n = self.findNodeById(nid)
            if n is None:
                continue
            points.append(self._xy_to_map(n.east, n.north))
        return points

    def _current_published_path_start_deviation_m(self):
        if self._display_start_xy is None:
            return float("inf")
        path_points = self._current_published_path_world_points()
        if not path_points:
            return float("inf")
        return self._distance_point_to_polyline(self._display_start_xy, path_points)

    def _world_path_conflicts_with_blocked_grid(self, g, blocked, world_points):
        pts = self._dedupe_world_points(world_points)
        if len(pts) < 2:
            return False

        start_ignore_m = max(0.0, self.replan_path_start_ignore_m)
        lookahead_m = max(start_ignore_m, self.replan_path_validation_lookahead_m)
        accumulated_m = 0.0

        for idx in range(len(pts) - 1):
            ax, ay = pts[idx]
            bx, by = pts[idx + 1]
            seg_len = math.hypot(float(bx) - float(ax), float(by) - float(ay))
            if seg_len <= 1e-6:
                continue

            seg_start_m = accumulated_m
            seg_end_m = accumulated_m + seg_len
            accumulated_m = seg_end_m

            if seg_end_m <= start_ignore_m:
                continue
            if seg_start_m >= lookahead_m:
                break

            t0 = 0.0
            t1 = 1.0
            if seg_start_m < start_ignore_m:
                t0 = (start_ignore_m - seg_start_m) / seg_len
            if seg_end_m > lookahead_m:
                t1 = (lookahead_m - seg_start_m) / seg_len
            t0 = max(0.0, min(1.0, t0))
            t1 = max(0.0, min(1.0, t1))
            if t1 <= t0:
                continue

            sx = float(ax) + (float(bx) - float(ax)) * t0
            sy = float(ay) + (float(by) - float(ay)) * t0
            ex = float(ax) + (float(bx) - float(ax)) * t1
            ey = float(ay) + (float(by) - float(ay)) * t1

            start_cell = self._world_to_grid_cell(g, sx, sy)
            end_cell = self._world_to_grid_cell(g, ex, ey)
            if (not self._grid_in_bounds(g, start_cell[0], start_cell[1])) or (
                not self._grid_in_bounds(g, end_cell[0], end_cell[1])
            ):
                return True
            if not self._has_line_of_sight(blocked, start_cell, end_cell):
                return True
        return False

    def _current_published_path_conflicts_with_current_grid(self):
        if (not self.use_drivable_grid_global) or self.drivable_grid is None:
            return True
        path_points = self._current_published_path_world_points()
        if len(path_points) < 2:
            return True
        blocked = self._build_blocked_grid(self.drivable_grid)
        return self._world_path_conflicts_with_blocked_grid(
            self.drivable_grid, blocked, path_points
        )

    def _can_keep_last_path_on_replan_failure(self):
        if not self.keep_last_path_on_replan_failure:
            return False, "disabled"
        if self._last_path_nodes is None and self._last_world_path_signature is None:
            return False, "no_previous_path"
        if (
            self._last_published_goal_id is not None
            and self.goal_id is not None
            and self.goal_id != self._last_published_goal_id
        ):
            return False, "goal_changed"
        if (self._goal_display_xy is None) != (self._last_published_goal_xy is None):
            return False, "goal_changed"
        if (
            self._goal_display_xy is not None
            and self._last_published_goal_xy is not None
            and self._xy_distance(self._goal_display_xy, self._last_published_goal_xy)
            > self.keep_last_path_goal_tolerance_m
        ):
            return False, "goal_changed"
        if self._manual_start_reset_seq != self._last_published_start_reset_seq:
            return False, "manual_start_reset"
        if self._display_start_xy is None:
            return False, "missing_start"
        path_points = self._current_published_path_world_points()
        if not path_points:
            return False, "missing_path_geometry"
        if (
            self._distance_point_to_polyline(self._display_start_xy, path_points)
            > self.keep_last_path_max_start_path_deviation_m
        ):
            return False, "start_off_path"
        return True, "ok"

    def publish_world_path_if_changed(self, world_points):
        now = rospy.Time.now()
        signature = self._world_path_signature(world_points)
        changed = self._last_world_path_signature != signature
        do_periodic = False
        if not changed and self.path_repub_period > 0.0:
            tnow = time.monotonic()
            if (tnow - self._last_path_pub_t) >= self.path_repub_period:
                do_periodic = True
                self._last_path_pub_t = tnow

        if changed:
            self._last_world_path_signature = signature
            self._last_world_path = list(world_points)
            self._last_path_nodes = None
            self._last_path_pub_t = time.monotonic()
            self._capture_published_path_context()
            self._publish_path_fallback_state(False, force=True)
            self._publish_world_path_messages(world_points, stamp=now, simplify=False)
            if self.debug_log_enable:
                goal_txt = ""
                if self._last_snapped_goal_xy is not None:
                    goal_txt = " snapped_goal=({:.2f}, {:.2f})".format(
                        self._last_snapped_goal_xy[0], self._last_snapped_goal_xy[1]
                    )
                rospy.loginfo(
                    "[astar] drivable-grid path published (%d pts)%s",
                    len(world_points),
                    goal_txt,
                )
        elif do_periodic:
            self._capture_published_path_context()
            self._publish_path_fallback_state(False)
            self._publish_world_path_messages(world_points, stamp=now, simplify=False)
            if self.debug_log_enable:
                rospy.loginfo("[astar] drivable-grid path republished (periodic)")
        elif self._path_is_fallback:
            self._capture_published_path_context()
            self._publish_path_fallback_state(False)
            self._last_path_pub_t = time.monotonic()
            self._publish_world_path_messages(world_points, stamp=now, simplify=False)
            if self.debug_log_enable:
                rospy.loginfo("[astar] drivable-grid path republished (fallback cleared)")

    def clear_published_path(self, stamp=None):
        if self._last_path_nodes is None and self._last_world_path_signature is None:
            self._clear_published_path_context()
            self.clear_candidate_world_paths(stamp=stamp, force=True)
            self._publish_path_fallback_state(False, force=True)
            return
        if stamp is None:
            stamp = rospy.Time.now()
        empty = Path()
        empty.header.frame_id = "map"
        empty.header.stamp = stamp
        self.pub_path.publish(empty)
        self.pub_path_display.publish(empty)
        self.pub_path_wgs84.publish(empty)
        self.pub_path_node_id_list.publish(Int32MultiArray(data=[]))
        self._last_path_nodes = None
        self._last_world_path_signature = None
        self._last_world_path = None
        self._last_path_pub_t = 0.0
        self._last_candidate_paths_pub_t = 0.0
        self._clear_published_path_context()
        self.clear_candidate_world_paths(stamp=stamp, force=True)
        self._publish_path_fallback_state(False, force=True)
        if self.debug_log_enable:
            rospy.loginfo("[astar] cleared published path")

    def validate_or_blacklist(self, path_ids):
        if not self.jump_guard_enable or len(path_ids) < 2: return path_ids
        thr = self.jump_guard_max_step_m
        for u_id, v_id in zip(path_ids, path_ids[1:]):
            u = self.findNodeById(u_id); v = self.findNodeById(v_id)
            if u is None or v is None: continue
            if self._seg_len_map(u, v) > thr:
                rospy.logwarn(f"[jump_guard] abnormal jump {u_id}->{v_id} (> {thr} m). Blacklist & replan.")
                self.bad_edges.add((u_id, v_id)); self.bad_edges.add((v_id, u_id))
                return None
        return path_ids

    # -------------------- Edge-projection snapping --------------------
    def _nearest_projection_on_edges(self, x, y):
        best_e = None; best_t = 0.0; best_px = best_py = 0.0; best_d2 = 1e18; best_len = 0.0
        for e in self.edge_list:
            sx, sy = self._xy_to_map(e.src.east, e.src.north)
            dx, dy = self._xy_to_map(e.dst.east, e.dst.north)
            vx, vy = dx - sx, dy - sy
            denom = vx*vx + vy*vy
            if denom < 1e-12:
                t = 0.0; px, py = sx, sy; elen = 0.0
            else:
                t = ((x - sx)*vx + (y - sy)*vy) / denom
                t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
                px, py = sx + t*vx, sy + t*vy
                elen = math.sqrt(denom)
            d2 = (x - px)**2 + (y - py)**2
            if d2 < best_d2:
                best_d2 = d2; best_e = e; best_t = t; best_px = px; best_py = py; best_len = elen
        return best_e, best_t, best_px, best_py, best_d2, best_len

    def _accept_or_clamp_projection(self, edge, t, edge_len):
        progressed_m = 0.0
        if self._snap_edge is None:
            self._snap_edge = (edge.src.id, edge.dst.id); self._snap_t = float(t)
            return self._snap_edge, self._snap_t, progressed_m

        prev_src, prev_dst = self._snap_edge
        cur_pair = (edge.src.id, edge.dst.id)

        if cur_pair == self._snap_edge:
            if t + (self._snap_back_allow_m / max(edge_len, 1e-6)) >= self._snap_t:
                progressed_m = (t - self._snap_t) * edge_len
                self._snap_t = max(self._snap_t, t)
            return self._snap_edge, self._snap_t, max(progressed_m, 0.0)

        if prev_dst == edge.src.id and (self._snap_t is not None and self._snap_t > 0.9):
            self._snap_edge = cur_pair; self._snap_t = float(t)
            progressed_m = t * edge_len
            return self._snap_edge, self._snap_t, progressed_m

        return self._snap_edge, self._snap_t, 0.0

    # -------------------- Drivable-grid global path --------------------
    @staticmethod
    def _grid_in_bounds(g, gx, gy):
        return 0 <= gx < int(g.info.width) and 0 <= gy < int(g.info.height)

    @staticmethod
    def _blocked_in_bounds(blocked, gx, gy):
        return 0 <= gy < len(blocked) and 0 <= gx < len(blocked[0])

    def _world_to_grid_cell(self, g, x, y):
        res = float(g.info.resolution)
        gx = int(math.floor((float(x) - float(g.info.origin.position.x)) / res))
        gy = int(math.floor((float(y) - float(g.info.origin.position.y)) / res))
        return gx, gy

    def _grid_cell_to_world(self, g, gx, gy):
        res = float(g.info.resolution)
        x = float(g.info.origin.position.x) + (gx + 0.5) * res
        y = float(g.info.origin.position.y) + (gy + 0.5) * res
        return x, y

    def _raw_grid_cell_is_free(self, g, gx, gy):
        if not self._grid_in_bounds(g, gx, gy):
            return False
        idx = gy * int(g.info.width) + gx
        val = int(g.data[idx])
        if val == 0:
            return True
        if val < 0 and (not self.grid_unknown_is_occupied):
            return True
        return False

    @staticmethod
    def _grid_layout_matches(a, b):
        if a is None or b is None:
            return False
        tol = 1e-6
        return (
            int(a.info.width) == int(b.info.width)
            and int(a.info.height) == int(b.info.height)
            and abs(float(a.info.resolution) - float(b.info.resolution)) <= tol
            and abs(float(a.info.origin.position.x) - float(b.info.origin.position.x)) <= tol
            and abs(float(a.info.origin.position.y) - float(b.info.origin.position.y)) <= tol
            and str(a.header.frame_id).strip() == str(b.header.frame_id).strip()
        )

    def _grid_message_differs(self, previous, current):
        if previous is None or current is None:
            return previous is not current
        if not self._grid_layout_matches(previous, current):
            return True
        prev_data = getattr(previous, "data", None)
        cur_data = getattr(current, "data", None)
        if prev_data is None or cur_data is None:
            return True
        if len(prev_data) != len(cur_data):
            return True
        return prev_data != cur_data

    def _build_blocked_grid(self, g, clearance_radius_m=None):
        w = int(g.info.width)
        h = int(g.info.height)
        blocked = [[False for _ in range(w)] for _ in range(h)]
        occupied = []
        risk_data = None
        global_obstacle_overlay_data = None

        if self.use_dynamic_risk_grid_global and self.dynamic_risk_grid is not None:
            if self._grid_layout_matches(g, self.dynamic_risk_grid):
                if len(self.dynamic_risk_grid.data) == (w * h):
                    risk_data = self.dynamic_risk_grid.data
                else:
                    rospy.logwarn_throttle(
                        5.0,
                        "[astar] dynamic risk grid size mismatch; ignoring global overlay",
                    )
            else:
                rospy.logwarn_throttle(
                    5.0,
                    "[astar] dynamic risk grid layout mismatch; ignoring global overlay",
                )
        if self.use_global_obstacle_overlay and self.global_obstacle_overlay is not None:
            if self._grid_layout_matches(g, self.global_obstacle_overlay):
                if len(self.global_obstacle_overlay.data) == (w * h):
                    global_obstacle_overlay_data = self.global_obstacle_overlay.data
                else:
                    rospy.logwarn_throttle(
                        5.0,
                        "[astar] global obstacle overlay size mismatch; ignoring overlay",
                    )
            else:
                rospy.logwarn_throttle(
                    5.0,
                    "[astar] global obstacle overlay layout mismatch; ignoring overlay",
                )

        for gy in range(h):
            row_offset = gy * w
            for gx in range(w):
                val = int(g.data[row_offset + gx])
                risk_occupied = False
                if risk_data is not None:
                    risk_occupied = int(risk_data[row_offset + gx]) >= self.dynamic_risk_occupied_threshold
                pointcloud_overlay_occupied = False
                if global_obstacle_overlay_data is not None:
                    pointcloud_overlay_occupied = (
                        int(global_obstacle_overlay_data[row_offset + gx])
                        >= self.global_obstacle_overlay_threshold
                    )
                is_free = (
                    ((val == 0) or (val < 0 and (not self.grid_unknown_is_occupied)))
                    and (not risk_occupied)
                    and (not pointcloud_overlay_occupied)
                )
                if not is_free:
                    occupied.append((gx, gy))

        # This is equivalent to shrinking the drivable region by the robot center margin.
        clearance_radius_m = (
            self._global_path_clearance_radius_m()
            if clearance_radius_m is None
            else max(0.0, float(clearance_radius_m))
        )
        inflate_radius_cells = int(
            math.ceil(clearance_radius_m / max(1e-6, float(g.info.resolution)))
        )
        if inflate_radius_cells <= 0:
            for gx, gy in occupied:
                blocked[gy][gx] = True
            return blocked

        offsets = []
        radius_sq = inflate_radius_cells * inflate_radius_cells
        for dy in range(-inflate_radius_cells, inflate_radius_cells + 1):
            for dx in range(-inflate_radius_cells, inflate_radius_cells + 1):
                if dx * dx + dy * dy <= radius_sq:
                    offsets.append((dx, dy))

        for ox, oy in occupied:
            for dx, dy in offsets:
                nx = ox + dx
                ny = oy + dy
                if self._blocked_in_bounds(blocked, nx, ny):
                    blocked[ny][nx] = True
        return blocked

    def _blocked_cell_is_free(self, blocked, gx, gy):
        return self._blocked_in_bounds(blocked, gx, gy) and (not blocked[gy][gx])

    def _nearest_free_grid_cell(self, blocked, cell):
        cx, cy = cell
        if self._blocked_cell_is_free(blocked, cx, cy):
            return (cx, cy)

        best = None
        best_d2 = float("inf")
        for r in range(1, self.grid_snap_search_radius_cells + 1):
            found_this_ring = False
            for gx in range(cx - r, cx + r + 1):
                for gy in range(cy - r, cy + r + 1):
                    if max(abs(gx - cx), abs(gy - cy)) != r:
                        continue
                    if not self._blocked_cell_is_free(blocked, gx, gy):
                        continue
                    d2 = float((gx - cx) * (gx - cx) + (gy - cy) * (gy - cy))
                    if d2 < best_d2:
                        best_d2 = d2
                        best = (gx, gy)
                        found_this_ring = True
            if found_this_ring:
                return best
        return None

    def _nearest_free_start_grid_cell(self, g, blocked, start_xy, goal_xy=None):
        start_raw = self._world_to_grid_cell(g, start_xy[0], start_xy[1])
        if self._blocked_cell_is_free(blocked, start_raw[0], start_raw[1]):
            return start_raw

        heading_yaw = self._display_start_yaw
        goal_dx = 0.0
        goal_dy = 0.0
        goal_norm = 0.0
        if goal_xy is not None:
            goal_dx = float(goal_xy[0]) - float(start_xy[0])
            goal_dy = float(goal_xy[1]) - float(start_xy[1])
            goal_norm = math.hypot(goal_dx, goal_dy)

        best = None
        best_score = float("inf")
        best_dist = float("inf")
        cx, cy = start_raw
        for r in range(1, self.grid_snap_search_radius_cells + 1):
            for gx in range(cx - r, cx + r + 1):
                for gy in range(cy - r, cy + r + 1):
                    if max(abs(gx - cx), abs(gy - cy)) != r:
                        continue
                    if not self._blocked_cell_is_free(blocked, gx, gy):
                        continue

                    wx, wy = self._grid_cell_to_world(g, gx, gy)
                    dx = float(wx) - float(start_xy[0])
                    dy = float(wy) - float(start_xy[1])
                    dist = math.hypot(dx, dy)
                    score = dist

                    if heading_yaw is not None:
                        hx = math.cos(float(heading_yaw))
                        hy = math.sin(float(heading_yaw))
                        forward = dx * hx + dy * hy
                        lateral = abs(-hy * dx + hx * dy)
                        score += 1.0 * lateral
                        score -= 0.75 * max(0.0, forward)
                        if forward < -0.05:
                            score += 0.75 + abs(forward)

                    if goal_norm > 1e-3:
                        progress = (dx * goal_dx + dy * goal_dy) / goal_norm
                        if progress < -0.05:
                            score += 0.50 + abs(progress)
                        else:
                            score -= 0.15 * min(progress, dist)

                    if (score + 1e-6) < best_score or (
                        abs(score - best_score) <= 1e-6 and dist < best_dist
                    ):
                        best = (gx, gy)
                        best_score = score
                        best_dist = dist
        return best

    @staticmethod
    def _grid_heur(a, b):
        return math.hypot(float(b[0] - a[0]), float(b[1] - a[1]))

    @staticmethod
    def _normalize_global_grid_planner(planner_name, legacy_any_angle):
        if not planner_name:
            return "theta" if legacy_any_angle else "astar4"
        aliases = {
            "theta": "theta",
            "theta*": "theta",
            "any_angle": "theta",
            "astar8": "astar8",
            "astar": "astar8",
            "a*": "astar8",
            "8way": "astar8",
            "8-way": "astar8",
            "diagonal": "astar8",
            "astar4": "astar4",
            "cardinal": "astar4",
            "4way": "astar4",
            "4-way": "astar4",
        }
        normalized = aliases.get(planner_name)
        if normalized is None:
            rospy.logwarn(
                "[astar] unsupported global_path_grid_planner=%s, falling back to astar8",
                planner_name,
            )
            return "astar8"
        return normalized

    def _global_grid_planner_label(self):
        if self.global_path_grid_planner == "theta":
            return "Theta*"
        if self.global_path_grid_planner == "astar4":
            return "A* (4-connected)"
        return "A* (8-connected)"

    def _grid_neighbors(self, blocked, cell, allow_diagonal=True):
        cx, cy = cell
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if allow_diagonal:
            nbrs.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
        out = []
        for dx, dy in nbrs:
            nx = cx + dx
            ny = cy + dy
            if not self._blocked_cell_is_free(blocked, nx, ny):
                continue
            if dx != 0 and dy != 0:
                if not self._blocked_cell_is_free(blocked, cx + dx, cy):
                    continue
                if not self._blocked_cell_is_free(blocked, cx, cy + dy):
                    continue
            out.append((nx, ny))
        return out

    @staticmethod
    def _reconstruct_grid_path(parent, goal_cell):
        path = []
        cur = goal_cell
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
            if not self._blocked_cell_is_free(blocked, x0, y0):
                return False
            if x0 == x1 and y0 == y1:
                return True
            e2 = 2 * err
            nx = x0
            ny = y0
            moved_x = False
            moved_y = False
            if e2 > -dy:
                err -= dy
                nx += sx
                moved_x = True
            if e2 < dx:
                err += dx
                ny += sy
                moved_y = True
            if moved_x and moved_y:
                if not self._blocked_cell_is_free(blocked, nx, y0):
                    return False
                if not self._blocked_cell_is_free(blocked, x0, ny):
                    return False
            x0, y0 = nx, ny

    def _simplify_grid_path(self, path, blocked):
        if not path:
            return path
        if len(path) <= 2:
            return path
        simplified = [path[0]]
        anchor_idx = 0
        while anchor_idx < len(path) - 1:
            farthest_idx = anchor_idx + 1
            probe_idx = farthest_idx + 1
            while probe_idx < len(path):
                if not self._has_line_of_sight(blocked, path[anchor_idx], path[probe_idx]):
                    break
                farthest_idx = probe_idx
                probe_idx += 1
            simplified.append(path[farthest_idx])
            anchor_idx = farthest_idx
        return simplified

    def _plan_single_grid_path(
        self,
        blocked,
        start_cell,
        goal_cell,
        planner_mode=None,
        cell_penalties=None,
    ):
        planner_mode = planner_mode or self.global_path_grid_planner
        if planner_mode == "theta":
            grid_path = self._theta_star_on_grid(
                blocked, start_cell, goal_cell, cell_penalties=cell_penalties
            )
            planner_name = "Theta*"
            if grid_path:
                grid_path = self._simplify_grid_path(grid_path, blocked)
            if not grid_path:
                grid_path = self._astar_on_grid(
                    blocked,
                    start_cell,
                    goal_cell,
                    cell_penalties=cell_penalties,
                )
                if grid_path:
                    grid_path = self._simplify_grid_path(grid_path, blocked)
                    planner_name = "A* (8-connected) fallback"
            return grid_path, planner_name

        allow_diagonal = planner_mode != "astar4"
        planner_name = "A* (8-connected)" if allow_diagonal else "A* (4-connected)"
        grid_path = self._astar_on_grid(
            blocked,
            start_cell,
            goal_cell,
            allow_diagonal=allow_diagonal,
            cell_penalties=cell_penalties,
        )
        if grid_path and allow_diagonal:
            grid_path = self._simplify_grid_path(grid_path, blocked)
        return grid_path, planner_name

    @staticmethod
    def _grid_path_similarity(path_a, path_b):
        if not path_a or not path_b:
            return 0.0
        cells_a = set(path_a)
        cells_b = set(path_b)
        denom = float(min(len(cells_a), len(cells_b)))
        if denom <= 0.0:
            return 0.0
        return float(len(cells_a & cells_b)) / denom

    def _candidate_penalty_offsets(self, g):
        res = max(1e-6, float(g.info.resolution))
        radius_cells = max(
            1, int(math.ceil(self.global_path_candidate_penalty_radius_m / res))
        )
        radius_sq = radius_cells * radius_cells
        offsets = []
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                dist_sq = dx * dx + dy * dy
                if dist_sq > radius_sq:
                    continue
                if dist_sq == 0:
                    weight = 1.0
                else:
                    dist = math.sqrt(float(dist_sq))
                    weight = max(0.20, 1.0 - dist / float(radius_cells + 1))
                offsets.append((dx, dy, weight))
        return offsets

    def _accumulate_candidate_penalties(self, penalty_map, path, offsets):
        if not path:
            return
        endpoint_skip = max(1, int(round(0.80 / max(1e-6, float(self.drivable_grid.info.resolution)))))
        if len(path) <= (2 * endpoint_skip + 1):
            cells = path[1:-1]
        else:
            cells = path[endpoint_skip:-endpoint_skip]
        for gx, gy in cells:
            for dx, dy, weight in offsets:
                key = (gx + dx, gy + dy)
                penalty_map[key] = penalty_map.get(key, 0.0) + (
                    self.global_path_candidate_penalty_cost * weight
                )

    def _generate_candidate_grid_paths(self, blocked, start_cell, goal_cell, primary_path):
        if not primary_path:
            return []

        candidate_paths = [list(primary_path)]
        if self.global_path_candidate_count <= 1:
            return candidate_paths

        penalty_map = {}
        offsets = self._candidate_penalty_offsets(self.drivable_grid)
        self._accumulate_candidate_penalties(penalty_map, primary_path, offsets)
        planner_mode = "astar4" if self.global_path_grid_planner == "astar4" else "astar8"
        max_attempts = max(
            self.global_path_candidate_count * 4,
            self.global_path_candidate_count + 2,
        )
        attempts = 0
        while len(candidate_paths) < self.global_path_candidate_count and attempts < max_attempts:
            attempts += 1
            alt_path, _ = self._plan_single_grid_path(
                blocked,
                start_cell,
                goal_cell,
                planner_mode=planner_mode,
                cell_penalties=penalty_map,
            )
            if not alt_path:
                break
            similarity = max(
                self._grid_path_similarity(alt_path, existing)
                for existing in candidate_paths
            )
            self._accumulate_candidate_penalties(penalty_map, alt_path, offsets)
            if similarity >= self.global_path_candidate_max_similarity:
                continue
            candidate_paths.append(list(alt_path))
        return candidate_paths

    def _grid_path_to_world_points(
        self,
        g,
        grid_path,
        start_xy,
        goal_xy,
        start_cell,
        goal_raw,
        goal_cell,
        goal_tail_blocked,
    ):
        snapped_start_xy = self._grid_cell_to_world(g, start_cell[0], start_cell[1])
        snapped_goal_xy = self._grid_cell_to_world(g, goal_cell[0], goal_cell[1])
        world_points = [tuple(start_xy)]
        if self._xy_distance(start_xy, snapped_start_xy) > 0.05:
            world_points.append(snapped_start_xy)
        for gx, gy in grid_path[1:-1]:
            world_points.append(self._grid_cell_to_world(g, gx, gy))
        clicked_goal_xy = (float(goal_xy[0]), float(goal_xy[1]))
        goal_gap_m = math.hypot(
            clicked_goal_xy[0] - float(snapped_goal_xy[0]),
            clicked_goal_xy[1] - float(snapped_goal_xy[1]),
        )
        goal_raw_is_free = self._blocked_cell_is_free(
            goal_tail_blocked, goal_raw[0], goal_raw[1]
        )
        snapped_goal_has_los_to_clicked = self._has_line_of_sight(
            goal_tail_blocked, goal_cell, goal_raw
        )
        extend_to_clicked_goal = goal_cell == goal_raw
        if (
            (not extend_to_clicked_goal)
            and goal_gap_m <= self.drivable_grid_goal_extension_max_gap_m
            and goal_raw_is_free
            and snapped_goal_has_los_to_clicked
        ):
            extend_to_clicked_goal = True
        preserve_goal_gap_cap_m = max(
            1e-6, float(self.drivable_grid_goal_extension_max_gap_m)
        )
        preserve_exact_goal = (
            self.preserve_user_goal_on_drivable_grid
            and goal_gap_m > 1e-6
            and goal_gap_m <= preserve_goal_gap_cap_m
        )
        if preserve_exact_goal and goal_cell != goal_raw:
            world_points.append(snapped_goal_xy)
        path_goal_xy = (
            clicked_goal_xy
            if (extend_to_clicked_goal or preserve_exact_goal)
            else snapped_goal_xy
        )
        world_points.append(path_goal_xy)
        return self._dedupe_world_points(world_points), snapped_goal_xy, goal_gap_m, goal_raw_is_free, snapped_goal_has_los_to_clicked, extend_to_clicked_goal, preserve_exact_goal

    def _astar_on_grid(self, blocked, start_cell, goal_cell, allow_diagonal=True, cell_penalties=None):
        if not blocked or not blocked[0]:
            return None
        if not self._blocked_cell_is_free(blocked, start_cell[0], start_cell[1]):
            return None
        if not self._blocked_cell_is_free(blocked, goal_cell[0], goal_cell[1]):
            return None

        pq = []
        heapq.heappush(pq, (self._grid_heur(start_cell, goal_cell), 0.0, start_cell))
        parent = {start_cell: None}
        g_cost = {start_cell: 0.0}

        while pq:
            _, gc, cur = heapq.heappop(pq)
            if cur == goal_cell:
                break
            if gc > g_cost.get(cur, float("inf")) + 1e-9:
                continue
            for nb in self._grid_neighbors(blocked, cur, allow_diagonal=allow_diagonal):
                step = self._grid_heur(cur, nb)
                penalty = float(cell_penalties.get(nb, 0.0)) if cell_penalties else 0.0
                ng = gc + step + penalty
                if ng >= g_cost.get(nb, float("inf")):
                    continue
                g_cost[nb] = ng
                parent[nb] = cur
                heapq.heappush(pq, (ng + self._grid_heur(nb, goal_cell), ng, nb))

        if goal_cell not in parent:
            return None
        return self._reconstruct_grid_path(parent, goal_cell)

    def _theta_star_on_grid(self, blocked, start_cell, goal_cell, cell_penalties=None):
        if not blocked or not blocked[0]:
            return None
        if not self._blocked_cell_is_free(blocked, start_cell[0], start_cell[1]):
            return None
        if not self._blocked_cell_is_free(blocked, goal_cell[0], goal_cell[1]):
            return None

        pq = []
        heapq.heappush(pq, (self._grid_heur(start_cell, goal_cell), start_cell))
        parent = {start_cell: None}
        g_cost = {start_cell: 0.0}
        closed = set()

        while pq:
            _, cur = heapq.heappop(pq)
            if cur in closed:
                continue
            if cur == goal_cell:
                break
            closed.add(cur)

            cur_parent = parent.get(cur)
            for nb in self._grid_neighbors(blocked, cur):
                penalty = float(cell_penalties.get(nb, 0.0)) if cell_penalties else 0.0
                best_parent = cur
                best_g = g_cost[cur] + self._grid_heur(cur, nb) + penalty

                if cur_parent is not None and self._has_line_of_sight(blocked, cur_parent, nb):
                    los_g = g_cost[cur_parent] + self._grid_heur(cur_parent, nb) + penalty
                    if los_g < best_g:
                        best_g = los_g
                        best_parent = cur_parent

                if best_g >= g_cost.get(nb, float("inf")):
                    continue
                g_cost[nb] = best_g
                parent[nb] = best_parent
                heapq.heappush(pq, (best_g + self._grid_heur(nb, goal_cell), nb))

        if goal_cell not in parent:
            return None
        return self._reconstruct_grid_path(parent, goal_cell)

    def _plan_with_drivable_grid(self, start_xy, goal_xy):
        g = self.drivable_grid
        if g is None:
            return None
        if int(g.info.width) <= 0 or int(g.info.height) <= 0:
            return None
        blocked = self._build_blocked_grid(g)
        goal_tail_clearance_radius = self._global_path_goal_tail_clearance_radius()
        if goal_tail_clearance_radius + 1e-6 < self._global_path_clearance_radius_m():
            goal_tail_blocked = self._build_blocked_grid(
                g, clearance_radius_m=goal_tail_clearance_radius
            )
        else:
            goal_tail_blocked = blocked

        start_raw = self._world_to_grid_cell(g, start_xy[0], start_xy[1])
        goal_raw = self._world_to_grid_cell(g, goal_xy[0], goal_xy[1])
        start_cell = self._nearest_free_start_grid_cell(g, blocked, start_xy, goal_xy)
        goal_cell = self._nearest_free_grid_cell(blocked, goal_raw)
        if start_cell is None or goal_cell is None:
            rospy.logwarn_throttle(
                1.0,
                "[astar] drivable-grid snap failed (start=%s goal=%s snapped_start=%s snapped_goal=%s)",
                str(start_raw),
                str(goal_raw),
                str(start_cell),
                str(goal_cell),
            )
            return None

        grid_path, planner_name = self._plan_single_grid_path(
            blocked,
            start_cell,
            goal_cell,
        )
        if not grid_path:
            rospy.logwarn_throttle(
                1.0,
                "[astar] drivable-grid path not found (start=%s goal=%s radius=%.2f m)",
                str(start_cell),
                str(goal_cell),
                self._global_path_clearance_radius_m(),
            )
            return None

        if self.debug_log_enable:
            rospy.loginfo_throttle(
                1.0,
                "[astar] drivable-grid planner=%s radius=%.2f m grid_pts=%d",
                planner_name,
                self._global_path_clearance_radius_m(),
                len(grid_path),
            )

        (
            world_points,
            snapped_goal_xy,
            goal_gap_m,
            goal_raw_is_free,
            snapped_goal_has_los_to_clicked,
            extend_to_clicked_goal,
            preserve_exact_goal,
        ) = self._grid_path_to_world_points(
            g,
            grid_path,
            start_xy,
            goal_xy,
            start_cell,
            goal_raw,
            goal_cell,
            goal_tail_blocked,
        )
        self._last_snapped_goal_xy = snapped_goal_xy
        if extend_to_clicked_goal and goal_cell != goal_raw and self.debug_log_enable:
            rospy.loginfo_throttle(
                1.0,
                "[astar] extending drivable-grid path from snapped goal to clicked goal (gap=%.2f m, limit=%.2f m)",
                goal_gap_m,
                self.drivable_grid_goal_extension_max_gap_m,
            )
        preserve_goal_gap_cap_m = max(
            1e-6, float(self.drivable_grid_goal_extension_max_gap_m)
        )
        if preserve_exact_goal and goal_cell != goal_raw and self.debug_log_enable:
            rospy.loginfo_throttle(
                1.0,
                "[astar] preserving exact clicked goal as terminal pose after snapped drivable-grid anchor (clicked=%.2f, %.2f snapped=%.2f, %.2f gap=%.2f m raw_free=%s los=%s)",
                clicked_goal_xy[0],
                clicked_goal_xy[1],
                float(snapped_goal_xy[0]),
                float(snapped_goal_xy[1]),
                goal_gap_m,
                "yes" if goal_raw_is_free else "no",
                "yes" if snapped_goal_has_los_to_clicked else "no",
            )
        elif (
            self.preserve_user_goal_on_drivable_grid
            and goal_gap_m > preserve_goal_gap_cap_m
            and self.debug_log_enable
        ):
            rospy.loginfo_throttle(
                1.0,
                "[astar] skipping exact clicked goal terminal pose because snapped gap is too large (clicked=%.2f, %.2f snapped=%.2f, %.2f gap=%.2f m limit=%.2f m raw_free=%s los=%s)",
                clicked_goal_xy[0],
                clicked_goal_xy[1],
                float(snapped_goal_xy[0]),
                float(snapped_goal_xy[1]),
                goal_gap_m,
                preserve_goal_gap_cap_m,
                "yes" if goal_raw_is_free else "no",
                "yes" if snapped_goal_has_los_to_clicked else "no",
            )
        elif (not extend_to_clicked_goal) and self.debug_log_enable:
            rospy.loginfo_throttle(
                1.0,
                "[astar] using snapped drivable-grid goal for display/path end (clicked=%.2f, %.2f snapped=%.2f, %.2f gap=%.2f m)",
                clicked_goal_xy[0],
                clicked_goal_xy[1],
                float(snapped_goal_xy[0]),
                float(snapped_goal_xy[1]),
                goal_gap_m,
            )
        candidate_grid_paths = self._generate_candidate_grid_paths(
            blocked,
            start_cell,
            goal_cell,
            grid_path,
        )
        candidate_world_paths = []
        for idx, cand_grid_path in enumerate(candidate_grid_paths):
            cand_world_points, _, _, _, _, _, _ = self._grid_path_to_world_points(
                g,
                cand_grid_path,
                start_xy,
                goal_xy,
                start_cell,
                goal_raw,
                goal_cell,
                goal_tail_blocked,
            )
            if idx > 0 and candidate_world_paths:
                similarity = max(
                    self._grid_path_similarity(cand_grid_path, candidate_grid_paths[j])
                    for j in range(min(idx, len(candidate_grid_paths)))
                )
                if similarity >= self.global_path_candidate_max_similarity:
                    continue
            candidate_world_paths.append(cand_world_points)
        if self.debug_log_enable and self.global_path_candidate_count > 1:
            rospy.loginfo_throttle(
                1.0,
                "[astar] generated %d/%d drivable-grid candidate paths",
                len(candidate_world_paths),
                self.global_path_candidate_count,
            )
        return {
            "selected_world_path": self._dedupe_world_points(world_points),
            "candidate_world_paths": candidate_world_paths,
        }

    # -------------------- Visualization --------------------
    def show_path(self, path, stamp=None):
        if not path: return
        if stamp is None: stamp = rospy.Time.now()
        p = Path(); p.header.frame_id = "map"; p.header.stamp = stamp
        pd = Path(); pd.header.frame_id = "map"; pd.header.stamp = stamp
        pw = Path(); pw.header.frame_id = "map"; pw.header.stamp = stamp

        world_points = []
        wgs_points = []
        for nid in path:
            n = self.findNodeById(nid)
            if n is None:
                continue
            world_points.append(self._xy_to_map(n.east, n.north))
            wgs_points.append((n.lat, n.lon))

        display_points = self._prepend_current_start_to_path_points(
            self._prepare_display_path(world_points)
        )
        display_yaws = self._apply_start_yaw_hint(self._path_yaws(display_points))
        for (x, y), yaw in zip(display_points, display_yaws):
            ps = PoseStamped(); ps.header = p.header
            ps.pose.position.x = x; ps.pose.position.y = y; ps.pose.position.z = 0.0
            self._set_pose_yaw(ps, yaw)
            p.poses.append(ps)

        viz_points = self._prepare_visualization_path(world_points)
        viz_yaws = self._apply_start_yaw_hint(self._path_yaws(viz_points))
        for (x, y), yaw in zip(viz_points, viz_yaws):
            pds = PoseStamped(); pds.header = pd.header
            pds.pose.position.x = x; pds.pose.position.y = y; pds.pose.position.z = 0.0
            self._set_pose_yaw(pds, yaw)
            pd.poses.append(pds)

        for lat, lon in wgs_points:
            pwps = PoseStamped(); pwps.header = pw.header
            pwps.pose.position.x = lat; pwps.pose.position.y = lon; pwps.pose.position.z = 0.0
            pw.poses.append(pwps)

        self.pub_path.publish(p)
        self.pub_path_display.publish(pd)
        self.pub_path_wgs84.publish(pw)

    def show_clicked_goal_marker(self):
        if self._goal_display_xy is None and self._goal_marker_xy is None:
            return
        marker_xy = self._goal_marker_xy if self._goal_marker_xy is not None else self._goal_display_xy
        gx, gy = marker_xy
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = rospy.Time.now()
        m.ns = "astar_clicked_goal"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = gx
        m.pose.position.y = gy
        m.pose.position.z = 0.12
        m.pose.orientation.w = 1.0
        m.scale.x = 0.28
        m.scale.y = 0.28
        m.scale.z = 0.28
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 1.0
        m.color.a = 0.95
        self.pub_goal_marker.publish(m)

        requested = Marker()
        requested.header.frame_id = "map"
        requested.header.stamp = m.header.stamp
        requested.ns = "astar_requested_goal"
        requested.id = 1
        requested.type = Marker.SPHERE
        requested.pose.orientation.w = 1.0
        if (
            self._goal_display_xy is not None
            and math.hypot(
                float(self._goal_display_xy[0]) - float(gx),
                float(self._goal_display_xy[1]) - float(gy),
            ) > 0.05
        ):
            requested.action = Marker.ADD
            requested.pose.position.x = float(self._goal_display_xy[0])
            requested.pose.position.y = float(self._goal_display_xy[1])
            requested.pose.position.z = 0.08
            requested.scale.x = 0.18
            requested.scale.y = 0.18
            requested.scale.z = 0.18
            requested.color.r = 0.8
            requested.color.g = 0.8
            requested.color.b = 0.8
            requested.color.a = 0.45
        else:
            requested.action = Marker.DELETE
        self.pub_goal_marker.publish(requested)

    def show_server_dst_nodes(self):
        npt = len(self.server_dst_node_list)
        if npt == 0: return
        hues = [0.67] if npt == 1 else [(0.67 - i/float(npt)) % 1.0 for i in range(npt)]
        colors = [struct.unpack('I', struct.pack('BBBB',
                   int(colorsys.hsv_to_rgb(h,1.0,1.0)[0]*255),
                   int(colorsys.hsv_to_rgb(h,1.0,1.0)[1]*255),
                   int(colorsys.hsv_to_rgb(h,1.0,1.0)[2]*255),
                   255))[0] for h in hues]
        fields = [PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                  PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                  PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                  PointField(name="rgba", offset=12, datatype=PointField.UINT32, count=1)]
        buf = []
        for nid, color in zip(self.server_dst_node_list, colors):
            n = self.findNodeById(nid)
            if n is None: continue
            x, y = self._xy_to_map(n.east, n.north)
            buf.append(struct.pack('fffI', float(x), float(y), 0.0, int(color)))
        header = Header(frame_id="map", stamp=rospy.Time.now())
        cloud = PointCloud2(header=header, height=1, width=len(buf), is_dense=True, is_bigendian=False,
                            fields=fields, point_step=16, row_step=16*len(buf), data=b''.join(buf))
        self.pub_server_dst_list.publish(cloud)

    def visualize_graph(self):
        m = Marker(); m.header.frame_id = "map"; m.header.stamp = rospy.Time.now()
        m.type = Marker.LINE_LIST; m.action = Marker.ADD; m.scale.x = 0.3
        m.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        white = ColorRGBA(1,1,1,1)
        for e in self.edge_list:
            sx, sy = self._xy_to_map(e.src.east, e.src.north)
            dx, dy = self._xy_to_map(e.dst.east, e.dst.north)
            m.points.append(Point(sx, sy, -0.1)); m.points.append(Point(dx, dy, -0.1))
            m.colors.append(white);               m.colors.append(white)
        self.pub_marker.publish(m)

    # -------------------- Callbacks --------------------
    def _find_nearest_node_simple(self, x, y):
        best=None; best_d2=1e18
        for n in self.node_list:
            nx, ny = self._xy_to_map(n.east, n.north)
            d2 = (nx-x)**2 + (ny-y)**2
            if d2 < best_d2: best_d2=d2; best=n
        return best

    def _reachable_node_ids(self, start_id):
        start = self.findNodeById(start_id)
        if start is None:
            return set()
        visited = set([start.id])
        queue = [start]
        while queue:
            cur = queue.pop(0)
            for e in self.edges(cur):
                nid = e.dst.id
                if nid in visited:
                    continue
                visited.add(nid)
                queue.append(e.dst)
        return visited

    def _find_nearest_node_in_set(self, x, y, node_ids):
        best = None
        best_d2 = 1e18
        for nid in node_ids:
            n = self.findNodeById(nid)
            if n is None:
                continue
            nx, ny = self._xy_to_map(n.east, n.north)
            d2 = (nx - x) ** 2 + (ny - y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = n
        return best

    def _snap_and_update_from_position(self, position):
        x, y = position.x, position.y
        edge, t, _, _, _, elen = self._nearest_projection_on_edges(x, y)
        if edge is None:
            return self._find_nearest_node_simple(x, y)

        (acc_edge, acc_t, progressed_m) = self._accept_or_clamp_projection(edge, t, elen)
        self._snap_last_update_s += progressed_m

        src_id, dst_id = acc_edge
        src = self.findNodeById(src_id); dst = self.findNodeById(dst_id)
        chosen = dst if acc_t >= 0.5 else src

        if self._snap_last_update_s >= self._snap_progress_min_step_m:
            self._snap_last_update_s = 0.0
            return chosen
        return self.findNodeById(self.start_id) if self.start_id is not None else chosen

    def _snap_goal_node_from_xy(self, x, y):
        px, py = x, y
        edge, t, px, py, _, _ = self._nearest_projection_on_edges(x, y)
        if edge is None:
            chosen = self._find_nearest_node_simple(x, y)
        else:
            chosen = edge.dst if t >= 0.5 else edge.src

        if self.start_id is not None:
            reachable = self._reachable_node_ids(self.start_id)
            if reachable and chosen is not None and chosen.id not in reachable:
                fallback = self._find_nearest_node_in_set(x, y, reachable)
                if fallback is not None:
                    chosen = fallback
                    px, py = self._xy_to_map(chosen.east, chosen.north)

        if chosen is None:
            return None, x, y
        return chosen, px, py

    def _allow_graph_fallback_for_goal(self):
        if not self.enable_graph_fallback:
            return False
        if (not self.use_drivable_grid_global) or self.drivable_grid is None:
            return True
        if self._goal_display_xy is None or self._last_graph_goal_snap_xy is None:
            return True
        gap_m = math.hypot(
            float(self._goal_display_xy[0]) - float(self._last_graph_goal_snap_xy[0]),
            float(self._goal_display_xy[1]) - float(self._last_graph_goal_snap_xy[1]),
        )
        if gap_m <= self.drivable_grid_graph_fallback_max_goal_gap_m:
            return True
        rospy.logwarn_throttle(
            1.0,
            "[astar] skipping graph fallback: snapped graph goal is %.2f m away from clicked goal (limit=%.2f m)",
            gap_m,
            self.drivable_grid_graph_fallback_max_goal_gap_m,
        )
        return False

    def callback_start(self, data):
        self._manual_start_reset_seq += 1
        n = self._snap_and_update_from_position(data.pose.pose.position)
        self._display_start_xy = (
            float(data.pose.pose.position.x),
            float(data.pose.pose.position.y),
        )
        self._display_start_yaw = self._quat_to_yaw(data.pose.pose.orientation)
        if n:
            self.start_id = n.id; self.start_init_flag = True
            if self.debug_log_enable:
                rospy.loginfo(f"[astar] start set by /initialpose -> node {n.id}")

    def pose_callback(self, data):
        n = self._snap_and_update_from_position(data.pose.pose.position)
        self._display_start_xy = (
            float(data.pose.pose.position.x),
            float(data.pose.pose.position.y),
        )
        self._display_start_yaw = self._quat_to_yaw(data.pose.pose.orientation)
        if n:
            self.start_id = n.id; self.start_init_flag = True

    def callback_goal_from_rviz(self, data):
        goal_x = data.pose.position.x
        goal_y = data.pose.position.y
        self._goal_display_xy = (goal_x, goal_y)
        self._goal_marker_xy = (goal_x, goal_y)
        n, snap_x, snap_y = self._snap_goal_node_from_xy(goal_x, goal_y)
        self._last_graph_goal_snap_xy = (float(snap_x), float(snap_y))
        if n:
            self.goal_id = n.id
        self.new_goal_flag = True
        if self.debug_log_enable:
            if self.use_drivable_grid_global:
                rospy.loginfo(
                    "[astar] goal set by RViz -> drivable-grid target (clicked=%.2f, %.2f graph_fallback=%.2f, %.2f)",
                    goal_x, goal_y, snap_x, snap_y
                )
            elif n and self.goal_id == n.id:
                rospy.loginfo(
                    "[astar] goal set by RViz -> node %d (clicked=%.2f, %.2f snapped=%.2f, %.2f)",
                    n.id, goal_x, goal_y, snap_x, snap_y
                )

    def callback_goal_from_server(self, data):
        if self.server_dst_node_list and 0 <= data.Cmd_dest_index < len(self.server_dst_node_list):
            nid = self.server_dst_node_list[data.Cmd_dest_index]
            n = self.findNodeById(nid)
            if n and self.goal_id != n.id:
                self.goal_id = n.id; self.new_goal_flag = True
                if self.debug_log_enable:
                    rospy.loginfo(f"[astar] goal set by server index -> node {n.id}")
        elif data.Cmd_dest_lat > 0.01 and data.Cmd_dest_lon > 0.01:
            # WGS84 dest -> local map XY (mode-aware)
            if self.mode == "UTM":
                ue, un, _, _ = utm.from_latlon(data.Cmd_dest_lat, data.Cmd_dest_lon)
                mx, my = self._xy_to_map(ue - self._utm_ref_e, un - self._utm_ref_n)
            else:
                e, n = self._ll_to_enu(data.Cmd_dest_lat, data.Cmd_dest_lon)
                mx, my = self._xy_to_map(e, n)
            self._goal_display_xy = (mx, my)
            self._goal_marker_xy = (mx, my)
            g, snap_x, snap_y = self._snap_goal_node_from_xy(mx, my)
            self._last_graph_goal_snap_xy = (float(snap_x), float(snap_y))
            if g:
                self.goal_id = g.id
            self.new_goal_flag = True
            if self.debug_log_enable:
                if self.use_drivable_grid_global:
                    rospy.loginfo(
                        "[astar] goal set by server WGS84 -> drivable-grid target (goal=%.2f, %.2f graph_fallback=%.2f, %.2f)",
                        mx, my, snap_x, snap_y
                    )
                elif g and self.goal_id == g.id:
                    rospy.loginfo(
                        "[astar] goal set by server WGS84 -> node %d (goal=%.2f, %.2f snapped=%.2f, %.2f)",
                        g.id, mx, my, snap_x, snap_y
                    )

    # -------------------- Path publish control --------------------
    def publish_path_if_changed(self, path_nodes):
        now = rospy.Time.now()
        changed = (self._last_path_nodes != path_nodes)
        do_periodic = False
        if not changed and self.path_repub_period > 0.0:
            tnow = time.monotonic()
            if (tnow - self._last_path_pub_t) >= self.path_repub_period:
                do_periodic = True; self._last_path_pub_t = tnow

        if changed:
            self._last_path_nodes = list(path_nodes)
            self._last_world_path_signature = None
            self._last_world_path = None
            self._last_path_pub_t = time.monotonic()
            self._capture_published_path_context()
            self._publish_path_fallback_state(False, force=True)
            msg = Int32MultiArray(); msg.data = path_nodes
            self.pub_path_node_id_list.publish(msg)
            self.show_path(path_nodes, stamp=now)
            if self.debug_log_enable:
                rospy.loginfo(f"[astar] path published ({len(path_nodes)} nodes) [mode={self.mode}]")
        elif do_periodic:
            self._capture_published_path_context()
            self._publish_path_fallback_state(False)
            self.show_path(path_nodes, stamp=now)
            if self.debug_log_enable:
                rospy.loginfo("[astar] path republished (periodic)")
        elif self._path_is_fallback:
            self._capture_published_path_context()
            self._publish_path_fallback_state(False)
            self._last_path_pub_t = time.monotonic()
            self.show_path(path_nodes, stamp=now)
            if self.debug_log_enable:
                rospy.loginfo("[astar] path republished (fallback cleared)")

# -------------------- Main --------------------
if __name__ == "__main__":
    try:
        rospy.init_node('astar_map_node')
        _ = rospy.Publisher('/path', Path, queue_size=10)  # legacy compat

        a = AStarPlanner()
        osm = rospy.get_param("astar_map_node/osm_file")
        ref = rospy.get_param("astar_map_node/ref_file", "")
        a.set_map_sources(osm, ref)
        a.load_osm_data(osm, ref)

        dst_str = rospy.get_param('~server_dst_node_list', "")
        if dst_str:
            a.set_dst_node_list([int(s) for s in dst_str.split(',') if s.strip()])

        rate = rospy.Rate(a.planner_loop_hz)
        path_nodes = []
        world_path = None
        candidate_world_paths = []

        while not rospy.is_shutdown():
            if a.consume_reload_request():
                if a.reload_map():
                    path_nodes = []
                    world_path = None
                    candidate_world_paths = []
                    a.clear_published_path()
            a.visualize_graph()
            a.show_clicked_goal_marker()
            a.show_server_dst_nodes()

            if a._should_replan_active_goal():
                prev_world_path = (
                    list(world_path)
                    if world_path
                    else (list(a._last_world_path) if a._last_world_path else None)
                )
                prev_candidate_world_paths = list(candidate_world_paths)
                prev_path_nodes = (
                    list(path_nodes)
                    if path_nodes
                    else (list(a._last_path_nodes) if a._last_path_nodes else [])
                )
                new_world_path = None
                new_candidate_world_paths = []
                new_path_nodes = []
                if (
                    a.use_drivable_grid_global
                    and a.drivable_grid is not None
                    and a._display_start_xy is not None
                    and a._goal_display_xy is not None
                ):
                    grid_plan_result = a._plan_with_drivable_grid(
                        a._display_start_xy, a._goal_display_xy
                    )
                    if grid_plan_result:
                        new_world_path = grid_plan_result.get("selected_world_path")
                        new_candidate_world_paths = list(
                            grid_plan_result.get("candidate_world_paths", [])
                        )
                if (not new_world_path) and a._allow_graph_fallback_for_goal():
                    a.graph_setup()
                    new_path_nodes = a.planning(a.start_id, a.goal_id)

                    attempt = 0
                    while (
                        a.jump_guard_enable
                        and new_path_nodes
                        and attempt < a.jump_guard_max_attempts
                    ):
                        validated = a.validate_or_blacklist(new_path_nodes)
                        if validated is not None:
                            new_path_nodes = validated
                            break
                        a.graph_setup()
                        new_path_nodes = a.planning(a.start_id, a.goal_id)
                        attempt += 1
                a.new_goal_flag = False
                replan_success = bool(new_world_path or new_path_nodes)
                a._mark_plan_context(replan_success)
                if new_world_path:
                    world_path = list(new_world_path)
                    candidate_world_paths = list(new_candidate_world_paths)
                    path_nodes = []
                    a.publish_candidate_world_paths_if_changed(
                        candidate_world_paths, stamp=rospy.Time.now()
                    )
                    a.publish_world_path_if_changed(world_path)
                elif new_path_nodes:
                    world_path = None
                    candidate_world_paths = []
                    path_nodes = list(new_path_nodes)
                    a.clear_candidate_world_paths(stamp=rospy.Time.now())
                    a.publish_path_if_changed(path_nodes)
                else:
                    keep_prev_path, keep_reason = a._can_keep_last_path_on_replan_failure()
                    if keep_prev_path and (prev_world_path or prev_path_nodes):
                        world_path = prev_world_path
                        candidate_world_paths = prev_candidate_world_paths
                        path_nodes = prev_path_nodes
                        a._publish_path_fallback_state(True)
                        if world_path:
                            a.publish_candidate_world_paths_if_changed(
                                candidate_world_paths, stamp=rospy.Time.now()
                            )
                        rospy.logwarn_throttle(
                            1.0,
                            "[astar] replanning failed; keeping last valid global path until replacement is ready",
                        )
                    else:
                        world_path = None
                        candidate_world_paths = []
                        path_nodes = []
                        a.clear_published_path(stamp=rospy.Time.now())
                        rospy.logwarn_throttle(
                            1.0,
                            "[astar] path not found; previous global path dropped (%s)",
                            keep_reason,
                        )

            if (not a._path_is_fallback) and world_path and a.path_repub_period > 0.0:
                a.publish_world_path_if_changed(world_path)
                if candidate_world_paths:
                    a.publish_candidate_world_paths_if_needed(
                        candidate_world_paths, stamp=rospy.Time.now()
                    )
            elif (not a._path_is_fallback) and path_nodes and a.path_repub_period > 0.0:
                a.publish_path_if_changed(path_nodes)

            rate.sleep()

    except rospy.ROSInterruptException:
        pass
