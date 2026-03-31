#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import heapq
import math
from collections import deque

import rospy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray


class ConstrainedLocalReplanner:
    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic", "/lio_localizer/odometry/optimization")
        self.global_path_topic = rospy.get_param("~global_path_topic", "/astar/path")
        self.drivable_grid_topic = rospy.get_param("~drivable_grid_topic", "/lio_sam/drivable_area/grid")
        self.dynamic_risk_grid_topic = rospy.get_param("~dynamic_risk_grid_topic", "/planning/dynamic_risk_grid")
        self.local_path_topic = rospy.get_param("~local_path_topic", "/planning/local_path")
        self.avoidance_path_topic = rospy.get_param("~avoidance_path_topic", "/planning/avoidance_path")
        self.path_history_topic = rospy.get_param("~path_history_topic", "/planning/path_history")
        self.travel_history_topic = rospy.get_param("~travel_history_topic", "/planning/travel_history")
        self.pointcloud_topic = rospy.get_param("~pointcloud_topic", "/ouster/points")
        self.use_direct_goal = bool(rospy.get_param("~use_direct_goal", False))
        self.direct_goal_topic = rospy.get_param("~direct_goal_topic", "/move_base_simple/goal")
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
        self.window_margin_m = max(1.0, float(rospy.get_param("~window_margin_m", 12.0)))
        legacy_robot_radius = max(0.05, float(rospy.get_param("~robot_radius_m", 0.45)))
        self.robot_width_m = max(
            0.05, float(rospy.get_param("~robot_width_m", 2.0 * legacy_robot_radius))
        )
        self.robot_length_m = max(
            0.05, float(rospy.get_param("~robot_length_m", self.robot_width_m))
        )
        self.robot_radius = 0.5 * math.hypot(self.robot_length_m, self.robot_width_m)
        self.footprint_padding_m = max(0.0, float(rospy.get_param("~footprint_padding_m", 0.0)))
        self.risk_threshold = int(rospy.get_param("~risk_occupied_threshold", 45))
        self.max_expand = max(100, int(rospy.get_param("~max_expand", 25000)))
        self.replan_hz = max(1.0, float(rospy.get_param("~replan_hz", 6.0)))
        self.simplify_stride = max(1, int(rospy.get_param("~simplify_stride", 1)))
        self.published_path_spacing_m = max(
            0.05, float(rospy.get_param("~published_path_spacing_m", 0.25))
        )
        self.path_history_max_paths = max(
            1, int(rospy.get_param("~path_history_max_paths", 12))
        )
        self.travel_history_max_points = max(
            2, int(rospy.get_param("~travel_history_max_points", 400))
        )
        self.travel_history_spacing_m = max(
            0.02, float(rospy.get_param("~travel_history_spacing_m", 0.05))
        )
        self.obstacle_min_z = float(rospy.get_param("~obstacle_min_z", -0.15))
        self.obstacle_max_z = float(rospy.get_param("~obstacle_max_z", 1.5))
        self.obstacle_max_range_m = max(1.0, float(rospy.get_param("~obstacle_max_range_m", 12.0)))
        self.obstacle_downsample = max(1, int(rospy.get_param("~obstacle_downsample", 6)))
        self.pointcloud_cluster_resolution_m = max(
            0.05, float(rospy.get_param("~pointcloud_cluster_resolution_m", 0.15))
        )
        self.pointcloud_min_cluster_points = max(
            1, int(rospy.get_param("~pointcloud_min_cluster_points", 4))
        )
        self.use_pointcloud_static_blocking = bool(
            rospy.get_param("~use_pointcloud_static_blocking", True)
        )
        self.pointcloud_static_block_margin_m = max(
            0.0, float(rospy.get_param("~pointcloud_static_block_margin_m", 0.10))
        )
        self.obstacle_block_margin_m = max(
            0.05, float(rospy.get_param("~obstacle_block_margin_m", 0.35))
        )
        self.use_pointcloud_avoidance_trigger = bool(
            rospy.get_param("~use_pointcloud_avoidance_trigger", False)
        )
        self.avoidance_trigger_margin_m = max(
            0.05, float(rospy.get_param("~avoidance_trigger_margin_m", 0.20))
        )
        self.avoidance_trigger_ahead_m = max(
            1.0, float(rospy.get_param("~avoidance_trigger_ahead_m", 8.0))
        )
        self.risk_block_confirm_cells = max(
            1, int(rospy.get_param("~risk_block_confirm_cells", 2))
        )
        self.avoidance_hold_s = max(0.0, float(rospy.get_param("~avoidance_hold_s", 1.5)))
        self.avoidance_clear_confirm_cycles = max(
            1, int(rospy.get_param("~avoidance_clear_confirm_cycles", 6))
        )
        self.avoidance_reuse_on_failure_s = max(
            0.0, float(rospy.get_param("~avoidance_reuse_on_failure_s", 0.5))
        )
        self.avoidance_reuse_max_deviation_m = max(
            0.0, float(rospy.get_param("~avoidance_reuse_max_deviation_m", 0.8))
        )
        self.avoidance_branch_backtrack_cells = max(
            0, int(rospy.get_param("~avoidance_branch_backtrack_cells", 2))
        )
        self.avoidance_rejoin_min_distance_m = max(
            0.3, float(rospy.get_param("~avoidance_rejoin_min_distance_m", 1.0))
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

        self.have_odom = False
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.global_path = None
        self.drivable_grid = None
        self.risk_grid = None
        self.direct_goal = None
        self.cached_direct_goal_cell = None
        self.frozen_direct_goal_cell = None
        self.frozen_direct_grid_path = None
        self.frozen_direct_start_xy = None
        self.frozen_direct_goal_xy = None
        self.last_published_goal_cell = None
        self.last_published_end_cell = None
        self.last_path_publish_sec = 0.0
        self.obstacle_points_map = []
        self.avoidance_active = False
        self.avoidance_clear_count = 0
        self.last_avoidance_publish_sec = 0.0
        self.last_avoidance_grid_path = None
        self.last_avoidance_solution_sec = 0.0
        self.path_history_entries = deque(maxlen=self.path_history_max_paths)
        self.path_history_next_id = 0
        self.last_history_signature = {"local": None, "avoidance": None}
        self.travel_history_points = deque(maxlen=self.travel_history_max_points)

        self.pub_local_path = rospy.Publisher(self.local_path_topic, Path, queue_size=2)
        self.pub_avoidance_path = rospy.Publisher(self.avoidance_path_topic, Path, queue_size=2)
        self.pub_path_history = rospy.Publisher(
            self.path_history_topic, MarkerArray, queue_size=2, latch=True
        )
        self.pub_travel_history = rospy.Publisher(
            self.travel_history_topic, Marker, queue_size=2, latch=True
        )
        self.sub_odom = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=20)
        self.sub_global = rospy.Subscriber(self.global_path_topic, Path, self.global_path_callback, queue_size=5)
        self.sub_drivable = rospy.Subscriber(self.drivable_grid_topic, OccupancyGrid, self.drivable_grid_callback, queue_size=3)
        self.sub_risk = rospy.Subscriber(self.dynamic_risk_grid_topic, OccupancyGrid, self.risk_grid_callback, queue_size=3)
        self.sub_cloud = rospy.Subscriber(self.pointcloud_topic, PointCloud2, self.cloud_callback, queue_size=1)
        self.sub_direct_goal = None
        if self.use_direct_goal:
            self.sub_direct_goal = rospy.Subscriber(self.direct_goal_topic, PoseStamped, self.direct_goal_callback, queue_size=2)

        self._clear_path_history()
        self._clear_travel_history()
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.replan_hz), self.on_timer)
        rospy.loginfo(
            "constrained_local_replanner started | global=%s drivable=%s risk=%s local=%s avoidance=%s direct_goal=%s(%s) footprint=%.2fm x %.2fm freeze_first=%s avoid=%s",
            self.global_path_topic,
            self.drivable_grid_topic,
            self.dynamic_risk_grid_topic,
            self.local_path_topic,
            self.avoidance_path_topic,
            "on" if self.use_direct_goal else "off",
            self.direct_goal_topic,
            self.robot_length_m,
            self.robot_width_m,
            "on" if self.freeze_path_on_first_plan else "off",
            "on" if self.enable_avoidance_path else "off",
        )

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.odom_x = float(p.x)
        self.odom_y = float(p.y)
        self.odom_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.have_odom = True
        self._record_travel_history_point(self.odom_x, self.odom_y)

    def global_path_callback(self, msg):
        self.global_path = msg

    def drivable_grid_callback(self, msg):
        self.drivable_grid = msg

    def risk_grid_callback(self, msg):
        self.risk_grid = msg

    def _local_to_map(self, x, y):
        c = math.cos(self.odom_yaw)
        s = math.sin(self.odom_yaw)
        mx = self.odom_x + c * x - s * y
        my = self.odom_y + s * x + c * y
        return mx, my

    def _pointcloud_cluster_cell(self, x, y):
        res = max(1e-3, self.pointcloud_cluster_resolution_m)
        return (int(math.floor(x / res)), int(math.floor(y / res)))

    def cloud_callback(self, msg):
        if not self.have_odom:
            return
        raw_pts = []
        cluster_counts = {}
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
                if z < self.obstacle_min_z or z > self.obstacle_max_z:
                    continue
                if x * x + y * y > rr:
                    continue
                if abs(x) <= self.self_filter_radius_x and abs(y) <= self.self_filter_radius_y:
                    continue
                raw_pts.append((x, y))
                cell = self._pointcloud_cluster_cell(x, y)
                cluster_counts[cell] = cluster_counts.get(cell, 0) + 1

            if self.pointcloud_min_cluster_points <= 1:
                filtered_pts = raw_pts
            else:
                filtered_pts = []
                for x, y in raw_pts:
                    cx, cy = self._pointcloud_cluster_cell(x, y)
                    support = 0
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            support += cluster_counts.get((cx + dx, cy + dy), 0)
                    if support >= self.pointcloud_min_cluster_points:
                        filtered_pts.append((x, y))

            self.obstacle_points_map = [self._local_to_map(x, y) for x, y in filtered_pts]
        except Exception as e:
            rospy.logwarn_throttle(1.0, "constrained_local_replanner cloud error: %s", str(e))

    def direct_goal_callback(self, msg):
        self.direct_goal = msg
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
        self.last_avoidance_solution_sec = 0.0
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

    def _inflate_blocked(self, dg, rg):
        w = int(dg.info.width)
        h = int(dg.info.height)
        res = float(dg.info.resolution)
        half_length_cells = max(
            1, int(math.ceil((0.5 * self.robot_length_m + self.footprint_padding_m) / max(1e-3, res)))
        )
        half_width_cells = max(
            1, int(math.ceil((0.5 * self.robot_width_m + self.footprint_padding_m) / max(1e-3, res)))
        )
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
                for dx in range(-half_length_cells, half_length_cells + 1):
                    for dy in range(-half_width_cells, half_width_cells + 1):
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            out[ny][nx] = True
        out, _ = self._overlay_pointcloud_obstacles(
            out,
            dg,
            keep_cells=None,
            enabled=self.use_pointcloud_static_blocking,
            margin_m=self.pointcloud_static_block_margin_m,
        )
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

    def _resolve_direct_goal_cell(self, blocked, raw_goal_cell):
        if self.cached_direct_goal_cell is not None:
            cgx, cgy = self.cached_direct_goal_cell
            if self._in_bounds_blocked(blocked, cgx, cgy) and not blocked[cgy][cgx]:
                return self.cached_direct_goal_cell
        self.cached_direct_goal_cell = self._nearest_free_cell(blocked, raw_goal_cell)
        return self.cached_direct_goal_cell

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

    def _astar(self, blocked, start, goal, allow_best_effort=False):
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

        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
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

    def _publish_grid_path(self, publisher, grid_path, dg, stamp, start_xy=None, end_xy=None):
        out = Path()
        out.header.stamp = stamp
        out.header.frame_id = dg.header.frame_id if dg.header.frame_id else "map"
        world_points = self._grid_path_to_world_points(
            grid_path,
            dg,
            start_xy=start_xy,
            end_xy=end_xy,
        )

        if len(world_points) < 2:
            return

        sampled_points = self._sample_world_points(world_points)

        for x, y in sampled_points:
            ps = PoseStamped()
            ps.header = out.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            out.poses.append(ps)
        if len(out.poses) >= 2:
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

    @staticmethod
    def _path_signature_from_points(points):
        return tuple((round(float(x), 2), round(float(y), 2)) for x, y in points)

    def _publish_path_history_markers(self):
        markers = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        branch_entries = [entry for entry in self.path_history_entries if entry["source"] == "avoidance"]
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
            for x, y in entry["points"]:
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
            entry_marker.pose.position.x = float(entry["points"][0][0])
            entry_marker.pose.position.y = float(entry["points"][0][1])
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
            exit_marker.pose.position.x = float(entry["points"][-1][0])
            exit_marker.pose.position.y = float(entry["points"][-1][1])
            exit_marker.pose.position.z = 0.14
            markers.markers.append(exit_marker)
        self.pub_path_history.publish(markers)

    def _publish_travel_history_marker(self):
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
        marker.scale.x = 0.10
        marker.color.a = 0.80
        marker.color.r = 0.10
        marker.color.g = 0.45
        marker.color.b = 0.95
        for x, y in self.travel_history_points:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.02
            marker.points.append(p)
        self.pub_travel_history.publish(marker)

    def _record_travel_history_point(self, x, y):
        x = float(x)
        y = float(y)
        if self.travel_history_points:
            lx, ly = self.travel_history_points[-1]
            if math.hypot(x - lx, y - ly) < self.travel_history_spacing_m:
                return
        self.travel_history_points.append((x, y))
        self._publish_travel_history_marker()

    def _clear_travel_history(self):
        self.travel_history_points.clear()
        self._publish_travel_history_marker()

    def _record_path_history(self, source, sampled_points, frame_id):
        if source != "avoidance":
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
        self.last_history_signature = {"local": None, "avoidance": None}
        self._publish_path_history_markers()

    def _publish_local_path(self, grid_path, dg, stamp, start_xy=None, end_xy=None):
        sampled_points, frame_id = self._publish_grid_path(
            self.pub_local_path,
            grid_path,
            dg,
            stamp,
            start_xy=start_xy,
            end_xy=end_xy,
        )

    def _publish_avoidance_path(self, grid_path, dg, stamp, history_points=None, start_xy=None, end_xy=None, record_history=True):
        sampled_points, frame_id = self._publish_grid_path(
            self.pub_avoidance_path,
            grid_path,
            dg,
            stamp,
            start_xy=start_xy,
            end_xy=end_xy,
        )
        self.last_avoidance_grid_path = list(grid_path) if grid_path is not None else None
        if record_history:
            self._record_path_history(
                "avoidance",
                history_points if history_points is not None else sampled_points,
                frame_id,
            )

    def _publish_world_path(self, world_points, frame_id, stamp):
        out = Path()
        out.header.stamp = stamp
        out.header.frame_id = frame_id if frame_id else "map"
        for x, y in world_points:
            ps = PoseStamped()
            ps.header = out.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            out.poses.append(ps)
        if len(out.poses) >= 2:
            self.pub_local_path.publish(out)

    @staticmethod
    def _dedupe_world_points(world_points):
        deduped = []
        for x, y in world_points:
            if deduped and math.hypot(float(x) - deduped[-1][0], float(y) - deduped[-1][1]) <= 1e-3:
                continue
            deduped.append((float(x), float(y)))
        return deduped

    def _publish_nominal_local_segment(self, pts, i0, ig, blocked, start_cell, dg, stamp):
        if not pts:
            return False
        end_idx = max(i0 + 1, ig)
        segment_world = [pts[i] for i in range(i0, min(len(pts), end_idx + 1))]
        if not segment_world:
            return False
        segment_world[0] = (self.odom_x, self.odom_y)
        segment_world = self._dedupe_world_points(segment_world)
        if len(segment_world) < 2:
            return False

        grid_path = []
        for wx, wy in segment_world:
            gx, gy = self._world_to_grid(dg, wx, wy)
            if not self._in_bounds_blocked(blocked, gx, gy):
                return False
            grid_path.append((gx, gy))

        if self._path_blocked_ahead(
            grid_path,
            blocked,
            start_cell,
            float(dg.info.resolution),
            max_check_m=self.lookahead_m,
        ):
            return False

        sampled_points = self._sample_world_points(segment_world)
        self._publish_world_path(sampled_points, dg.header.frame_id, stamp)
        return True

    def _publish_empty_path(self, publisher, frame_id, stamp):
        out = Path()
        out.header.stamp = stamp
        out.header.frame_id = frame_id if frame_id else "map"
        publisher.publish(out)

    def _clear_avoidance_path(self, frame_id, stamp, force=False):
        if self.avoidance_active and (not force):
            now_sec = stamp.to_sec()
            if now_sec > 0.0 and (now_sec - self.last_avoidance_publish_sec) < self.avoidance_hold_s:
                return
            self.avoidance_clear_count += 1
            if self.avoidance_clear_count < self.avoidance_clear_confirm_cycles:
                return
        self._publish_empty_path(self.pub_avoidance_path, frame_id, stamp)
        if self.avoidance_active:
            self.avoidance_active = False
            rospy.loginfo("constrained_local_replanner: avoidance path cleared")
        self.avoidance_clear_count = 0
        self.last_avoidance_publish_sec = 0.0
        self.last_avoidance_grid_path = None
        self.last_avoidance_solution_sec = 0.0

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

    def _republish_last_avoidance_path(self, dg, stamp):
        if self.last_avoidance_grid_path is None or len(self.last_avoidance_grid_path) < 2:
            return False
        if self.last_avoidance_solution_sec <= 0.0:
            return False
        age_s = stamp.to_sec() - self.last_avoidance_solution_sec
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
        self._publish_avoidance_path(
            self.last_avoidance_grid_path,
            dg,
            stamp,
            record_history=False,
        )
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

    def _debug_avoidance_log(self, message):
        if not self.debug_avoidance_logging:
            return
        rospy.loginfo_throttle(self.debug_avoidance_log_period_s, message)

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

    def _overlay_pointcloud_obstacles(self, blocked, dg, keep_cells=None, enabled=True, margin_m=None):
        if (not enabled) or (not self.obstacle_points_map):
            return [row[:] for row in blocked], 0

        res = max(1e-3, float(dg.info.resolution))
        if margin_m is None:
            margin_m = self.obstacle_block_margin_m
        inflate_m = self.robot_radius + self.footprint_padding_m + max(0.0, float(margin_m))
        inflate_cells = max(1, int(math.ceil(inflate_m / res)))
        out = [row[:] for row in blocked]
        keep = set(keep_cells or [])
        marked_sources = 0

        for wx, wy in self.obstacle_points_map:
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

    def _overlay_dynamic_obstacles(self, blocked, dg, keep_cells=None):
        return self._overlay_pointcloud_obstacles(
            blocked,
            dg,
            keep_cells=keep_cells,
            enabled=self.use_pointcloud_avoidance_trigger,
            margin_m=self.obstacle_block_margin_m,
        )

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

    def _path_blocked_by_obstacles(self, path, dg, start_cell):
        if not path or not self.obstacle_points_map:
            return False

        start_idx = self._nearest_path_cell_index(path, start_cell)
        if start_idx >= len(path) - 1:
            return False

        world_path = [self._grid_to_world(dg, gx, gy) for gx, gy in path]
        corridor_half = (
            max(0.5 * self.robot_width_m, 0.5 * self.robot_length_m)
            + self.footprint_padding_m
            + self.obstacle_block_margin_m
            + self.avoidance_trigger_margin_m
        )
        corridor_half_sq = corridor_half * corridor_half
        remain_m = 0.0

        for seg_idx in range(start_idx, len(world_path) - 1):
            x0, y0 = world_path[seg_idx]
            x1, y1 = world_path[seg_idx + 1]
            seg_len = math.hypot(x1 - x0, y1 - y0)
            if seg_len <= 1e-6:
                continue
            remain_m += seg_len
            for ox, oy in self.obstacle_points_map:
                if self._point_to_segment_distance_sq(ox, oy, x0, y0, x1, y1) <= corridor_half_sq:
                    return True
            if remain_m >= self.avoidance_trigger_ahead_m:
                break
        return False

    def _first_blocked_path_index(self, path, blocked, start_cell, dg, max_check_m=None):
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

        if not self.obstacle_points_map:
            return None

        world_path = [self._grid_to_world(dg, gx, gy) for gx, gy in path]
        corridor_half = (
            max(0.5 * self.robot_width_m, 0.5 * self.robot_length_m)
            + self.footprint_padding_m
            + self.obstacle_block_margin_m
            + self.avoidance_trigger_margin_m
        )
        corridor_half_sq = corridor_half * corridor_half
        remain_m = 0.0

        for seg_idx in range(start_idx, len(world_path) - 1):
            x0, y0 = world_path[seg_idx]
            x1, y1 = world_path[seg_idx + 1]
            seg_len = math.hypot(x1 - x0, y1 - y0)
            if seg_len <= 1e-6:
                continue
            remain_m += seg_len
            for ox, oy in self.obstacle_points_map:
                if self._point_to_segment_distance_sq(ox, oy, x0, y0, x1, y1) <= corridor_half_sq:
                    return min(seg_idx + 1, len(path) - 1)
            if remain_m >= self.avoidance_trigger_ahead_m:
                break
        return None

    @staticmethod
    def _append_path_segment(out_path, segment):
        for cell in segment:
            if not out_path or out_path[-1] != cell:
                out_path.append(cell)

    def _build_branch_avoidance_path(self, nominal_path, dynamic_blocked, start_cell, dg):
        if len(nominal_path) < 2:
            return None, None

        start_idx = self._nearest_path_cell_index(nominal_path, start_cell)
        blocked_idx = self._first_blocked_path_index(
            nominal_path,
            dynamic_blocked,
            start_cell,
            dg,
            max_check_m=self.avoidance_trigger_ahead_m,
        )
        if blocked_idx is None:
            return None, None

        branch_start_idx = max(start_idx, blocked_idx - self.avoidance_branch_backtrack_cells)
        while branch_start_idx > start_idx:
            bx, by = nominal_path[branch_start_idx]
            if self._in_bounds_blocked(dynamic_blocked, bx, by) and not dynamic_blocked[by][bx]:
                break
            branch_start_idx -= 1

        branch_start = nominal_path[branch_start_idx]
        min_rejoin_cells = max(
            1, int(math.ceil(self.avoidance_rejoin_min_distance_m / max(1e-3, float(dg.info.resolution))))
        )
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
            )
            if detour is None or len(detour) < 2:
                continue

            detour = self._simplify_grid_path(
                detour,
                dynamic_blocked,
                float(dg.info.resolution),
                force=self.smooth_avoidance_line_of_sight,
            )
            if len(detour) < 2:
                continue

            composed = []
            self._append_path_segment(composed, [start_cell])
            self._append_path_segment(composed, nominal_path[start_idx:branch_start_idx + 1])
            self._append_path_segment(composed, detour[1:])
            self._append_path_segment(composed, nominal_path[rejoin_idx + 1:])

            branch_history_points = self._sample_world_points(
                [self._grid_to_world(dg, gx, gy) for gx, gy in detour]
            )
            return composed, branch_history_points

        return None, None

    def _update_avoidance_path(self, nominal_path, base_blocked, start_cell, goal_cell, dg, stamp, label):
        frame_id = dg.header.frame_id if dg.header.frame_id else "map"
        if not self.enable_avoidance_path or len(nominal_path) < 2:
            self._clear_avoidance_path(frame_id, stamp)
            return

        dynamic_blocked, obstacle_count = self._overlay_dynamic_obstacles(
            base_blocked,
            dg,
            keep_cells=(start_cell, goal_cell),
        )
        predicted_overlap = self._path_blocked_ahead(
            nominal_path,
            dynamic_blocked,
            start_cell,
            float(dg.info.resolution),
            max_check_m=self.avoidance_trigger_ahead_m,
        )
        pointcloud_overlap = False
        if (not predicted_overlap) and self.use_pointcloud_avoidance_trigger:
            pointcloud_overlap = self._path_blocked_by_obstacles(nominal_path, dg, start_cell)

        raw_point_count = len(self.obstacle_points_map)
        self._debug_avoidance_log(
            "constrained_local_replanner: avoid_eval | base={} risk_grid={} predicted_overlap={} pointcloud_enabled={} pointcloud_overlap={} raw_points={} overlay_points={} ahead={:.1f}m".format(
                label,
                "on" if self.risk_grid is not None else "off",
                "yes" if predicted_overlap else "no",
                "on" if self.use_pointcloud_avoidance_trigger else "off",
                "yes" if pointcloud_overlap else "no",
                raw_point_count,
                obstacle_count,
                self.avoidance_trigger_ahead_m,
            )
        )

        trigger_reason = None
        if predicted_overlap:
            trigger_reason = "predicted_overlap"
        elif pointcloud_overlap:
            trigger_reason = "pointcloud_overlap"

        if trigger_reason is None:
            self._clear_avoidance_path(frame_id, stamp)
            return

        avoid_path, branch_history_points = self._build_branch_avoidance_path(
            nominal_path,
            dynamic_blocked,
            start_cell,
            dg,
        )
        if avoid_path is None:
            rospy.logwarn_throttle(
                1.0,
                "constrained_local_replanner: obstacle detected on %s path (%s) but no branch-rejoin avoidance found",
                label,
                trigger_reason,
            )
            if self.avoidance_active and self._republish_last_avoidance_path(dg, stamp):
                return
            self._clear_avoidance_path(frame_id, stamp)
            return

        if len(avoid_path) < 2:
            if self.avoidance_active and self._republish_last_avoidance_path(dg, stamp):
                return
            self._clear_avoidance_path(frame_id, stamp)
            return

        self._publish_avoidance_path(
            avoid_path,
            dg,
            stamp,
            history_points=branch_history_points,
        )
        self.avoidance_clear_count = 0
        self.last_avoidance_publish_sec = stamp.to_sec()
        self.last_avoidance_solution_sec = stamp.to_sec()
        if not self.avoidance_active:
            rospy.loginfo(
                "constrained_local_replanner: avoidance path active | base=%s reason=%s obstacle_points=%d cells=%d",
                label,
                trigger_reason,
                raw_point_count,
                len(avoid_path),
            )
        self.avoidance_active = True

    def _plan_direct_goal(self, dg, rg, stamp):
        if self.direct_goal is None:
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
            return True

        sx, sy = self._world_to_grid(dg, start_xy[0], start_xy[1])
        gx, gy = self._world_to_grid(dg, goal_xy[0], goal_xy[1])

        blocked = self._inflate_blocked(dg, rg)
        start_cell = self._nearest_free_cell(blocked, (sx, sy))
        goal_cell = self._resolve_direct_goal_cell(blocked, (gx, gy))
        if start_cell is None or goal_cell is None:
            rospy.logwarn_throttle(
                1.0,
                "constrained_local_replanner: no free snapped cell for direct goal (start=%s goal=%s)",
                str((sx, sy)),
                str((gx, gy)),
            )
            self._clear_avoidance_path(dg.header.frame_id, stamp)
            return True

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
            self._update_avoidance_path(
                self.frozen_direct_grid_path,
                blocked,
                start_cell,
                frozen_goal_cell,
                dg,
                stamp,
                "direct(frozen)",
            )
            return True

        path = self._astar(
            blocked,
            start_cell,
            goal_cell,
            allow_best_effort=self.allow_best_effort_path,
        )
        if path is None:
            rospy.logwarn_throttle(
                1.0,
                "constrained_local_replanner: no direct-goal path (start=%s goal=%s snapped_start=%s snapped_goal=%s)",
                str((sx, sy)),
                str((gx, gy)),
                str(start_cell),
                str(goal_cell),
            )
            self._clear_avoidance_path(dg.header.frame_id, stamp)
            return True
        # If the direct goal is visible on the blocked grid, prefer a single
        # straight segment over the staircase-like A* cell path.
        if path[-1] == goal_cell and self._has_line_of_sight(blocked, start_cell, goal_cell):
            path = [start_cell, goal_cell]
        else:
            path = self._simplify_grid_path(path, blocked, float(dg.info.resolution))
        if not self._best_effort_path_is_acceptable(goal_cell, path, dg, "direct"):
            self._publish_empty_path(self.pub_local_path, dg.header.frame_id, stamp)
            self._clear_avoidance_path(dg.header.frame_id, stamp)
            return True

        if not self._should_publish_path(goal_cell, path):
            self._update_avoidance_path(path, blocked, start_cell, goal_cell, dg, stamp, "direct")
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
        self._update_avoidance_path(path, blocked, start_cell, goal_cell, dg, stamp, "direct")
        self._record_published_path(goal_cell, path)
        return True

    def on_timer(self, _evt):
        try:
            if (not self.have_odom) or self.drivable_grid is None:
                return
            dg = self.drivable_grid
            rg = self.risk_grid
            stamp = rospy.Time.now()

            if self.use_direct_goal and self._plan_direct_goal(dg, rg, stamp):
                return

            if self.global_path is None or len(self.global_path.poses) < 2:
                self._clear_avoidance_path(dg.header.frame_id, stamp)
                return
            pts = self._path_points(self.global_path)
            i0 = self._nearest_idx(pts, self.odom_x, self.odom_y)
            ig = self._accum_distance(pts, i0, self.lookahead_m)
            start_xy = (self.odom_x, self.odom_y)
            goal_xy = pts[ig]

            sx, sy = self._world_to_grid(dg, start_xy[0], start_xy[1])
            gx, gy = self._world_to_grid(dg, goal_xy[0], goal_xy[1])

            blocked = self._inflate_blocked(dg, rg)
            start_cell = self._nearest_free_cell(blocked, (sx, sy))
            goal_cell = self._nearest_free_cell(blocked, (gx, gy))
            if start_cell is None or goal_cell is None:
                rospy.logwarn_throttle(
                    1.0,
                    "constrained_local_replanner: no local path snap cell (start=%s goal=%s)",
                    str((sx, sy)),
                    str((gx, gy)),
                )
                self._clear_avoidance_path(dg.header.frame_id, stamp)
                return
            path = self._astar(
                blocked,
                start_cell,
                goal_cell,
                allow_best_effort=self.allow_best_effort_path,
            )
            if path is None:
                if self._publish_nominal_local_segment(pts, i0, ig, blocked, start_cell, dg, stamp):
                    self._clear_avoidance_path(dg.header.frame_id, stamp)
                    return
                rospy.logwarn_throttle(
                    1.0,
                    "constrained_local_replanner: no local path (start=%s goal=%s snapped_start=%s snapped_goal=%s)",
                    str((sx, sy)),
                    str((gx, gy)),
                    str(start_cell),
                    str(goal_cell),
                )
                self._clear_avoidance_path(dg.header.frame_id, stamp)
                return
            path = self._simplify_grid_path(path, blocked, float(dg.info.resolution))
            if not self._best_effort_path_is_acceptable(goal_cell, path, dg, "local"):
                if self._publish_nominal_local_segment(pts, i0, ig, blocked, start_cell, dg, stamp):
                    self._clear_avoidance_path(dg.header.frame_id, stamp)
                    return
                self._publish_empty_path(self.pub_local_path, dg.header.frame_id, stamp)
                self._clear_avoidance_path(dg.header.frame_id, stamp)
                return
            if not self._should_publish_path(goal_cell, path):
                self._update_avoidance_path(path, blocked, start_cell, goal_cell, dg, stamp, "local")
                return

            if path[-1] != goal_cell:
                rospy.logwarn_throttle(
                    1.0,
                    "constrained_local_replanner: best-effort local path only (snapped_goal=%s reached=%s)",
                    str(goal_cell),
                    str(path[-1]),
                )
            self._publish_local_path(
                path,
                dg,
                stamp,
                start_xy=start_xy,
                end_xy=goal_xy if path[-1] == goal_cell else None,
            )
            self._update_avoidance_path(path, blocked, start_cell, goal_cell, dg, stamp, "local")
            self._record_published_path(goal_cell, path)
        except Exception as e:
            rospy.logwarn_throttle(1.0, "constrained_local_replanner error: %s", str(e))


def main():
    rospy.init_node("constrained_local_replanner", anonymous=False)
    ConstrainedLocalReplanner()
    rospy.spin()


if __name__ == "__main__":
    main()
