#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import heapq
import math

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path


class ConstrainedLocalReplanner:
    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic", "/lio_localizer/odometry/optimization")
        self.global_path_topic = rospy.get_param("~global_path_topic", "/astar/path")
        self.drivable_grid_topic = rospy.get_param("~drivable_grid_topic", "/lio_sam/drivable_area/grid")
        self.dynamic_risk_grid_topic = rospy.get_param("~dynamic_risk_grid_topic", "/planning/dynamic_risk_grid")
        self.local_path_topic = rospy.get_param("~local_path_topic", "/planning/local_path")
        self.use_direct_goal = bool(rospy.get_param("~use_direct_goal", False))
        self.direct_goal_topic = rospy.get_param("~direct_goal_topic", "/move_base_simple/goal")
        self.goal_tolerance_m = max(0.05, float(rospy.get_param("~goal_tolerance_m", 0.35)))
        self.snap_search_radius_cells = max(1, int(rospy.get_param("~snap_search_radius_cells", 30)))
        self.allow_best_effort_path = bool(rospy.get_param("~allow_best_effort_path", True))
        self.best_effort_improve_margin_cells = max(
            0.0, float(rospy.get_param("~best_effort_improve_margin_cells", 2.0))
        )
        self.best_effort_update_period_s = max(
            0.1, float(rospy.get_param("~best_effort_update_period_s", 1.5))
        )

        self.lookahead_m = max(2.0, float(rospy.get_param("~lookahead_m", 10.0)))
        self.window_margin_m = max(1.0, float(rospy.get_param("~window_margin_m", 12.0)))
        self.robot_radius_m = max(0.1, float(rospy.get_param("~robot_radius_m", 0.45)))
        self.risk_threshold = int(rospy.get_param("~risk_occupied_threshold", 45))
        self.max_expand = max(100, int(rospy.get_param("~max_expand", 25000)))
        self.replan_hz = max(1.0, float(rospy.get_param("~replan_hz", 6.0)))
        self.simplify_stride = max(1, int(rospy.get_param("~simplify_stride", 2)))

        self.have_odom = False
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.global_path = None
        self.drivable_grid = None
        self.risk_grid = None
        self.direct_goal = None
        self.last_published_goal_cell = None
        self.last_published_end_cell = None
        self.last_path_publish_sec = 0.0

        self.pub_local_path = rospy.Publisher(self.local_path_topic, Path, queue_size=2)
        self.sub_odom = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=20)
        self.sub_global = rospy.Subscriber(self.global_path_topic, Path, self.global_path_callback, queue_size=5)
        self.sub_drivable = rospy.Subscriber(self.drivable_grid_topic, OccupancyGrid, self.drivable_grid_callback, queue_size=3)
        self.sub_risk = rospy.Subscriber(self.dynamic_risk_grid_topic, OccupancyGrid, self.risk_grid_callback, queue_size=3)
        self.sub_direct_goal = None
        if self.use_direct_goal:
            self.sub_direct_goal = rospy.Subscriber(self.direct_goal_topic, PoseStamped, self.direct_goal_callback, queue_size=2)

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.replan_hz), self.on_timer)
        rospy.loginfo(
            "constrained_local_replanner started | global=%s drivable=%s risk=%s local=%s direct_goal=%s(%s)",
            self.global_path_topic,
            self.drivable_grid_topic,
            self.dynamic_risk_grid_topic,
            self.local_path_topic,
            "on" if self.use_direct_goal else "off",
            self.direct_goal_topic,
        )

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        self.odom_x = float(p.x)
        self.odom_y = float(p.y)
        self.have_odom = True

    def global_path_callback(self, msg):
        self.global_path = msg

    def drivable_grid_callback(self, msg):
        self.drivable_grid = msg

    def risk_grid_callback(self, msg):
        self.risk_grid = msg

    def direct_goal_callback(self, msg):
        self.direct_goal = msg
        self.last_published_goal_cell = None
        self.last_published_end_cell = None
        self.last_path_publish_sec = 0.0
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
        inflate = max(1, int(math.ceil(self.robot_radius_m / max(1e-3, res))))
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
        rr = inflate * inflate
        for y in range(h):
            for x in range(w):
                if not base[y][x]:
                    continue
                for dx in range(-inflate, inflate + 1):
                    for dy in range(-inflate, inflate + 1):
                        if dx * dx + dy * dy > rr:
                            continue
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            out[ny][nx] = True
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

    def _publish_local_path(self, grid_path, dg, stamp):
        out = Path()
        out.header.stamp = stamp
        out.header.frame_id = dg.header.frame_id if dg.header.frame_id else "map"
        for i, (gx, gy) in enumerate(grid_path):
            if self.simplify_stride > 1 and i not in (0, len(grid_path) - 1) and (i % self.simplify_stride != 0):
                continue
            x, y = self._grid_to_world(dg, gx, gy)
            ps = PoseStamped()
            ps.header = out.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            out.poses.append(ps)
        if len(out.poses) >= 2:
            self.pub_local_path.publish(out)

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

    def _plan_direct_goal(self, dg, rg, stamp):
        if self.direct_goal is None:
            return False

        start_xy = (self.odom_x, self.odom_y)
        goal_xy = (
            float(self.direct_goal.pose.position.x),
            float(self.direct_goal.pose.position.y),
        )
        dist_to_goal = math.hypot(goal_xy[0] - start_xy[0], goal_xy[1] - start_xy[1])
        if dist_to_goal <= self.goal_tolerance_m:
            self._publish_world_path([start_xy, goal_xy], dg.header.frame_id, stamp)
            return True

        sx, sy = self._world_to_grid(dg, start_xy[0], start_xy[1])
        gx, gy = self._world_to_grid(dg, goal_xy[0], goal_xy[1])

        blocked = self._inflate_blocked(dg, rg)
        start_cell = self._nearest_free_cell(blocked, (sx, sy))
        goal_cell = self._nearest_free_cell(blocked, (gx, gy))
        if start_cell is None or goal_cell is None:
            rospy.logwarn_throttle(
                1.0,
                "constrained_local_replanner: no free snapped cell for direct goal (start=%s goal=%s)",
                str((sx, sy)),
                str((gx, gy)),
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
            return True

        if not self._should_publish_path(goal_cell, path):
            return True

        if path[-1] != goal_cell:
            rospy.logwarn_throttle(
                1.0,
                "constrained_local_replanner: best-effort direct path only (snapped_goal=%s reached=%s)",
                str(goal_cell),
                str(path[-1]),
            )
        self._publish_local_path(path, dg, stamp)
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
                return
            path = self._astar(
                blocked,
                start_cell,
                goal_cell,
                allow_best_effort=self.allow_best_effort_path,
            )
            if path is None:
                rospy.logwarn_throttle(
                    1.0,
                    "constrained_local_replanner: no local path (start=%s goal=%s snapped_start=%s snapped_goal=%s)",
                    str((sx, sy)),
                    str((gx, gy)),
                    str(start_cell),
                    str(goal_cell),
                )
                return
            if not self._should_publish_path(goal_cell, path):
                return

            if path[-1] != goal_cell:
                rospy.logwarn_throttle(
                    1.0,
                    "constrained_local_replanner: best-effort local path only (snapped_goal=%s reached=%s)",
                    str(goal_cell),
                    str(path[-1]),
                )
            self._publish_local_path(path, dg, stamp)
            self._record_published_path(goal_cell, path)
        except Exception as e:
            rospy.logwarn_throttle(1.0, "constrained_local_replanner error: %s", str(e))


def main():
    rospy.init_node("constrained_local_replanner", anonymous=False)
    ConstrainedLocalReplanner()
    rospy.spin()


if __name__ == "__main__":
    main()
