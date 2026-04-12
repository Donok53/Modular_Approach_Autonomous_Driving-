#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rospy
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2


class GlobalObstacleOverlayPublisher:
    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic", "/lio_localizer/odometry/planning")
        self.global_path_topic = rospy.get_param("~global_path_topic", "/astar/path")
        self.drivable_grid_topic = rospy.get_param("~drivable_grid_topic", "/lio_sam/drivable_area/grid")
        self.obstacle_pointcloud_topic = rospy.get_param(
            "~obstacle_pointcloud_topic", "/move_base/filtered_obstacles"
        )
        self.global_obstacle_overlay_topic = str(
            rospy.get_param("~global_obstacle_overlay_topic", "/planning/global_obstacle_overlay")
        ).strip()

        self.robot_width_m = max(0.05, float(rospy.get_param("~robot_width_m", 0.58)))
        self.robot_length_m = max(0.05, float(rospy.get_param("~robot_length_m", 0.612)))
        self.footprint_padding_m = max(0.0, float(rospy.get_param("~footprint_padding_m", 0.0)))
        self.self_filter_radius_x = max(
            0.0, float(rospy.get_param("~self_filter_radius_x", 0.5 * self.robot_length_m))
        )
        self.self_filter_radius_y = max(
            0.0, float(rospy.get_param("~self_filter_radius_y", 0.5 * self.robot_width_m))
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

        self.have_odom = False
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.global_path = None
        self.drivable_grid = None
        self.global_obstacle_overlay_memory = []
        self.global_obstacle_overlay_points_map = []

        self.pub_global_obstacle_overlay = rospy.Publisher(
            self.global_obstacle_overlay_topic, OccupancyGrid, queue_size=1
        )
        self.sub_odom = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=20)
        self.sub_global = rospy.Subscriber(
            self.global_path_topic, Path, self.global_path_callback, queue_size=5
        )
        self.sub_drivable = rospy.Subscriber(
            self.drivable_grid_topic, OccupancyGrid, self.drivable_grid_callback, queue_size=3
        )
        self.sub_cloud = rospy.Subscriber(
            self.obstacle_pointcloud_topic,
            PointCloud2,
            self.cloud_callback,
            queue_size=1,
        )

        rospy.loginfo(
            "global_obstacle_overlay started | cloud=%s global=%s grid=%s out=%s persist=%d ttl=%.1fs blind_ttl=%.1fs blind_radius=%.2fm range=%.1fm lookahead=%.1fm corridor_margin=%.2fm",
            self.obstacle_pointcloud_topic,
            self.global_path_topic,
            self.drivable_grid_topic,
            self.global_obstacle_overlay_topic,
            self.global_pointcloud_overlay_persistence_frames,
            self.global_pointcloud_overlay_ttl_s,
            self.global_pointcloud_overlay_blind_zone_hold_ttl_s,
            self.global_pointcloud_overlay_blind_zone_radius_m,
            self.global_pointcloud_overlay_max_range_m,
            self.global_pointcloud_overlay_lookahead_m,
            self.global_pointcloud_overlay_corridor_margin_m,
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

    def global_path_callback(self, msg):
        self.global_path = msg

    def drivable_grid_callback(self, msg):
        self.drivable_grid = msg

    def cloud_callback(self, msg):
        if not self.have_odom:
            return

        cluster_counts = {}
        cluster_sums = {}
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

                cell = self._pointcloud_cluster_cell(x, y)
                cluster_counts[cell] = cluster_counts.get(cell, 0) + 1
                sx, sy = cluster_sums.get(cell, (0.0, 0.0))
                cluster_sums[cell] = (sx + x, sy + y)

            current_points_map = []
            for (cx, cy), count in cluster_counts.items():
                support = 0
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        support += cluster_counts.get((cx + dx, cy + dy), 0)
                if support < self.pointcloud_min_cluster_points:
                    continue
                sx, sy = cluster_sums[(cx, cy)]
                current_points_map.append(self._local_to_map(sx / float(count), sy / float(count)))

            stamp_sec = msg.header.stamp.to_sec()
            if stamp_sec <= 0.0:
                stamp_sec = rospy.Time.now().to_sec()

            candidates = self._select_global_overlay_candidate_points(current_points_map)
            self._update_global_obstacle_overlay_memory(candidates, stamp_sec)
            self._publish_global_obstacle_overlay(msg.header.stamp)
        except Exception as exc:
            rospy.logwarn_throttle(1.0, "global_obstacle_overlay cloud error: %s", str(exc))

    def _local_to_map(self, x, y):
        c = math.cos(self.odom_yaw)
        s = math.sin(self.odom_yaw)
        mx = self.odom_x + c * x - s * y
        my = self.odom_y + s * x + c * y
        return mx, my

    def _pointcloud_cluster_cell(self, x, y):
        res = max(1e-3, self.pointcloud_cluster_resolution_m)
        return (int(math.floor(x / res)), int(math.floor(y / res)))

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
            "global_obstacle_overlay: expanding single-pose global path to odom->goal fallback (goal=%.2f, %.2f)",
            goal_x,
            goal_y,
        )
        return [(self.odom_x, self.odom_y), (goal_x, goal_y)]

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

    def _pointcloud_corridor_half_width_m(self, margin_m):
        return max(
            0.05,
            0.5 * self.robot_width_m + self.footprint_padding_m + max(0.0, float(margin_m)),
        )

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
        if not current_points_map:
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
        corridor_half_sq = self._pointcloud_corridor_half_width_m(
            self.global_pointcloud_overlay_corridor_margin_m
        ) ** 2

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
        if self.global_pointcloud_overlay_ttl_s <= 0.0 or self.global_pointcloud_overlay_max_points <= 0:
            self.global_obstacle_overlay_memory = []
            self.global_obstacle_overlay_points_map = []
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
        return confirmed

    def _update_global_obstacle_overlay_memory(self, candidates_map, now_sec):
        confirmed = self._prune_global_obstacle_overlay_memory(now_sec)
        if not candidates_map:
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
                    min(hits + 1, self.global_pointcloud_overlay_persistence_frames + 8),
                )

        self.global_obstacle_overlay_memory = memory
        return self._prune_global_obstacle_overlay_memory(now_sec)

    def _publish_global_obstacle_overlay(self, stamp):
        if self.drivable_grid is None:
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

    def _world_to_grid(self, g, x, y):
        res = float(g.info.resolution)
        gx = int(math.floor((x - float(g.info.origin.position.x)) / res))
        gy = int(math.floor((y - float(g.info.origin.position.y)) / res))
        return gx, gy

    @staticmethod
    def _in_bounds(g, gx, gy):
        return 0 <= gx < int(g.info.width) and 0 <= gy < int(g.info.height)


if __name__ == "__main__":
    rospy.init_node("global_obstacle_overlay", anonymous=False)
    GlobalObstacleOverlayPublisher()
    rospy.spin()
