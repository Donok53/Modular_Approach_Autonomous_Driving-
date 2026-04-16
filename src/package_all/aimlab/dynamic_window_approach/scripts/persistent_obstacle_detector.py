#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rospy
import tf.transformations as transformations
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray

from dynamic_window_approach.msg import TrackedObject, TrackedObjectArray


class PersistentObstacleDetector:
    def __init__(self):
        self.pointcloud_topic = rospy.get_param("~pointcloud_topic", "/ouster/points")
        self.obstacle_pointcloud_topic = str(
            rospy.get_param("~obstacle_pointcloud_topic", "")
        ).strip()
        self.odom_topic = rospy.get_param(
            "~odom_topic", "/lio_localizer/odometry/optimization"
        )
        self.output_topic = rospy.get_param(
            "~output_topic", "/perception/tracked_objects"
        )
        self.drivable_grid_topic = rospy.get_param(
            "~drivable_grid_topic", "/lio_sam/drivable_area/grid"
        )
        self.persistence_grid_topic = rospy.get_param(
            "~persistence_grid_topic", "/perception/persistent_obstacle_grid"
        )
        self.detector_markers_topic = rospy.get_param(
            "~detector_markers_topic", "/perception/persistent_obstacle_detector_markers"
        )

        self.min_z = float(rospy.get_param("~min_z", -0.4))
        self.max_z = float(rospy.get_param("~max_z", 2.2))
        self.max_range_m = float(rospy.get_param("~max_range_m", 25.0))
        self.downsample = max(1, int(rospy.get_param("~downsample", 2)))
        self.cell_size_m = max(0.1, float(rospy.get_param("~cell_size_m", 0.25)))
        self.min_points_per_cell = max(
            1, int(rospy.get_param("~min_points_per_cell", 2))
        )
        self.far_field_min_points_per_cell = max(
            1,
            int(
                rospy.get_param(
                    "~far_field_min_points_per_cell",
                    min(1, self.min_points_per_cell),
                )
            ),
        )
        self.min_cluster_cells = max(
            1, int(rospy.get_param("~min_cluster_cells", 3))
        )
        self.far_field_min_cluster_cells = max(
            1,
            int(
                rospy.get_param(
                    "~far_field_min_cluster_cells",
                    min(1, self.min_cluster_cells),
                )
            ),
        )

        self.max_assoc_dist_m = max(
            0.1, float(rospy.get_param("~max_assoc_dist_m", 1.5))
        )
        self.dynamic_assoc_bonus_m = max(
            0.0, float(rospy.get_param("~dynamic_assoc_bonus_m", 0.5))
        )
        self.track_timeout_s = max(
            0.1, float(rospy.get_param("~track_timeout_s", 2.0))
        )
        self.vel_alpha = min(
            1.0, max(0.01, float(rospy.get_param("~vel_alpha", 0.25)))
        )
        self.publish_static = bool(rospy.get_param("~publish_static", False))
        self.publish_static_persons = bool(
            rospy.get_param("~publish_static_persons", True)
        )
        self.publish_static_large_obstacles = bool(
            rospy.get_param("~publish_static_large_obstacles", True)
        )
        self.static_speed_thresh_mps = max(
            0.01, float(rospy.get_param("~static_speed_thresh_mps", 0.12))
        )
        self.dynamic_min_age = max(
            1, int(rospy.get_param("~dynamic_min_age", 3))
        )
        self.pedestrian_static_speed_thresh_mps = max(
            0.01,
            min(
                self.static_speed_thresh_mps,
                float(
                    rospy.get_param(
                        "~pedestrian_static_speed_thresh_mps", 0.06
                    )
                ),
            ),
        )
        self.pedestrian_dynamic_min_age = max(
            1,
            min(
                self.dynamic_min_age,
                int(rospy.get_param("~pedestrian_dynamic_min_age", 2)),
            ),
        )
        self.position_jitter_m = max(
            0.0, float(rospy.get_param("~position_jitter_m", 0.16))
        )
        self.dynamic_min_displacement_m = max(
            self.position_jitter_m,
            float(
                rospy.get_param(
                    "~dynamic_min_displacement_m",
                    max(0.30, self.position_jitter_m * 1.8),
                )
            ),
        )
        self.pedestrian_dynamic_min_displacement_m = max(
            self.position_jitter_m,
            min(
                self.dynamic_min_displacement_m,
                float(
                    rospy.get_param(
                        "~pedestrian_dynamic_min_displacement_m",
                        max(0.14, self.position_jitter_m * 0.9),
                    )
                ),
            ),
        )
        self.recent_dynamic_hold_s = max(
            0.0, float(rospy.get_param("~recent_dynamic_hold_s", 2.5))
        )
        self.recent_dynamic_velocity_decay = min(
            1.0,
            max(
                0.0,
                float(rospy.get_param("~recent_dynamic_velocity_decay", 0.65)),
            ),
        )

        self.known_map_subtraction_enabled = bool(
            rospy.get_param("~known_map_subtraction_enabled", True)
        )
        self.known_map_subtraction_radius_m = max(
            0.0, float(rospy.get_param("~known_map_subtraction_radius_m", 0.30))
        )
        self.allow_out_of_grid_clusters = bool(
            rospy.get_param("~allow_out_of_grid_clusters", True)
        )
        self.grid_relaxation_distance_m = max(
            0.0, float(rospy.get_param("~grid_relaxation_distance_m", 10.0))
        )
        self.free_space_support_radius_m = max(
            0.0, float(rospy.get_param("~free_space_support_radius_m", 0.35))
        )
        self.free_space_support_min_cells = max(
            1, int(rospy.get_param("~free_space_support_min_cells", 3))
        )
        self.far_field_free_space_support_min_cells = max(
            1, int(rospy.get_param("~far_field_free_space_support_min_cells", 1))
        )

        self.persistence_hit_value = max(
            1.0, float(rospy.get_param("~persistence_hit_value", 28.0))
        )
        self.persistence_point_gain = max(
            0.0, float(rospy.get_param("~persistence_point_gain", 2.0))
        )
        self.persistence_decay_per_s = max(
            0.1, float(rospy.get_param("~persistence_decay_per_s", 8.0))
        )
        self.persistence_confirm_threshold = max(
            1.0, float(rospy.get_param("~persistence_confirm_threshold", 40.0))
        )
        self.persistence_publish_threshold = max(
            1.0,
            min(
                self.persistence_confirm_threshold,
                float(rospy.get_param("~persistence_publish_threshold", 12.0)),
            ),
        )
        self.persistence_delete_threshold = max(
            0.0, float(rospy.get_param("~persistence_delete_threshold", 3.0))
        )
        self.persistence_max_value = max(
            self.persistence_confirm_threshold,
            float(rospy.get_param("~persistence_max_value", 100.0)),
        )

        self.marker_lifetime_s = max(
            0.0, float(rospy.get_param("~marker_lifetime_s", 0.6))
        )
        self.show_labels = bool(rospy.get_param("~show_labels", True))
        self.show_velocity = bool(rospy.get_param("~show_velocity", True))
        self.show_provisional_cells = bool(
            rospy.get_param("~show_provisional_cells", True)
        )
        self.z_offset_m = float(rospy.get_param("~z_offset_m", 0.25))
        self.text_height_m = max(
            0.05, float(rospy.get_param("~text_height_m", 0.28))
        )

        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.have_odom = False
        self.drivable_grid = None

        self.next_track_id = 1
        self.tracks = {}
        self.persistent_cells = {}
        self.last_processed_stamp_sec = -1.0

        self.last_clusters = []
        self.last_confirmed_cells = []
        self.last_provisional_cells = []

        self.pub_tracks = rospy.Publisher(
            self.output_topic, TrackedObjectArray, queue_size=2
        )
        self.pub_grid = rospy.Publisher(
            self.persistence_grid_topic, OccupancyGrid, queue_size=1
        )
        self.pub_markers = rospy.Publisher(
            self.detector_markers_topic, MarkerArray, queue_size=1
        )

        self.sub_odom = rospy.Subscriber(
            self.odom_topic, Odometry, self.odom_callback, queue_size=20
        )
        self.sub_drivable = rospy.Subscriber(
            self.drivable_grid_topic,
            OccupancyGrid,
            self.drivable_grid_callback,
            queue_size=3,
        )
        self.sub_cloud = rospy.Subscriber(
            self.pointcloud_topic, PointCloud2, self.cloud_callback, queue_size=1
        )
        self.sub_obstacle_cloud = None
        if (
            self.obstacle_pointcloud_topic
            and self.obstacle_pointcloud_topic != self.pointcloud_topic
        ):
            self.sub_obstacle_cloud = rospy.Subscriber(
                self.obstacle_pointcloud_topic,
                PointCloud2,
                self.cloud_callback,
                queue_size=1,
            )

        rospy.loginfo(
            "persistent_obstacle_detector started | cloud=%s obstacle_cloud=%s odom=%s grid=%s out=%s persist_grid=%s markers=%s range=%.1fm cell=%.2fm min_pts=%d far_pts=%d min_cells=%d far_cells=%d confirm=%.1f decay=%.1f/s",
            self.pointcloud_topic,
            self.obstacle_pointcloud_topic if self.obstacle_pointcloud_topic else "-",
            self.odom_topic,
            self.drivable_grid_topic,
            self.output_topic,
            self.persistence_grid_topic,
            self.detector_markers_topic,
            self.max_range_m,
            self.cell_size_m,
            self.min_points_per_cell,
            self.far_field_min_points_per_cell,
            self.min_cluster_cells,
            self.far_field_min_cluster_cells,
            self.persistence_confirm_threshold,
            self.persistence_decay_per_s,
        )

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.odom_x = float(p.x)
        self.odom_y = float(p.y)
        self.odom_yaw = transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w]
        )[2]
        self.have_odom = True

    def drivable_grid_callback(self, msg):
        self.drivable_grid = msg

    def _local_to_map(self, x, y):
        c = math.cos(self.odom_yaw)
        s = math.sin(self.odom_yaw)
        mx = self.odom_x + c * x - s * y
        my = self.odom_y + s * x + c * y
        return mx, my

    def _map_distance_from_robot(self, x, y):
        return math.hypot(float(x) - self.odom_x, float(y) - self.odom_y)

    @staticmethod
    def _world_to_grid(g, x, y):
        res = float(g.info.resolution)
        gx = int(math.floor((float(x) - float(g.info.origin.position.x)) / res))
        gy = int(math.floor((float(y) - float(g.info.origin.position.y)) / res))
        return gx, gy

    @staticmethod
    def _in_bounds(g, gx, gy):
        return 0 <= gx < int(g.info.width) and 0 <= gy < int(g.info.height)

    def _cell_center(self, ix, iy):
        return ((ix + 0.5) * self.cell_size_m, (iy + 0.5) * self.cell_size_m)

    @staticmethod
    def _grid_cell_is_drivable_free(g, gx, gy):
        if not (0 <= gx < int(g.info.width) and 0 <= gy < int(g.info.height)):
            return False
        idx = gy * int(g.info.width) + gx
        return int(g.data[idx]) == 0

    def _cluster_overlaps_known_map_obstacle(self, cluster):
        if (not self.known_map_subtraction_enabled) or self.drivable_grid is None:
            return False

        g = self.drivable_grid
        margin_m = max(0.0, self.known_map_subtraction_radius_m)
        min_x = float(cluster["min_x"]) - margin_m
        max_x = float(cluster["max_x"]) + margin_m
        min_y = float(cluster["min_y"]) - margin_m
        max_y = float(cluster["max_y"]) + margin_m
        gx0, gy0 = self._world_to_grid(g, min_x, min_y)
        gx1, gy1 = self._world_to_grid(g, max_x, max_y)
        saw_in_bounds = False
        for gy in range(min(gy0, gy1), max(gy0, gy1) + 1):
            for gx in range(min(gx0, gx1), max(gx0, gx1) + 1):
                if not self._in_bounds(g, gx, gy):
                    continue
                saw_in_bounds = True
                if self._grid_cell_is_drivable_free(g, gx, gy):
                    continue
                return True
        if saw_in_bounds:
            return False
        if not self.allow_out_of_grid_clusters:
            return True
        return self._map_distance_from_robot(
            cluster["x"], cluster["y"]
        ) < self.grid_relaxation_distance_m

    def _pose_has_free_space_support(self, x, y):
        if (not self.known_map_subtraction_enabled) or self.drivable_grid is None:
            return True

        g = self.drivable_grid
        gx, gy = self._world_to_grid(g, x, y)
        range_m = self._map_distance_from_robot(x, y)
        far_field_relaxed = (
            self.allow_out_of_grid_clusters
            and range_m >= self.grid_relaxation_distance_m
        )
        if not self._in_bounds(g, gx, gy):
            return far_field_relaxed

        required_free_cells = self.free_space_support_min_cells
        if far_field_relaxed:
            required_free_cells = min(
                required_free_cells, self.far_field_free_space_support_min_cells
            )
        required_free_cells = max(1, required_free_cells)
        radius_cells = int(
            math.ceil(self.free_space_support_radius_m / float(g.info.resolution))
        )
        free_cells = 0
        for ny in range(gy - radius_cells, gy + radius_cells + 1):
            for nx in range(gx - radius_cells, gx + radius_cells + 1):
                if not self._in_bounds(g, nx, ny):
                    continue
                if self._grid_cell_is_drivable_free(g, nx, ny):
                    free_cells += 1
                    if free_cells >= required_free_cells:
                        return True
        return far_field_relaxed and free_cells > 0

    def _extract_observed_cells(self, msg):
        if not self.have_odom:
            return {}

        rr = self.max_range_m * self.max_range_m
        cells = {}
        i = 0
        for p in point_cloud2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True
        ):
            i += 1
            if self.downsample > 1 and (i % self.downsample != 0):
                continue
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            if z < self.min_z or z > self.max_z:
                continue
            if x * x + y * y > rr:
                continue
            mx, my = self._local_to_map(x, y)
            ix = int(math.floor(mx / self.cell_size_m))
            iy = int(math.floor(my / self.cell_size_m))
            cell = cells.setdefault((ix, iy), {"points": 0, "x": 0.0, "y": 0.0})
            cell["points"] += 1
            cell["x"] += mx
            cell["y"] += my

        observed = {}
        for (ix, iy), data in cells.items():
            cx, cy = self._cell_center(ix, iy)
            range_m = self._map_distance_from_robot(cx, cy)
            min_points = self.min_points_per_cell
            if range_m >= self.grid_relaxation_distance_m:
                min_points = min(min_points, self.far_field_min_points_per_cell)
            min_points = max(1, min_points)
            if int(data["points"]) < min_points:
                continue
            observed[(ix, iy)] = {
                "count": int(data["points"]),
                "x": float(data["x"]) / max(1, int(data["points"])),
                "y": float(data["y"]) / max(1, int(data["points"])),
            }
        return observed

    def _decay_cell(self, cell, now_sec):
        dt = max(0.0, now_sec - float(cell.get("last_t", now_sec)))
        if dt <= 0.0:
            return
        cell["score"] = max(
            0.0, float(cell.get("score", 0.0)) - self.persistence_decay_per_s * dt
        )
        cell["last_t"] = now_sec

    def _update_persistence(self, observed_cells, now_sec):
        for key in list(self.persistent_cells.keys()):
            cell = self.persistent_cells.get(key)
            if cell is None:
                continue
            self._decay_cell(cell, now_sec)
            if key in observed_cells:
                obs = observed_cells[key]
                cell["score"] = min(
                    self.persistence_max_value,
                    float(cell.get("score", 0.0))
                    + self.persistence_hit_value
                    + self.persistence_point_gain * float(obs["count"]),
                )
                cell["last_seen_t"] = now_sec
                cell["hits"] = int(cell.get("hits", 0)) + 1
                cell["point_count"] = int(obs["count"])
                cell["x"] = float(obs["x"])
                cell["y"] = float(obs["y"])
            if float(cell.get("score", 0.0)) <= self.persistence_delete_threshold:
                self.persistent_cells.pop(key, None)

        for key, obs in observed_cells.items():
            if key in self.persistent_cells:
                continue
            self.persistent_cells[key] = {
                "score": min(
                    self.persistence_max_value,
                    self.persistence_hit_value
                    + self.persistence_point_gain * float(obs["count"]),
                ),
                "hits": 1,
                "last_t": now_sec,
                "last_seen_t": now_sec,
                "point_count": int(obs["count"]),
                "x": float(obs["x"]),
                "y": float(obs["y"]),
            }

    def _extract_clusters(self):
        provisional = {
            key
            for key, cell in self.persistent_cells.items()
            if float(cell.get("score", 0.0)) >= self.persistence_publish_threshold
        }
        confirmed = {
            key
            for key, cell in self.persistent_cells.items()
            if float(cell.get("score", 0.0)) >= self.persistence_confirm_threshold
        }
        self.last_provisional_cells = list(provisional - confirmed)
        self.last_confirmed_cells = list(confirmed)

        visited = set()
        clusters = []
        for seed in provisional:
            if seed in visited:
                continue
            stack = [seed]
            visited.add(seed)
            comp = []
            comp_confirmed = False
            while stack:
                c = stack.pop()
                comp.append(c)
                if c in confirmed:
                    comp_confirmed = True
                cx, cy = c
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nc = (cx + dx, cy + dy)
                        if nc in provisional and nc not in visited:
                            visited.add(nc)
                            stack.append(nc)

            if not comp:
                continue

            min_x = 1e9
            min_y = 1e9
            max_x = -1e9
            max_y = -1e9
            weight_sum = 0.0
            weighted_x_sum = 0.0
            weighted_y_sum = 0.0
            max_score = 0.0
            max_hits = 0

            for ix, iy in comp:
                cell = self.persistent_cells.get((ix, iy))
                if cell is None:
                    continue
                cx, cy = self._cell_center(ix, iy)
                min_x = min(min_x, cx - 0.5 * self.cell_size_m)
                min_y = min(min_y, cy - 0.5 * self.cell_size_m)
                max_x = max(max_x, cx + 0.5 * self.cell_size_m)
                max_y = max(max_y, cy + 0.5 * self.cell_size_m)
                weight = max(1.0, float(cell.get("score", 0.0)))
                weight_sum += weight
                weighted_x_sum += float(cell.get("x", cx)) * weight
                weighted_y_sum += float(cell.get("y", cy)) * weight
                max_score = max(max_score, float(cell.get("score", 0.0)))
                max_hits = max(max_hits, int(cell.get("hits", 0)))

            if weight_sum <= 0.0:
                continue

            cx = weighted_x_sum / weight_sum
            cy = weighted_y_sum / weight_sum
            range_m = self._map_distance_from_robot(cx, cy)
            min_cluster_cells = self.min_cluster_cells
            if range_m >= self.grid_relaxation_distance_m:
                min_cluster_cells = min(
                    min_cluster_cells, self.far_field_min_cluster_cells
                )
            min_cluster_cells = max(1, min_cluster_cells)

            if len(comp) < min_cluster_cells and not comp_confirmed:
                continue

            cluster = {
                "x": cx,
                "y": cy,
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
                "size_x": max(0.20, max_x - min_x),
                "size_y": max(0.20, max_y - min_y),
                "score": min(1.0, max_score / max(1.0, self.persistence_max_value)),
                "max_score": max_score,
                "max_hits": max_hits,
                "cell_count": len(comp),
                "confirmed": comp_confirmed,
            }
            if self._cluster_overlaps_known_map_obstacle(cluster):
                continue
            if not self._pose_has_free_space_support(cluster["x"], cluster["y"]):
                continue
            clusters.append(cluster)

        self.last_clusters = list(clusters)
        return clusters

    def _associate_and_update(self, clusters, now_sec):
        track_ids = list(self.tracks.keys())
        unmatched_tracks = set(track_ids)
        unmatched_clusters = set(range(len(clusters)))
        matches = []

        for ci, c in enumerate(clusters):
            best_tid = None
            best_d = 1e9
            for tid in track_ids:
                if tid not in unmatched_tracks:
                    continue
                t = self.tracks[tid]
                dt = max(0.0, now_sec - float(t["last_t"]))
                pred_horizon = min(dt, self.track_timeout_s)
                pred_x = float(t["x"]) + float(t["vx"]) * pred_horizon
                pred_y = float(t["y"]) + float(t["vy"]) * pred_horizon
                assoc_limit = self.max_assoc_dist_m
                if (
                    self.recent_dynamic_hold_s > 0.0
                    and (
                        now_sec - float(t.get("last_dynamic_t", 0.0))
                    )
                    <= self.recent_dynamic_hold_s
                ):
                    assoc_limit += self.dynamic_assoc_bonus_m
                d = math.hypot(c["x"] - pred_x, c["y"] - pred_y)
                if d < best_d and d <= assoc_limit:
                    best_d = d
                    best_tid = tid
            if best_tid is not None:
                matches.append((ci, best_tid))
                unmatched_tracks.discard(best_tid)
                unmatched_clusters.discard(ci)

        for ci, tid in matches:
            c = clusters[ci]
            t = self.tracks[tid]
            dt = max(1e-3, now_sec - float(t["last_t"]))
            dx = float(c["x"]) - float(t["x"])
            dy = float(c["y"]) - float(t["y"])
            if math.hypot(dx, dy) < self.position_jitter_m:
                vx_obs = 0.0
                vy_obs = 0.0
            else:
                vx_obs = dx / dt
                vy_obs = dy / dt
            t["vx"] = (1.0 - self.vel_alpha) * float(t["vx"]) + self.vel_alpha * vx_obs
            t["vy"] = (1.0 - self.vel_alpha) * float(t["vy"]) + self.vel_alpha * vy_obs
            t["x"] = float(c["x"])
            t["y"] = float(c["y"])
            t["size_x"] = float(c["size_x"])
            t["size_y"] = float(c["size_y"])
            t["score"] = float(c["score"])
            t["confirmed"] = bool(c.get("confirmed", False))
            t["cell_count"] = int(c.get("cell_count", 0))
            t["last_t"] = now_sec
            t["age"] = int(t.get("age", 0)) + 1

        for ci in unmatched_clusters:
            c = clusters[ci]
            tid = self.next_track_id
            self.next_track_id += 1
            self.tracks[tid] = {
                "x": float(c["x"]),
                "y": float(c["y"]),
                "anchor_x": float(c["x"]),
                "anchor_y": float(c["y"]),
                "vx": 0.0,
                "vy": 0.0,
                "size_x": float(c["size_x"]),
                "size_y": float(c["size_y"]),
                "score": float(c["score"]),
                "confirmed": bool(c.get("confirmed", False)),
                "cell_count": int(c.get("cell_count", 0)),
                "last_t": now_sec,
                "age": 1,
                "last_dynamic_t": 0.0,
            }

        stale = [
            tid
            for tid, t in self.tracks.items()
            if (now_sec - float(t["last_t"])) > self.track_timeout_s
        ]
        for tid in stale:
            self.tracks.pop(tid, None)

    @staticmethod
    def _label_track(size_x, size_y, speed):
        area = size_x * size_y
        if area >= 2.5 or max(size_x, size_y) >= 1.8:
            return "vehicle" if speed > 0.2 else "static_vehicle"
        if area >= 0.5:
            return "pedestrian" if speed > 0.15 else "static_person"
        return "unknown"

    @staticmethod
    def _is_person_like_label(label):
        text = (label or "").lower()
        return ("ped" in text) or ("person" in text) or ("walker" in text)

    @staticmethod
    def _is_large_static_track(track):
        size_x = max(0.0, float(track.get("size_x", 0.0)))
        size_y = max(0.0, float(track.get("size_y", 0.0)))
        span = max(size_x, size_y)
        area = size_x * size_y
        return span >= 1.0 or area >= 0.8

    def _should_publish_static_track(self, label, track):
        if self.publish_static:
            return True
        if self._is_person_like_label(label):
            return self.publish_static_persons
        if self.publish_static_large_obstacles and self._is_large_static_track(track):
            return True
        return False

    def _dynamic_min_displacement_for_label(self, label):
        if self._is_person_like_label(label):
            return self.pedestrian_dynamic_min_displacement_m
        return self.dynamic_min_displacement_m

    def cloud_callback(self, msg):
        try:
            now = (
                msg.header.stamp
                if msg.header.stamp.to_sec() > 0.0
                else rospy.Time.now()
            )
            now_sec = now.to_sec()
            if (
                self.last_processed_stamp_sec > 0.0
                and abs(now_sec - self.last_processed_stamp_sec) < 0.04
            ):
                return
            self.last_processed_stamp_sec = now_sec
            observed_cells = self._extract_observed_cells(msg)
            self._update_persistence(observed_cells, now_sec)
            clusters = self._extract_clusters()
            self._associate_and_update(clusters, now_sec)
            self.publish_tracks(now)
            self.publish_persistence_grid(now)
            self.publish_detector_markers(now)
        except Exception as exc:
            rospy.logwarn_throttle(1.0, "persistent_obstacle_detector error: %s", str(exc))

    def publish_tracks(self, stamp):
        out = TrackedObjectArray()
        out.header.stamp = stamp if stamp.to_sec() > 0.0 else rospy.Time.now()
        out.header.frame_id = "map"
        stamp_sec = out.header.stamp.to_sec()

        for tid, t in self.tracks.items():
            speed = math.hypot(float(t["vx"]), float(t["vy"]))
            raw_label = self._label_track(t["size_x"], t["size_y"], speed)
            is_person_like = self._is_person_like_label(raw_label)
            dynamic_min_age = (
                self.pedestrian_dynamic_min_age
                if is_person_like
                else self.dynamic_min_age
            )
            static_speed_thresh = (
                self.pedestrian_static_speed_thresh_mps
                if is_person_like
                else self.static_speed_thresh_mps
            )
            min_dynamic_displacement = self._dynamic_min_displacement_for_label(raw_label)
            anchor_x = float(t.get("anchor_x", t["x"]))
            anchor_y = float(t.get("anchor_y", t["y"]))
            track_displacement = math.hypot(
                float(t["x"]) - anchor_x, float(t["y"]) - anchor_y
            )
            effective_vx = float(t["vx"])
            effective_vy = float(t["vy"])
            effective_speed = speed
            was_recent_dynamic = (
                self.recent_dynamic_hold_s > 0.0
                and (stamp_sec - float(t.get("last_dynamic_t", 0.0)))
                <= self.recent_dynamic_hold_s
            )
            observed_dynamic = True
            if int(t["age"]) < dynamic_min_age:
                observed_dynamic = False
                effective_vx = 0.0
                effective_vy = 0.0
                effective_speed = 0.0
            elif track_displacement < min_dynamic_displacement:
                observed_dynamic = False
                effective_vx = 0.0
                effective_vy = 0.0
                effective_speed = 0.0
            if effective_speed < static_speed_thresh:
                observed_dynamic = False
                if was_recent_dynamic and self.recent_dynamic_velocity_decay > 0.0:
                    effective_vx *= self.recent_dynamic_velocity_decay
                    effective_vy *= self.recent_dynamic_velocity_decay
                    effective_speed = math.hypot(effective_vx, effective_vy)
                    if effective_speed < min(static_speed_thresh * 0.5, 0.03):
                        effective_vx = 0.0
                        effective_vy = 0.0
                        effective_speed = 0.0
                else:
                    effective_vx = 0.0
                    effective_vy = 0.0
                    effective_speed = 0.0
            if observed_dynamic:
                t["last_dynamic_t"] = float(stamp_sec)
            recent_dynamic = (
                observed_dynamic
                or (
                    self.recent_dynamic_hold_s > 0.0
                    and (stamp_sec - float(t.get("last_dynamic_t", 0.0)))
                    <= self.recent_dynamic_hold_s
                )
            )
            if (
                effective_speed < static_speed_thresh
                and (not recent_dynamic)
                and (not self._should_publish_static_track(raw_label, t))
            ):
                continue
            if not self._pose_has_free_space_support(float(t["x"]), float(t["y"])):
                continue

            obj = TrackedObject()
            obj.id = int(tid)
            label = self._label_track(t["size_x"], t["size_y"], effective_speed)
            if recent_dynamic and effective_speed < static_speed_thresh:
                label = "recent_" + label
            obj.label = label
            obj.confidence = float(t["score"])
            obj.pose.position.x = float(t["x"])
            obj.pose.position.y = float(t["y"])
            obj.pose.position.z = 0.0
            obj.pose.orientation.w = 1.0
            obj.twist.linear.x = effective_vx
            obj.twist.linear.y = effective_vy
            obj.twist.linear.z = 0.0
            obj.size.x = float(t["size_x"])
            obj.size.y = float(t["size_y"])
            obj.size.z = 1.5
            out.objects.append(obj)

        self.pub_tracks.publish(out)

    def publish_persistence_grid(self, stamp):
        if self.drivable_grid is None:
            return

        grid = OccupancyGrid()
        grid.header.stamp = stamp if stamp.to_sec() > 0.0 else rospy.Time.now()
        grid.header.frame_id = self.drivable_grid.header.frame_id or "map"
        grid.info = self.drivable_grid.info
        total = int(grid.info.width) * int(grid.info.height)
        grid.data = [0] * total

        for (ix, iy), cell in self.persistent_cells.items():
            score = float(cell.get("score", 0.0))
            if score < self.persistence_publish_threshold:
                continue
            cx, cy = self._cell_center(ix, iy)
            gx, gy = self._world_to_grid(grid, cx, cy)
            if not self._in_bounds(grid, gx, gy):
                continue
            idx = gy * int(grid.info.width) + gx
            val = int(
                max(
                    1.0,
                    min(
                        100.0,
                        100.0 * score / max(1.0, self.persistence_max_value),
                    ),
                )
            )
            if val > int(grid.data[idx]):
                grid.data[idx] = val

        self.pub_grid.publish(grid)

    def publish_detector_markers(self, stamp):
        marker_array = MarkerArray()
        frame_id = "map"
        lifetime = rospy.Duration(self.marker_lifetime_s)

        delete_all = Marker()
        delete_all.header.stamp = stamp
        delete_all.header.frame_id = frame_id
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        marker_id = 0

        if self.show_provisional_cells and self.last_provisional_cells:
            provisional = Marker()
            provisional.header.stamp = stamp
            provisional.header.frame_id = frame_id
            provisional.ns = "provisional_cells"
            provisional.id = marker_id
            marker_id += 1
            provisional.type = Marker.CUBE_LIST
            provisional.action = Marker.ADD
            provisional.pose.orientation.w = 1.0
            provisional.scale.x = self.cell_size_m
            provisional.scale.y = self.cell_size_m
            provisional.scale.z = 0.10
            provisional.color.r = 1.0
            provisional.color.g = 0.72
            provisional.color.b = 0.20
            provisional.color.a = 0.35
            provisional.lifetime = lifetime
            for ix, iy in self.last_provisional_cells:
                cx, cy = self._cell_center(ix, iy)
                p = Point()
                p.x = cx
                p.y = cy
                p.z = self.z_offset_m
                provisional.points.append(p)
            marker_array.markers.append(provisional)

        if self.last_confirmed_cells:
            confirmed = Marker()
            confirmed.header.stamp = stamp
            confirmed.header.frame_id = frame_id
            confirmed.ns = "confirmed_cells"
            confirmed.id = marker_id
            marker_id += 1
            confirmed.type = Marker.CUBE_LIST
            confirmed.action = Marker.ADD
            confirmed.pose.orientation.w = 1.0
            confirmed.scale.x = self.cell_size_m
            confirmed.scale.y = self.cell_size_m
            confirmed.scale.z = 0.12
            confirmed.color.r = 0.20
            confirmed.color.g = 0.92
            confirmed.color.b = 0.36
            confirmed.color.a = 0.45
            confirmed.lifetime = lifetime
            for ix, iy in self.last_confirmed_cells:
                cx, cy = self._cell_center(ix, iy)
                p = Point()
                p.x = cx
                p.y = cy
                p.z = self.z_offset_m + 0.04
                confirmed.points.append(p)
            marker_array.markers.append(confirmed)

        for cluster in self.last_clusters:
            box = Marker()
            box.header.stamp = stamp
            box.header.frame_id = frame_id
            box.ns = "cluster_boxes"
            box.id = marker_id
            marker_id += 1
            box.type = Marker.CUBE
            box.action = Marker.ADD
            box.pose.orientation.w = 1.0
            box.pose.position.x = float(cluster["x"])
            box.pose.position.y = float(cluster["y"])
            box.pose.position.z = self.z_offset_m + 0.45
            box.scale.x = max(0.10, float(cluster["size_x"]))
            box.scale.y = max(0.10, float(cluster["size_y"]))
            box.scale.z = 0.90
            if cluster.get("confirmed", False):
                box.color.r = 0.15
                box.color.g = 0.95
                box.color.b = 0.35
                box.color.a = 0.28
            else:
                box.color.r = 1.0
                box.color.g = 0.78
                box.color.b = 0.16
                box.color.a = 0.20
            box.lifetime = lifetime
            marker_array.markers.append(box)

            if self.show_labels:
                text = Marker()
                text.header.stamp = stamp
                text.header.frame_id = frame_id
                text.ns = "cluster_labels"
                text.id = marker_id
                marker_id += 1
                text.type = Marker.TEXT_VIEW_FACING
                text.action = Marker.ADD
                text.pose.orientation.w = 1.0
                text.pose.position.x = float(cluster["x"])
                text.pose.position.y = float(cluster["y"])
                text.pose.position.z = self.z_offset_m + 1.05
                text.scale.z = self.text_height_m
                text.color.r = 1.0
                text.color.g = 1.0
                text.color.b = 1.0
                text.color.a = 0.95
                text.text = "cells={} score={:.0f}".format(
                    int(cluster.get("cell_count", 0)),
                    float(cluster.get("max_score", 0.0)),
                )
                text.lifetime = lifetime
                marker_array.markers.append(text)

        if self.show_velocity:
            for tid, track in self.tracks.items():
                vx = float(track.get("vx", 0.0))
                vy = float(track.get("vy", 0.0))
                if math.hypot(vx, vy) < 0.02:
                    continue
                arrow = Marker()
                arrow.header.stamp = stamp
                arrow.header.frame_id = frame_id
                arrow.ns = "velocity"
                arrow.id = marker_id
                marker_id += 1
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                arrow.scale.x = 0.07
                arrow.scale.y = 0.12
                arrow.scale.z = 0.12
                arrow.color.r = 0.18
                arrow.color.g = 0.75
                arrow.color.b = 1.0
                arrow.color.a = 0.95
                start = Point()
                start.x = float(track.get("x", 0.0))
                start.y = float(track.get("y", 0.0))
                start.z = self.z_offset_m + 0.85
                end = Point()
                end.x = start.x + vx * 0.8
                end.y = start.y + vy * 0.8
                end.z = start.z
                arrow.points = [start, end]
                arrow.lifetime = lifetime
                marker_array.markers.append(arrow)

        self.pub_markers.publish(marker_array)


def main():
    rospy.init_node("persistent_obstacle_detector", anonymous=False)
    PersistentObstacleDetector()
    rospy.spin()


if __name__ == "__main__":
    main()
