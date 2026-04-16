#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rospy
import tf.transformations as transformations
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

from dynamic_window_approach.msg import TrackedObject, TrackedObjectArray


class CloudClusterTracker:
    def __init__(self):
        self.pointcloud_topic = rospy.get_param("~pointcloud_topic", "/ouster/points")
        self.odom_topic = rospy.get_param("~odom_topic", "/lio_localizer/odometry/optimization")
        self.output_topic = rospy.get_param("~output_topic", "/perception/tracked_objects")
        self.drivable_grid_topic = rospy.get_param(
            "~drivable_grid_topic", "/lio_sam/drivable_area/grid"
        )

        self.min_z = float(rospy.get_param("~min_z", -0.4))
        self.max_z = float(rospy.get_param("~max_z", 2.2))
        self.max_range_m = float(rospy.get_param("~max_range_m", 25.0))
        self.downsample = max(1, int(rospy.get_param("~downsample", 5)))
        self.cell_size_m = max(0.1, float(rospy.get_param("~cell_size_m", 0.40)))
        self.min_points_per_cell = max(1, int(rospy.get_param("~min_points_per_cell", 2)))
        self.min_cluster_cells = max(2, int(rospy.get_param("~min_cluster_cells", 3)))

        self.max_assoc_dist_m = max(0.1, float(rospy.get_param("~max_assoc_dist_m", 1.6)))
        self.dynamic_assoc_bonus_m = max(
            0.0, float(rospy.get_param("~dynamic_assoc_bonus_m", 0.5))
        )
        self.track_timeout_s = max(0.1, float(rospy.get_param("~track_timeout_s", 1.0)))
        self.vel_alpha = min(1.0, max(0.01, float(rospy.get_param("~vel_alpha", 0.4))))
        # Static obstacles are already handled by costmaps / raw pointcloud
        # blocking in the rest of the stack. Publishing static tracks here tends
        # to create false "static_vehicle" behavior stops from walls or map
        # structures when localization jitters.
        self.publish_static = bool(rospy.get_param("~publish_static", False))
        self.publish_static_persons = bool(
            rospy.get_param("~publish_static_persons", True)
        )
        self.publish_static_large_obstacles = bool(
            rospy.get_param("~publish_static_large_obstacles", True)
        )
        self.static_speed_thresh_mps = max(0.01, float(rospy.get_param("~static_speed_thresh_mps", 0.15)))
        self.dynamic_min_age = max(1, int(rospy.get_param("~dynamic_min_age", 4)))
        self.pedestrian_static_speed_thresh_mps = max(
            0.01,
            min(
                self.static_speed_thresh_mps,
                float(rospy.get_param("~pedestrian_static_speed_thresh_mps", 0.10)),
            ),
        )
        self.pedestrian_dynamic_min_age = max(
            1,
            min(
                self.dynamic_min_age,
                int(rospy.get_param("~pedestrian_dynamic_min_age", 2)),
            ),
        )
        self.position_jitter_m = max(0.0, float(rospy.get_param("~position_jitter_m", 0.12)))
        # Static structures can drift slightly in the map frame as localization
        # settles. Require some travel from the first observation before we
        # publish a track as dynamic.
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
                        max(0.12, self.position_jitter_m * 0.9),
                    )
                ),
            ),
        )
        self.recent_dynamic_hold_s = max(
            0.0, float(rospy.get_param("~recent_dynamic_hold_s", 1.5))
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
        self.forward_non_drivable_detection_enabled = bool(
            rospy.get_param("~forward_non_drivable_detection_enabled", True)
        )
        self.forward_non_drivable_forward_range_m = max(
            0.0, float(rospy.get_param("~forward_non_drivable_forward_range_m", 18.0))
        )
        self.forward_non_drivable_side_lateral_m = max(
            0.0, float(rospy.get_param("~forward_non_drivable_side_lateral_m", 4.0))
        )
        self.forward_non_drivable_rear_margin_m = float(
            rospy.get_param("~forward_non_drivable_rear_margin_m", -0.4)
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
        self.prediction_lead_s = max(
            0.0, min(self.track_timeout_s, float(rospy.get_param("~prediction_lead_s", 0.10)))
        )
        self.pedestrian_prediction_lead_s = max(
            0.0,
            min(
                self.track_timeout_s,
                float(rospy.get_param("~pedestrian_prediction_lead_s", 0.18)),
            ),
        )

        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.have_odom = False
        self.drivable_grid = None

        self.next_track_id = 1
        self.tracks = {}

        self.pub_tracks = rospy.Publisher(self.output_topic, TrackedObjectArray, queue_size=2)
        self.sub_odom = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=20)
        self.sub_drivable = rospy.Subscriber(
            self.drivable_grid_topic, OccupancyGrid, self.drivable_grid_callback, queue_size=3
        )
        self.sub_cloud = rospy.Subscriber(self.pointcloud_topic, PointCloud2, self.cloud_callback, queue_size=1)

        rospy.loginfo(
            "cloud_cluster_tracker started | cloud=%s odom=%s grid=%s out=%s z=[%.1f, %.1f] range=%.1fm downsample=%d cell=%.2fm support=%dpts/%dcells dyn_age=%d ped_age=%d jitter=%.2fm disp=%.2fm ped_disp=%.2fm recent_hold=%.2fs assoc_bonus=%.2fm lead=%.2fs ped_lead=%.2fs decay=%.2f static=%s static_person=%s static_large=%s map_subtract=%s radius=%.2fm grid_relax=%s@%.1fm forward_non_drivable=%s front=%.1fm side=%.1fm rear=%.1fm free_support=%.2fm/%dcells far_support=%dcells",
            self.pointcloud_topic,
            self.odom_topic,
            self.drivable_grid_topic,
            self.output_topic,
            self.min_z,
            self.max_z,
            self.max_range_m,
            self.downsample,
            self.cell_size_m,
            self.min_points_per_cell,
            self.min_cluster_cells,
            self.dynamic_min_age,
            self.pedestrian_dynamic_min_age,
            self.position_jitter_m,
            self.dynamic_min_displacement_m,
            self.pedestrian_dynamic_min_displacement_m,
            self.recent_dynamic_hold_s,
            self.dynamic_assoc_bonus_m,
            self.prediction_lead_s,
            self.pedestrian_prediction_lead_s,
            self.recent_dynamic_velocity_decay,
            "on" if self.publish_static else "off",
            "on" if self.publish_static_persons else "off",
            "on" if self.publish_static_large_obstacles else "off",
            "on" if self.known_map_subtraction_enabled else "off",
            self.known_map_subtraction_radius_m,
            "on" if self.allow_out_of_grid_clusters else "off",
            self.grid_relaxation_distance_m,
            "on" if self.forward_non_drivable_detection_enabled else "off",
            self.forward_non_drivable_forward_range_m,
            self.forward_non_drivable_side_lateral_m,
            self.forward_non_drivable_rear_margin_m,
            self.free_space_support_radius_m,
            self.free_space_support_min_cells,
            self.far_field_free_space_support_min_cells,
        )

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.odom_x = float(p.x)
        self.odom_y = float(p.y)
        self.odom_yaw = transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
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

    def _map_to_local(self, x, y):
        dx = float(x) - self.odom_x
        dy = float(y) - self.odom_y
        c = math.cos(self.odom_yaw)
        s = math.sin(self.odom_yaw)
        lx = c * dx + s * dy
        ly = -s * dx + c * dy
        return lx, ly

    def _pose_in_forward_non_drivable_roi(self, x, y):
        if not self.forward_non_drivable_detection_enabled:
            return False
        lx, ly = self._map_to_local(x, y)
        if lx < self.forward_non_drivable_rear_margin_m:
            return False
        if lx > self.forward_non_drivable_forward_range_m:
            return False
        return abs(ly) <= self.forward_non_drivable_side_lateral_m

    def _cluster_in_forward_non_drivable_roi(self, cluster):
        return self._pose_in_forward_non_drivable_roi(cluster["x"], cluster["y"])

    @staticmethod
    def _world_to_grid(g, x, y):
        res = float(g.info.resolution)
        gx = int(math.floor((float(x) - float(g.info.origin.position.x)) / res))
        gy = int(math.floor((float(y) - float(g.info.origin.position.y)) / res))
        return gx, gy

    @staticmethod
    def _in_bounds(g, gx, gy):
        return 0 <= gx < int(g.info.width) and 0 <= gy < int(g.info.height)

    def _grid_cell_is_drivable_free(self, g, gx, gy):
        if not self._in_bounds(g, gx, gy):
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
                if self._cluster_in_forward_non_drivable_roi(cluster):
                    cluster["relaxed_forward_roi"] = True
                    return False
                return True
        if saw_in_bounds:
            return False
        if not self.allow_out_of_grid_clusters:
            if self._cluster_in_forward_non_drivable_roi(cluster):
                cluster["relaxed_forward_roi"] = True
                return False
            return True
        return self._map_distance_from_robot(
            cluster["x"], cluster["y"]
        ) < self.grid_relaxation_distance_m

    def _pose_has_free_space_support(self, x, y, relaxed_forward_ok=False):
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
            return far_field_relaxed or (
                relaxed_forward_ok and self._pose_in_forward_non_drivable_roi(x, y)
            )

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
        return (far_field_relaxed and free_cells > 0) or (
            relaxed_forward_ok and self._pose_in_forward_non_drivable_roi(x, y)
        )

    def _extract_clusters(self, msg):
        if not self.have_odom:
            return []

        cells = {}
        rr = self.max_range_m * self.max_range_m
        i = 0
        for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
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
            cell = cells.setdefault(
                (ix, iy),
                {
                    "count": 0,
                    "sum_x": 0.0,
                    "sum_y": 0.0,
                    "min_x": mx,
                    "max_x": mx,
                    "min_y": my,
                    "max_y": my,
                },
            )
            cell["count"] += 1
            cell["sum_x"] += mx
            cell["sum_y"] += my
            cell["min_x"] = min(float(cell["min_x"]), mx)
            cell["max_x"] = max(float(cell["max_x"]), mx)
            cell["min_y"] = min(float(cell["min_y"]), my)
            cell["max_y"] = max(float(cell["max_y"]), my)

        occ = {
            k
            for k, cell in cells.items()
            if int(cell.get("count", 0)) >= self.min_points_per_cell
        }
        visited = set()
        clusters = []
        for seed in occ:
            if seed in visited:
                continue
            stack = [seed]
            visited.add(seed)
            comp = []
            while stack:
                c = stack.pop()
                comp.append(c)
                cx, cy = c
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nc = (cx + dx, cy + dy)
                        if nc in occ and nc not in visited:
                            visited.add(nc)
                            stack.append(nc)
            if len(comp) < self.min_cluster_cells:
                continue

            min_x = 1e9
            min_y = 1e9
            max_x = -1e9
            max_y = -1e9
            weight_sum = 0
            weighted_x_sum = 0.0
            weighted_y_sum = 0.0
            for (ix, iy) in comp:
                cell = cells.get((ix, iy))
                if cell is None:
                    continue
                w = float(max(1, int(cell.get("count", 1))))
                cx = float(cell.get("sum_x", 0.0)) / w
                cy = float(cell.get("sum_y", 0.0)) / w
                min_x = min(min_x, float(cell.get("min_x", cx)))
                min_y = min(min_y, float(cell.get("min_y", cy)))
                max_x = max(max_x, float(cell.get("max_x", cx)))
                max_y = max(max_y, float(cell.get("max_y", cy)))
                weight_sum += w
                weighted_x_sum += cx * w
                weighted_y_sum += cy * w
            if weight_sum <= 0:
                continue
            cx = weighted_x_sum / weight_sum
            cy = weighted_y_sum / weight_sum
            cluster = {
                "x": cx,
                "y": cy,
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
                "size_x": max(0.2, max_x - min_x + 0.5 * self.cell_size_m),
                "size_y": max(0.2, max_y - min_y + 0.5 * self.cell_size_m),
                "score": min(1.0, len(comp) / 20.0),
                "relaxed_forward_roi": False,
            }
            if self._cluster_overlaps_known_map_obstacle(cluster):
                continue
            if not self._pose_has_free_space_support(
                cluster["x"],
                cluster["y"],
                relaxed_forward_ok=bool(cluster.get("relaxed_forward_roi", False)),
            ):
                continue
            clusters.append(cluster)
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
                    and (now_sec - float(t.get("last_dynamic_t", 0.0))) <= self.recent_dynamic_hold_s
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
            dt = max(1e-3, now_sec - t["last_t"])
            dx = c["x"] - t["x"]
            dy = c["y"] - t["y"]
            if math.hypot(dx, dy) < self.position_jitter_m:
                vx_obs = 0.0
                vy_obs = 0.0
            else:
                vx_obs = dx / dt
                vy_obs = dy / dt
            t["vx"] = (1.0 - self.vel_alpha) * t["vx"] + self.vel_alpha * vx_obs
            t["vy"] = (1.0 - self.vel_alpha) * t["vy"] + self.vel_alpha * vy_obs
            t["x"] = c["x"]
            t["y"] = c["y"]
            t["size_x"] = c["size_x"]
            t["size_y"] = c["size_y"]
            t["score"] = c["score"]
            t["relaxed_forward_roi"] = bool(c.get("relaxed_forward_roi", False))
            t["last_t"] = now_sec
            t["age"] += 1

        for ci in unmatched_clusters:
            c = clusters[ci]
            tid = self.next_track_id
            self.next_track_id += 1
            self.tracks[tid] = {
                "x": c["x"],
                "y": c["y"],
                "anchor_x": c["x"],
                "anchor_y": c["y"],
                "vx": 0.0,
                "vy": 0.0,
                "size_x": c["size_x"],
                "size_y": c["size_y"],
                "score": c["score"],
                "relaxed_forward_roi": bool(c.get("relaxed_forward_roi", False)),
                "last_t": now_sec,
                "age": 1,
                "last_dynamic_t": 0.0,
            }

        stale = [tid for tid, t in self.tracks.items() if (now_sec - t["last_t"]) > self.track_timeout_s]
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
            now_sec = msg.header.stamp.to_sec() if msg.header.stamp.to_sec() > 0.0 else rospy.Time.now().to_sec()
            clusters = self._extract_clusters(msg)
            self._associate_and_update(clusters, now_sec)
            self.publish_tracks(msg.header.stamp)
        except Exception as e:
            rospy.logwarn_throttle(1.0, "cloud_cluster_tracker error: %s", str(e))

    def publish_tracks(self, stamp):
        out = TrackedObjectArray()
        out.header.stamp = stamp if stamp.to_sec() > 0.0 else rospy.Time.now()
        out.header.frame_id = "map"
        stamp_sec = out.header.stamp.to_sec() if out.header.stamp.to_sec() > 0.0 else rospy.Time.now().to_sec()
        for tid, t in self.tracks.items():
            speed = math.hypot(t["vx"], t["vy"])
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
                and (stamp_sec - float(t.get("last_dynamic_t", 0.0))) <= self.recent_dynamic_hold_s
            )
            observed_dynamic = True
            # Require a track to persist for a few updates before we trust any
            # measured motion; otherwise doorway edges and map jitter can look
            # like short-lived moving objects.
            if t["age"] < dynamic_min_age:
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
                    and (stamp_sec - float(t.get("last_dynamic_t", 0.0))) <= self.recent_dynamic_hold_s
                )
            )
            if (
                effective_speed < static_speed_thresh
                and (not recent_dynamic)
                and (not self._should_publish_static_track(raw_label, t))
            ):
                continue
            if not self._pose_has_free_space_support(
                t["x"],
                t["y"],
                relaxed_forward_ok=bool(t.get("relaxed_forward_roi", False)),
            ):
                continue
            obj = TrackedObject()
            obj.id = int(tid)
            label = self._label_track(t["size_x"], t["size_y"], effective_speed)
            if recent_dynamic and effective_speed < static_speed_thresh:
                label = "recent_" + label
            lead_s = (
                self.pedestrian_prediction_lead_s
                if is_person_like
                else self.prediction_lead_s
            )
            if (not recent_dynamic) or effective_speed < static_speed_thresh:
                lead_s = 0.0
            obj.label = label
            obj.confidence = float(t["score"])
            obj.pose.position.x = float(t["x"]) + effective_vx * lead_s
            obj.pose.position.y = float(t["y"]) + effective_vy * lead_s
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


def main():
    rospy.init_node("cloud_cluster_tracker", anonymous=False)
    CloudClusterTracker()
    rospy.spin()


if __name__ == "__main__":
    main()
