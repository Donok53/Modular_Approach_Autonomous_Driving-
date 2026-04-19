#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math

import numpy as np
import rospy
import tf.transformations as transformations
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Imu, PointCloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


class LidarObstaclePerceptionExperimental:
    def __init__(self):
        self.pointcloud_topic = rospy.get_param(
            "~pointcloud_topic", "/lio_localizer/localization/cloud_deskewed"
        )
        self.imu_topic = rospy.get_param("~imu_topic", "/imu_correct")
        self.odom_topic = rospy.get_param(
            "~odom_topic", "/lio_localizer/odometry/optimization"
        )
        self.map_frame = str(rospy.get_param("~map_frame", "map")).strip() or "map"
        self.require_odom_for_processing = bool(
            rospy.get_param("~require_odom_for_processing", False)
        )
        self.ground_only_mode = bool(rospy.get_param("~ground_only_mode", False))
        self.input_cloud_is_non_ground = bool(
            rospy.get_param("~input_cloud_is_non_ground", False)
        )
        self.publish_static_current_frame_only = bool(
            rospy.get_param("~publish_static_current_frame_only", False)
        )
        self.local_fallback_frame = str(
            rospy.get_param("~local_fallback_frame", "")
        ).strip()

        self.ground_cloud_topic = rospy.get_param(
            "~ground_cloud_topic", "/experimental/lidar_obstacle_perception/ground_cloud"
        )
        self.non_ground_cloud_topic = rospy.get_param(
            "~non_ground_cloud_topic",
            "/experimental/lidar_obstacle_perception/non_ground_cloud",
        )
        self.static_cloud_topic = rospy.get_param(
            "~static_cloud_topic", "/experimental/lidar_obstacle_perception/static_cloud"
        )
        self.dynamic_cloud_topic = rospy.get_param(
            "~dynamic_cloud_topic", "/experimental/lidar_obstacle_perception/dynamic_cloud"
        )
        self.dynamic_markers_topic = rospy.get_param(
            "~dynamic_markers_topic",
            "/experimental/lidar_obstacle_perception/dynamic_markers",
        )
        self.local_obstacle_grid_topic = rospy.get_param(
            "~local_obstacle_grid_topic",
            "/experimental/lidar_obstacle_perception/local_obstacle_grid",
        )

        self.max_range_m = max(1.0, float(rospy.get_param("~max_range_m", 25.0)))
        self.min_z = float(rospy.get_param("~min_z", -1.5))
        self.max_z = float(rospy.get_param("~max_z", 3.0))
        self.downsample = max(1, int(rospy.get_param("~downsample", 2)))

        self.use_imu_leveling = bool(rospy.get_param("~use_imu_leveling", True))
        self.ground_cell_size_m = max(
            0.05, float(rospy.get_param("~ground_cell_size_m", 0.25))
        )
        self.ground_neighbor_radius_cells = max(
            0, int(rospy.get_param("~ground_neighbor_radius_cells", 1))
        )
        self.ground_clearance_m = max(
            0.02, float(rospy.get_param("~ground_clearance_m", 0.16))
        )
        self.ground_range_clearance_per_m = max(
            0.0, float(rospy.get_param("~ground_range_clearance_per_m", 0.015))
        )
        self.ground_pitch_clearance_gain_m = max(
            0.0, float(rospy.get_param("~ground_pitch_clearance_gain_m", 0.20))
        )

        self.cluster_cell_size_m = max(
            0.05, float(rospy.get_param("~cluster_cell_size_m", 0.30))
        )
        self.cluster_min_points_per_cell = max(
            1, int(rospy.get_param("~cluster_min_points_per_cell", 2))
        )
        self.cluster_min_cells = max(
            1, int(rospy.get_param("~cluster_min_cells", 2))
        )
        self.cluster_min_points = max(
            1, int(rospy.get_param("~cluster_min_points", 12))
        )

        self.track_assoc_dist_m = max(
            0.2, float(rospy.get_param("~track_assoc_dist_m", 1.8))
        )
        self.track_timeout_s = max(
            0.1, float(rospy.get_param("~track_timeout_s", 1.2))
        )
        self.track_velocity_alpha = clamp(
            float(rospy.get_param("~track_velocity_alpha", 0.45)), 0.05, 1.0
        )
        self.dynamic_min_age = max(
            1, int(rospy.get_param("~dynamic_min_age", 6))
        )
        self.dynamic_min_motion_hits = max(
            1, int(rospy.get_param("~dynamic_min_motion_hits", 4))
        )
        self.dynamic_speed_thresh_mps = max(
            0.05, float(rospy.get_param("~dynamic_speed_thresh_mps", 0.35))
        )
        self.dynamic_min_displacement_m = max(
            0.05, float(rospy.get_param("~dynamic_min_displacement_m", 0.35))
        )
        self.dynamic_candidate_max_span_m = max(
            0.5, float(rospy.get_param("~dynamic_candidate_max_span_m", 4.5))
        )
        self.dynamic_candidate_max_area_m2 = max(
            0.5, float(rospy.get_param("~dynamic_candidate_max_area_m2", 8.0))
        )
        self.dynamic_candidate_max_points = max(
            10, int(rospy.get_param("~dynamic_candidate_max_points", 220))
        )
        self.dynamic_candidate_max_height_m = max(
            0.5, float(rospy.get_param("~dynamic_candidate_max_height_m", 2.4))
        )
        self.dynamic_candidate_max_aspect_ratio = max(
            1.0, float(rospy.get_param("~dynamic_candidate_max_aspect_ratio", 3.5))
        )

        self.static_voxel_size_m = max(
            0.05, float(rospy.get_param("~static_voxel_size_m", 0.20))
        )
        self.static_confirm_hits = max(
            1, int(rospy.get_param("~static_confirm_hits", 3))
        )
        self.static_ttl_s = max(0.5, float(rospy.get_param("~static_ttl_s", 30.0)))
        self.dynamic_exclusion_radius_m = max(
            0.1, float(rospy.get_param("~dynamic_exclusion_radius_m", 1.2))
        )
        self.dynamic_static_support_min_hits = max(
            1,
            int(
                rospy.get_param(
                    "~dynamic_static_support_min_hits",
                    max(2, self.static_confirm_hits - 1),
                )
            ),
        )
        self.dynamic_static_support_ratio = clamp(
            float(rospy.get_param("~dynamic_static_support_ratio", 0.08)), 0.0, 1.0
        )
        self.dynamic_static_support_neighbor_cells = max(
            0, int(rospy.get_param("~dynamic_static_support_neighbor_cells", 2))
        )
        self.motion_suppress_yaw_rate_radps = max(
            0.0, float(rospy.get_param("~motion_suppress_yaw_rate_radps", 0.18))
        )
        self.motion_suppress_speed_mps = max(
            0.0, float(rospy.get_param("~motion_suppress_speed_mps", 0.60))
        )
        self.local_grid_resolution_m = max(
            0.05, float(rospy.get_param("~local_grid_resolution_m", 0.20))
        )
        self.local_grid_width_m = max(
            self.local_grid_resolution_m,
            float(rospy.get_param("~local_grid_width_m", 20.0)),
        )
        self.local_grid_height_m = max(
            self.local_grid_resolution_m,
            float(rospy.get_param("~local_grid_height_m", 20.0)),
        )
        self.local_grid_mark_value = clamp(
            int(rospy.get_param("~local_grid_mark_value", 100)), 1, 100
        )

        self.marker_lifetime_s = max(
            0.0, float(rospy.get_param("~marker_lifetime_s", 0.6))
        )
        self.show_labels = bool(rospy.get_param("~show_labels", True))
        self.show_velocity = bool(rospy.get_param("~show_velocity", True))

        self.have_imu = False
        self.imu_roll = 0.0
        self.imu_pitch = 0.0
        self.imu_yaw_rate = 0.0

        self.have_odom = False
        self.odom_pos = np.zeros(3, dtype=np.float32)
        self.odom_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.odom_roll = 0.0
        self.odom_pitch = 0.0
        self.odom_yaw = 0.0
        self.odom_speed = 0.0
        self.odom_yaw_rate = 0.0

        self.next_track_id = 1
        self.tracks = {}
        self.static_voxels = {}

        self.pub_ground = rospy.Publisher(
            self.ground_cloud_topic, PointCloud2, queue_size=1
        )
        self.pub_non_ground = rospy.Publisher(
            self.non_ground_cloud_topic, PointCloud2, queue_size=1
        )
        self.pub_static = rospy.Publisher(
            self.static_cloud_topic, PointCloud2, queue_size=1
        )
        self.pub_dynamic = rospy.Publisher(
            self.dynamic_cloud_topic, PointCloud2, queue_size=1
        )
        self.pub_dynamic_markers = rospy.Publisher(
            self.dynamic_markers_topic, MarkerArray, queue_size=1
        )
        self.pub_local_grid = rospy.Publisher(
            self.local_obstacle_grid_topic, OccupancyGrid, queue_size=1
        )

        self.sub_imu = rospy.Subscriber(
            self.imu_topic, Imu, self.imu_callback, queue_size=50
        )
        self.sub_odom = rospy.Subscriber(
            self.odom_topic, Odometry, self.odom_callback, queue_size=20
        )
        self.sub_cloud = rospy.Subscriber(
            self.pointcloud_topic, PointCloud2, self.cloud_callback, queue_size=1
        )

        rospy.loginfo(
            "lidar_obstacle_perception_experimental started | cloud=%s imu=%s odom=%s range=%.1fm downsample=%d ground_cell=%.2fm clearance=%.2fm cluster_cell=%.2fm track_assoc=%.2fm static_voxel=%.2fm ttl=%.1fs grid=%.1fx%.1fm@%.2fm ground_only=%s input_non_ground=%s current_frame_static=%s",
            self.pointcloud_topic,
            self.imu_topic,
            self.odom_topic,
            self.max_range_m,
            self.downsample,
            self.ground_cell_size_m,
            self.ground_clearance_m,
            self.cluster_cell_size_m,
            self.track_assoc_dist_m,
            self.static_voxel_size_m,
            self.static_ttl_s,
            self.local_grid_width_m,
            self.local_grid_height_m,
            self.local_grid_resolution_m,
            str(self.ground_only_mode).lower(),
            str(self.input_cloud_is_non_ground).lower(),
            str(self.publish_static_current_frame_only).lower(),
        )

    @staticmethod
    def _empty_points():
        return np.empty((0, 3), dtype=np.float32)

    def imu_callback(self, msg):
        q = msg.orientation
        self.imu_roll, self.imu_pitch, _ = transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w]
        )
        self.imu_yaw_rate = float(msg.angular_velocity.z)
        self.have_imu = True

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.odom_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
        self.odom_quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float32)
        self.odom_roll, self.odom_pitch, self.odom_yaw = (
            transformations.euler_from_quaternion(self.odom_quat)
        )
        twist = msg.twist.twist
        self.odom_speed = math.hypot(float(twist.linear.x), float(twist.linear.y))
        self.odom_yaw_rate = float(twist.angular.z)
        self.have_odom = True

    def _current_roll_pitch(self):
        if self.have_imu and self.use_imu_leveling:
            return self.imu_roll, self.imu_pitch
        if self.have_odom:
            return self.odom_roll, self.odom_pitch
        return 0.0, 0.0

    def _extract_points(self, msg):
        rr = self.max_range_m * self.max_range_m
        points = []
        count = 0
        for p in point_cloud2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True
        ):
            count += 1
            if self.downsample > 1 and (count % self.downsample) != 0:
                continue
            x = float(p[0])
            y = float(p[1])
            z = float(p[2])
            if z < self.min_z or z > self.max_z:
                continue
            if (x * x + y * y) > rr:
                continue
            points.append((x, y, z))
        if not points:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(points, dtype=np.float32)

    def _level_points(self, points_local):
        if points_local.shape[0] == 0:
            return points_local.copy()
        roll, pitch = self._current_roll_pitch()
        if abs(roll) < 1e-4 and abs(pitch) < 1e-4:
            return points_local.copy()
        rot = transformations.euler_matrix(-roll, -pitch, 0.0)[:3, :3]
        return points_local.dot(rot.T)

    def _transform_points_to_map(self, points_local):
        if points_local.shape[0] == 0:
            return points_local.copy()
        rot = transformations.quaternion_matrix(self.odom_quat)[:3, :3]
        return points_local.dot(rot.T) + self.odom_pos

    def _processing_frame_id(self, msg):
        if self.have_odom:
            return self.map_frame
        if self.local_fallback_frame:
            return self.local_fallback_frame
        frame_id = str(msg.header.frame_id).strip()
        return frame_id or "base_link"

    @staticmethod
    def _points_to_cloud(points_xyz, stamp, frame_id):
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id
        if points_xyz is None or len(points_xyz) == 0:
            return point_cloud2.create_cloud_xyz32(header, [])
        return point_cloud2.create_cloud_xyz32(
            header, np.asarray(points_xyz, dtype=np.float32).tolist()
        )

    def _segment_ground(self, leveled_points):
        n = leveled_points.shape[0]
        if n == 0:
            return np.zeros((0,), dtype=bool)

        _, pitch = self._current_roll_pitch()
        pitch_clearance = abs(float(pitch)) * self.ground_pitch_clearance_gain_m

        min_z_cells = {}
        cell_keys = []
        inv = 1.0 / self.ground_cell_size_m
        for x, y, z in leveled_points:
            key = (int(math.floor(x * inv)), int(math.floor(y * inv)))
            cell_keys.append(key)
            prev = min_z_cells.get(key)
            if prev is None or z < prev:
                min_z_cells[key] = float(z)

        ground_ref = {}
        r = self.ground_neighbor_radius_cells
        for key, base_min in min_z_cells.items():
            gx, gy = key
            best = base_min
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nbr = min_z_cells.get((gx + dx, gy + dy))
                    if nbr is not None and nbr < best:
                        best = nbr
            ground_ref[key] = best

        mask = np.zeros((n,), dtype=bool)
        for idx, (x, y, z) in enumerate(leveled_points):
            ref_z = ground_ref.get(cell_keys[idx], float(z))
            range_clearance = math.hypot(float(x), float(y)) * self.ground_range_clearance_per_m
            clearance = self.ground_clearance_m + pitch_clearance + range_clearance
            if z <= (ref_z + clearance):
                mask[idx] = True
        return mask

    def _cluster_points(self, points_map):
        if points_map.shape[0] == 0:
            return []

        inv = 1.0 / self.cluster_cell_size_m
        cells = {}
        for idx, (x, y, z) in enumerate(points_map):
            key = (int(math.floor(x * inv)), int(math.floor(y * inv)))
            cell = cells.setdefault(key, {"indices": [], "min_z": z, "max_z": z})
            cell["indices"].append(idx)
            cell["min_z"] = min(float(cell["min_z"]), float(z))
            cell["max_z"] = max(float(cell["max_z"]), float(z))

        occ = {
            key
            for key, cell in cells.items()
            if len(cell["indices"]) >= self.cluster_min_points_per_cell
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
                cell_key = stack.pop()
                comp.append(cell_key)
                cx, cy = cell_key
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nbr = (cx + dx, cy + dy)
                        if nbr in occ and nbr not in visited:
                            visited.add(nbr)
                            stack.append(nbr)
            if len(comp) < self.cluster_min_cells:
                continue
            indices = []
            min_x = 1e9
            min_y = 1e9
            min_z = 1e9
            max_x = -1e9
            max_y = -1e9
            max_z = -1e9
            for key in comp:
                idxs = cells[key]["indices"]
                indices.extend(idxs)
                pts = points_map[idxs]
                min_x = min(min_x, float(np.min(pts[:, 0])))
                min_y = min(min_y, float(np.min(pts[:, 1])))
                min_z = min(min_z, float(np.min(pts[:, 2])))
                max_x = max(max_x, float(np.max(pts[:, 0])))
                max_y = max(max_y, float(np.max(pts[:, 1])))
                max_z = max(max_z, float(np.max(pts[:, 2])))
            if len(indices) < self.cluster_min_points:
                continue
            pts = points_map[indices]
            centroid = np.mean(pts, axis=0)
            clusters.append(
                {
                    "indices": np.asarray(indices, dtype=np.int32),
                    "centroid": centroid,
                    "size_x": max(0.10, max_x - min_x),
                    "size_y": max(0.10, max_y - min_y),
                    "size_z": max(0.10, max_z - min_z),
                    "min_x": min_x,
                    "max_x": max_x,
                    "min_y": min_y,
                    "max_y": max_y,
                    "min_z": min_z,
                    "max_z": max_z,
                    "num_points": len(indices),
                }
            )
        return clusters

    def _associate_tracks(self, clusters, now_sec):
        unmatched_tracks = set(self.tracks.keys())
        cluster_to_track = {}

        for ci, cluster in enumerate(clusters):
            best_tid = None
            best_dist = 1e9
            cx = float(cluster["centroid"][0])
            cy = float(cluster["centroid"][1])
            for tid in list(unmatched_tracks):
                track = self.tracks[tid]
                dt = max(0.0, now_sec - float(track["last_t"]))
                pred_x = float(track["x"]) + float(track["vx"]) * dt
                pred_y = float(track["y"]) + float(track["vy"]) * dt
                dist = math.hypot(cx - pred_x, cy - pred_y)
                if dist <= self.track_assoc_dist_m and dist < best_dist:
                    best_tid = tid
                    best_dist = dist
            if best_tid is not None:
                cluster_to_track[ci] = best_tid
                unmatched_tracks.discard(best_tid)

        for ci, tid in cluster_to_track.items():
            cluster = clusters[ci]
            track = self.tracks[tid]
            dt = max(1e-3, now_sec - float(track["last_t"]))
            new_x = float(cluster["centroid"][0])
            new_y = float(cluster["centroid"][1])
            vx_obs = (new_x - float(track["x"])) / dt
            vy_obs = (new_y - float(track["y"])) / dt
            track["vx"] = (1.0 - self.track_velocity_alpha) * float(
                track["vx"]
            ) + self.track_velocity_alpha * vx_obs
            track["vy"] = (1.0 - self.track_velocity_alpha) * float(
                track["vy"]
            ) + self.track_velocity_alpha * vy_obs
            track["x"] = new_x
            track["y"] = new_y
            track["z"] = float(cluster["centroid"][2])
            track["size_x"] = float(cluster["size_x"])
            track["size_y"] = float(cluster["size_y"])
            track["size_z"] = float(cluster["size_z"])
            track["num_points"] = int(cluster["num_points"])
            track["last_t"] = now_sec
            track["age"] += 1

        for ci, cluster in enumerate(clusters):
            if ci in cluster_to_track:
                continue
            tid = self.next_track_id
            self.next_track_id += 1
            self.tracks[tid] = {
                "x": float(cluster["centroid"][0]),
                "y": float(cluster["centroid"][1]),
                "z": float(cluster["centroid"][2]),
                "anchor_x": float(cluster["centroid"][0]),
                "anchor_y": float(cluster["centroid"][1]),
                "vx": 0.0,
                "vy": 0.0,
                "size_x": float(cluster["size_x"]),
                "size_y": float(cluster["size_y"]),
                "size_z": float(cluster["size_z"]),
                "num_points": int(cluster["num_points"]),
                "last_t": now_sec,
                "age": 1,
                "motion_hits": 0,
            }
            cluster_to_track[ci] = tid

        stale = [
            tid
            for tid, track in self.tracks.items()
            if (now_sec - float(track["last_t"])) > self.track_timeout_s
        ]
        for tid in stale:
            self.tracks.pop(tid, None)

        return cluster_to_track

    def _track_is_dynamic(self, track):
        speed = math.hypot(float(track["vx"]), float(track["vy"]))
        disp = math.hypot(
            float(track["x"]) - float(track["anchor_x"]),
            float(track["y"]) - float(track["anchor_y"]),
        )
        return (
            int(track["age"]) >= self.dynamic_min_age
            and int(track.get("motion_hits", 0)) >= self.dynamic_min_motion_hits
            and speed >= self.dynamic_speed_thresh_mps
            and disp >= self.dynamic_min_displacement_m
        )

    def _cluster_is_plausibly_dynamic(self, cluster):
        span = max(float(cluster["size_x"]), float(cluster["size_y"]))
        short_span = max(0.10, min(float(cluster["size_x"]), float(cluster["size_y"])))
        area = float(cluster["size_x"]) * float(cluster["size_y"])
        num_points = int(cluster["num_points"])
        height = float(cluster["size_z"])
        aspect = span / short_span
        return (
            span <= self.dynamic_candidate_max_span_m
            and area <= self.dynamic_candidate_max_area_m2
            and num_points <= self.dynamic_candidate_max_points
            and height <= self.dynamic_candidate_max_height_m
            and aspect <= self.dynamic_candidate_max_aspect_ratio
        )

    def _cluster_has_static_support(self, cluster_points):
        if cluster_points is None or len(cluster_points) == 0 or not self.static_voxels:
            return False

        support_keys = {
            key
            for key, voxel in self.static_voxels.items()
            if int(voxel["hits"]) >= self.dynamic_static_support_min_hits
        }
        if not support_keys:
            return False

        inv = 1.0 / self.static_voxel_size_m
        radius = self.dynamic_static_support_neighbor_cells
        checked = 0
        supported = 0
        sample_step = max(1, int(len(cluster_points) / 80))
        for point in cluster_points[::sample_step]:
            x, y, z = point
            key = (
                int(math.floor(float(x) * inv)),
                int(math.floor(float(y) * inv)),
                int(math.floor(float(z) * inv)),
            )
            checked += 1
            found = False
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        if (key[0] + dx, key[1] + dy, key[2] + dz) in support_keys:
                            supported += 1
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        if checked <= 0:
            return False
        return (float(supported) / float(checked)) >= self.dynamic_static_support_ratio

    def _ego_motion_is_aggressive(self):
        yaw_rate = abs(self.imu_yaw_rate) if self.have_imu else abs(self.odom_yaw_rate)
        speed = abs(self.odom_speed) if self.have_odom else 0.0
        return (
            yaw_rate >= self.motion_suppress_yaw_rate_radps
            or speed >= self.motion_suppress_speed_mps
        )

    @staticmethod
    def _dynamic_label(track):
        span = max(float(track["size_x"]), float(track["size_y"]))
        area = float(track["size_x"]) * float(track["size_y"])
        if span >= 1.8 or area >= 2.5 or int(track.get("num_points", 0)) >= 60:
            return "vehicle"
        if span <= 1.1 and area <= 1.3:
            return "pedestrian"
        return "dynamic"

    def _update_static_voxels(self, points_map, now_sec):
        if points_map.shape[0] > 0:
            inv = 1.0 / self.static_voxel_size_m
            for x, y, z in points_map:
                key = (
                    int(math.floor(float(x) * inv)),
                    int(math.floor(float(y) * inv)),
                    int(math.floor(float(z) * inv)),
                )
                voxel = self.static_voxels.get(key)
                if voxel is None:
                    self.static_voxels[key] = {
                        "x": float(x),
                        "y": float(y),
                        "z": float(z),
                        "hits": 1,
                        "last_seen": now_sec,
                    }
                else:
                    voxel["x"] = float(x)
                    voxel["y"] = float(y)
                    voxel["z"] = float(z)
                    voxel["hits"] = min(voxel["hits"] + 1, self.static_confirm_hits + 10)
                    voxel["last_seen"] = now_sec

        stale = [
            key
            for key, voxel in self.static_voxels.items()
            if (now_sec - float(voxel["last_seen"])) > self.static_ttl_s
        ]
        for key in stale:
            self.static_voxels.pop(key, None)

    def _confirmed_static_points(self):
        points = []
        for voxel in self.static_voxels.values():
            if int(voxel["hits"]) >= self.static_confirm_hits:
                points.append((voxel["x"], voxel["y"], voxel["z"]))
        if not points:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(points, dtype=np.float32)

    def _build_local_grid(self, static_points, dynamic_points, stamp, frame_id, center_xy):
        width_cells = max(
            1, int(math.ceil(self.local_grid_width_m / self.local_grid_resolution_m))
        )
        height_cells = max(
            1, int(math.ceil(self.local_grid_height_m / self.local_grid_resolution_m))
        )
        origin_x = float(center_xy[0]) - 0.5 * self.local_grid_width_m
        origin_y = float(center_xy[1]) - 0.5 * self.local_grid_height_m
        data = np.zeros((height_cells, width_cells), dtype=np.int8)

        def mark_points(points):
            for x, y, _ in points:
                gx = int(math.floor((float(x) - origin_x) / self.local_grid_resolution_m))
                gy = int(math.floor((float(y) - origin_y) / self.local_grid_resolution_m))
                if 0 <= gx < width_cells and 0 <= gy < height_cells:
                    data[gy, gx] = self.local_grid_mark_value

        mark_points(static_points)
        mark_points(dynamic_points)

        grid = OccupancyGrid()
        grid.header.stamp = stamp
        grid.header.frame_id = frame_id
        grid.info = MapMetaData()
        grid.info.map_load_time = stamp
        grid.info.resolution = self.local_grid_resolution_m
        grid.info.width = width_cells
        grid.info.height = height_cells
        grid.info.origin.position.x = origin_x
        grid.info.origin.position.y = origin_y
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.w = 1.0
        grid.data = data.reshape(-1).tolist()
        return grid

    def _build_dynamic_markers(self, dynamic_entries, stamp):
        return self._build_dynamic_markers_for_frame(dynamic_entries, stamp, self.map_frame)

    def _build_dynamic_markers_for_frame(self, dynamic_entries, stamp, frame_id):
        marker_array = MarkerArray()
        delete_all = Marker()
        delete_all.header.stamp = stamp
        delete_all.header.frame_id = frame_id
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        marker_id = 0
        lifetime = rospy.Duration(self.marker_lifetime_s)
        for entry in dynamic_entries:
            track = entry["track"]
            label = entry["label"]
            speed = math.hypot(float(track["vx"]), float(track["vy"]))
            yaw = math.atan2(float(track["vy"]), float(track["vx"])) if speed > 0.05 else 0.0
            q = transformations.quaternion_from_euler(0.0, 0.0, yaw)

            box = Marker()
            box.header.stamp = stamp
            box.header.frame_id = frame_id
            box.ns = "dynamic_boxes"
            box.id = marker_id
            marker_id += 1
            box.type = Marker.CUBE
            box.action = Marker.ADD
            box.pose.position.x = float(track["x"])
            box.pose.position.y = float(track["y"])
            box.pose.position.z = max(0.15, float(track["size_z"]) * 0.5)
            box.pose.orientation.x = q[0]
            box.pose.orientation.y = q[1]
            box.pose.orientation.z = q[2]
            box.pose.orientation.w = q[3]
            box.scale.x = max(0.20, float(track["size_x"]))
            box.scale.y = max(0.20, float(track["size_y"]))
            box.scale.z = max(0.60, float(track["size_z"]))
            box.color.r = 0.95
            box.color.g = 0.15
            box.color.b = 0.10
            box.color.a = 0.55
            box.lifetime = lifetime
            marker_array.markers.append(box)

            if self.show_velocity and speed > 0.05:
                arrow = Marker()
                arrow.header.stamp = stamp
                arrow.header.frame_id = frame_id
                arrow.ns = "dynamic_velocity"
                arrow.id = marker_id
                marker_id += 1
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                arrow.scale.x = 0.08
                arrow.scale.y = 0.14
                arrow.scale.z = 0.14
                arrow.color.r = 1.0
                arrow.color.g = 0.3
                arrow.color.b = 0.1
                arrow.color.a = 0.95
                start = Point()
                start.x = float(track["x"])
                start.y = float(track["y"])
                start.z = max(0.20, float(track["size_z"]) + 0.20)
                end = Point()
                end.x = start.x + float(track["vx"]) * 0.8
                end.y = start.y + float(track["vy"]) * 0.8
                end.z = start.z
                arrow.points = [start, end]
                arrow.lifetime = lifetime
                marker_array.markers.append(arrow)

            if self.show_labels:
                text = Marker()
                text.header.stamp = stamp
                text.header.frame_id = frame_id
                text.ns = "dynamic_labels"
                text.id = marker_id
                marker_id += 1
                text.type = Marker.TEXT_VIEW_FACING
                text.action = Marker.ADD
                text.pose.position.x = float(track["x"])
                text.pose.position.y = float(track["y"])
                text.pose.position.z = max(0.50, float(track["size_z"]) + 0.50)
                text.pose.orientation.w = 1.0
                text.scale.z = 0.35
                text.color.r = 1.0
                text.color.g = 1.0
                text.color.b = 1.0
                text.color.a = 0.95
                text.text = "{} #{:d}".format(label, int(entry["track_id"]))
                text.lifetime = lifetime
                marker_array.markers.append(text)

        return marker_array

    def cloud_callback(self, msg):
        stamp = msg.header.stamp if msg.header.stamp.to_sec() > 0.0 else rospy.Time.now()
        now_sec = stamp.to_sec() if stamp.to_sec() > 0.0 else rospy.Time.now().to_sec()
        frame_id = self._processing_frame_id(msg)

        if not self.have_odom and self.require_odom_for_processing:
            rospy.logwarn_throttle(
                2.0,
                "lidar_obstacle_perception_experimental: waiting for odom on %s",
                self.odom_topic,
            )
            return
        if not self.have_odom:
            rospy.logwarn_throttle(
                5.0,
                "lidar_obstacle_perception_experimental: odom missing on %s, falling back to local frame %s",
                self.odom_topic,
                frame_id,
            )

        local_points = self._extract_points(msg)
        if local_points.shape[0] == 0:
            empty = self._empty_points()
            static_points = self._confirmed_static_points() if self.have_odom else empty
            center_xy = self.odom_pos[:2] if self.have_odom else (0.0, 0.0)
            self.pub_ground.publish(self._points_to_cloud([], stamp, frame_id))
            self.pub_non_ground.publish(self._points_to_cloud([], stamp, frame_id))
            self.pub_dynamic.publish(self._points_to_cloud([], stamp, frame_id))
            self.pub_static.publish(
                self._points_to_cloud(static_points, stamp, frame_id)
            )
            self.pub_dynamic_markers.publish(
                self._build_dynamic_markers_for_frame([], stamp, frame_id)
            )
            self.pub_local_grid.publish(
                self._build_local_grid(static_points, empty, stamp, frame_id, center_xy)
            )
            return

        if self.input_cloud_is_non_ground:
            ground_local = self._empty_points()
            non_ground_local = local_points
        else:
            leveled_points = self._level_points(local_points)
            ground_mask = self._segment_ground(leveled_points)
            non_ground_mask = np.logical_not(ground_mask)
            ground_local = local_points[ground_mask]
            non_ground_local = local_points[non_ground_mask]
        if self.have_odom:
            ground_points = self._transform_points_to_map(ground_local)
            non_ground_points = self._transform_points_to_map(non_ground_local)
            center_xy = self.odom_pos[:2]
        else:
            ground_points = ground_local.copy()
            non_ground_points = non_ground_local.copy()
            center_xy = (0.0, 0.0)

        if self.ground_only_mode:
            self.pub_ground.publish(self._points_to_cloud(ground_points, stamp, frame_id))
            self.pub_non_ground.publish(
                self._points_to_cloud(non_ground_points, stamp, frame_id)
            )
            return

        clusters = self._cluster_points(non_ground_points)
        cluster_to_track = self._associate_tracks(clusters, now_sec)

        dynamic_entries = []
        dynamic_point_indices = []
        aggressive_ego_motion = self._ego_motion_is_aggressive()
        for ci, tid in cluster_to_track.items():
            track = self.tracks.get(tid)
            if track is None:
                continue
            cluster = clusters[ci]
            cluster_points = non_ground_points[cluster["indices"]]
            if (
                not aggressive_ego_motion
                and self._cluster_is_plausibly_dynamic(cluster)
                and not self._cluster_has_static_support(cluster_points)
            ):
                track["motion_hits"] = min(
                    int(track.get("motion_hits", 0)) + 1,
                    self.dynamic_min_motion_hits + 5,
                )
            else:
                track["motion_hits"] = max(int(track.get("motion_hits", 0)) - 1, 0)

            if not self._track_is_dynamic(track):
                continue
            dynamic_entries.append(
                {
                    "track_id": tid,
                    "track": track,
                    "label": self._dynamic_label(track),
                }
            )
            dynamic_point_indices.extend(clusters[ci]["indices"].tolist())

        dynamic_point_indices = np.asarray(dynamic_point_indices, dtype=np.int32)
        dynamic_points = (
            non_ground_points[dynamic_point_indices]
            if dynamic_point_indices.size > 0
            else self._empty_points()
        )

        if dynamic_point_indices.size > 0:
            static_mask = np.ones((non_ground_points.shape[0],), dtype=bool)
            static_mask[dynamic_point_indices] = False
            static_points_current = non_ground_points[static_mask]
        else:
            static_points_current = non_ground_points

        if dynamic_entries:
            filtered_static_points = []
            for point in static_points_current:
                keep = True
                for entry in dynamic_entries:
                    track = entry["track"]
                    if (
                        math.hypot(
                            float(point[0]) - float(track["x"]),
                            float(point[1]) - float(track["y"]),
                        )
                        <= self.dynamic_exclusion_radius_m
                    ):
                        keep = False
                        break
                if keep:
                    filtered_static_points.append(point)
            if filtered_static_points:
                static_points_current = np.asarray(filtered_static_points, dtype=np.float32)
            else:
                static_points_current = self._empty_points()

        if self.have_odom and not aggressive_ego_motion:
            self._update_static_voxels(static_points_current, now_sec)
            static_points_accumulated = self._confirmed_static_points()
        elif self.have_odom:
            static_points_accumulated = self._confirmed_static_points()
        else:
            static_points_accumulated = static_points_current
        static_points = (
            static_points_current
            if self.publish_static_current_frame_only
            else static_points_accumulated
        )
        local_grid = self._build_local_grid(
            static_points, dynamic_points, stamp, frame_id, center_xy
        )
        markers = self._build_dynamic_markers_for_frame(dynamic_entries, stamp, frame_id)

        self.pub_ground.publish(self._points_to_cloud(ground_points, stamp, frame_id))
        self.pub_non_ground.publish(
            self._points_to_cloud(non_ground_points, stamp, frame_id)
        )
        self.pub_static.publish(
            self._points_to_cloud(static_points, stamp, frame_id)
        )
        self.pub_dynamic.publish(
            self._points_to_cloud(dynamic_points, stamp, frame_id)
        )
        self.pub_dynamic_markers.publish(markers)
        self.pub_local_grid.publish(local_grid)


def main():
    rospy.init_node("lidar_obstacle_perception_experimental", anonymous=False)
    LidarObstaclePerceptionExperimental()
    rospy.spin()


if __name__ == "__main__":
    main()
