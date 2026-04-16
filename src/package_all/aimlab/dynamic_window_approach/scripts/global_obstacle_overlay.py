#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from collections import deque

import rospy
from dynamic_window_approach.msg import TrackedObjectArray
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray


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
        self.global_obstacle_overlay_boxes_topic = str(
            rospy.get_param(
                "~global_obstacle_overlay_boxes_topic", "/planning/global_obstacle_overlay_boxes"
            )
        ).strip()
        self.tracked_objects_topic = str(
            rospy.get_param("~tracked_objects_topic", "/perception/tracked_objects")
        ).strip()
        self.suppress_dynamic_tracked_boxes = bool(
            rospy.get_param("~suppress_dynamic_tracked_boxes", True)
        )
        self.promote_dynamic_tracked_boxes_to_global_overlay = bool(
            rospy.get_param("~promote_dynamic_tracked_boxes_to_global_overlay", False)
        )
        self.dynamic_tracked_overlay_confirm_immediately = bool(
            rospy.get_param("~dynamic_tracked_overlay_confirm_immediately", True)
        )
        self.dynamic_tracked_box_timeout_s = max(
            0.0, float(rospy.get_param("~dynamic_tracked_box_timeout_s", 1.0))
        )
        self.dynamic_tracked_speed_thresh_mps = max(
            0.0, float(rospy.get_param("~dynamic_tracked_speed_thresh_mps", 0.05))
        )
        self.dynamic_tracked_box_margin_m = max(
            0.0, float(rospy.get_param("~dynamic_tracked_box_margin_m", 0.15))
        )
        self.enable_travel_history = bool(rospy.get_param("~enable_travel_history", False))
        self.travel_history_topic = str(
            rospy.get_param("~travel_history_topic", "/planning/travel_history")
        ).strip()
        self.travel_history_path_topic = str(
            rospy.get_param("~travel_history_path_topic", "/planning/travel_history_path")
        ).strip()
        self.travel_history_max_points = max(
            2, int(rospy.get_param("~travel_history_max_points", 400))
        )
        self.travel_history_spacing_m = max(
            0.02, float(rospy.get_param("~travel_history_spacing_m", 0.05))
        )

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
            0.0, float(rospy.get_param("~lidar_height_m", 0.46))
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
        self.known_map_subtraction_enabled = bool(
            rospy.get_param("~known_map_subtraction_enabled", True)
        )
        self.known_map_subtraction_radius_m = max(
            0.0, float(rospy.get_param("~known_map_subtraction_radius_m", 0.30))
        )

        self.global_pointcloud_overlay_persistence_frames = max(
            1, int(rospy.get_param("~global_pointcloud_overlay_persistence_frames", 3))
        )
        self.global_pointcloud_overlay_static_lock_frames = max(
            self.global_pointcloud_overlay_persistence_frames,
            int(rospy.get_param("~global_pointcloud_overlay_static_lock_frames", 5)),
        )
        self.global_pointcloud_overlay_ttl_s = max(
            0.0, float(rospy.get_param("~global_pointcloud_overlay_ttl_s", 2.0))
        )
        self.global_pointcloud_overlay_static_lock_ttl_s = max(
            self.global_pointcloud_overlay_ttl_s,
            float(rospy.get_param("~global_pointcloud_overlay_static_lock_ttl_s", 30.0)),
        )
        self.global_pointcloud_overlay_merge_radius_m = max(
            0.05, float(rospy.get_param("~global_pointcloud_overlay_merge_radius_m", 0.25))
        )
        self.global_pointcloud_overlay_max_range_m = max(
            0.5, float(rospy.get_param("~global_pointcloud_overlay_max_range_m", 8.0))
        )
        self.global_pointcloud_overlay_static_lock_keep_range_m = max(
            self.global_pointcloud_overlay_max_range_m,
            float(rospy.get_param("~global_pointcloud_overlay_static_lock_keep_range_m", 15.0)),
        )
        self.global_pointcloud_overlay_static_box_margin_m = max(
            0.0, float(rospy.get_param("~global_pointcloud_overlay_static_box_margin_m", 0.08))
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
        self.global_pointcloud_overlay_far_field_candidate_blocking_enabled = bool(
            rospy.get_param("~global_pointcloud_overlay_far_field_candidate_blocking_enabled", True)
        )
        self.global_pointcloud_overlay_far_field_candidate_min_distance_m = max(
            0.0,
            float(
                rospy.get_param(
                    "~global_pointcloud_overlay_far_field_candidate_min_distance_m", 2.5
                )
            ),
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
        self.odom_roll = 0.0
        self.odom_pitch = 0.0
        self.global_path = None
        self.drivable_grid = None
        self.global_obstacle_overlay_memory = []
        self.global_obstacle_overlay_boxes_map = []
        self.travel_history_points = deque(maxlen=self.travel_history_max_points)
        self.tracked_objects = []
        self.tracked_objects_stamp_sec = 0.0

        self.pub_global_obstacle_overlay = rospy.Publisher(
            self.global_obstacle_overlay_topic, OccupancyGrid, queue_size=1
        )
        self.pub_global_obstacle_overlay_boxes = rospy.Publisher(
            self.global_obstacle_overlay_boxes_topic, MarkerArray, queue_size=1, latch=True
        )
        self.pub_travel_history = None
        self.pub_travel_history_path = None
        if self.enable_travel_history and self.travel_history_topic:
            self.pub_travel_history = rospy.Publisher(
                self.travel_history_topic, Marker, queue_size=2, latch=True
            )
        if self.enable_travel_history and self.travel_history_path_topic:
            self.pub_travel_history_path = rospy.Publisher(
                self.travel_history_path_topic, Path, queue_size=2, latch=True
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
        self.sub_tracked_objects = None
        if (
            (self.suppress_dynamic_tracked_boxes or self.promote_dynamic_tracked_boxes_to_global_overlay)
            and self.tracked_objects_topic
        ):
            self.sub_tracked_objects = rospy.Subscriber(
                self.tracked_objects_topic,
                TrackedObjectArray,
                self.tracked_objects_callback,
                queue_size=5,
            )

        rospy.loginfo(
            "global_obstacle_overlay started | cloud=%s global=%s grid=%s tracked=%s out=%s boxes=%s slope_comp=%s max_tilt=%.1fdeg ground_band=%s lidar_h=%.2fm ground=[%.2f, %.2f] persist=%d static_lock=%d ttl=%.1fs keep=%.1fm box_margin=%.2fm dyn_filter=%s dyn_promote=%s dyn_timeout=%.1fs blind_ttl=%.1fs blind_radius=%.2fm range=%.1fm lookahead=%.1fm corridor_margin=%.2fm far_field_relax=%s min_dist=%.2fm map_subtract=%s radius=%.2fm",
            self.obstacle_pointcloud_topic,
            self.global_path_topic,
            self.drivable_grid_topic,
            self.tracked_objects_topic if self.tracked_objects_topic else "-",
            self.global_obstacle_overlay_topic,
            self.global_obstacle_overlay_boxes_topic,
            "on" if self.enable_slope_compensation else "off",
            math.degrees(self.slope_compensation_max_abs_rad),
            "on" if self.enable_ground_band_rejection else "off",
            self.lidar_height_m,
            self.ground_reject_min_m,
            self.ground_reject_max_m,
            self.global_pointcloud_overlay_persistence_frames,
            self.global_pointcloud_overlay_static_lock_frames,
            self.global_pointcloud_overlay_static_lock_ttl_s,
            self.global_pointcloud_overlay_static_lock_keep_range_m,
            self.global_pointcloud_overlay_static_box_margin_m,
            "on" if self.suppress_dynamic_tracked_boxes else "off",
            "on" if self.promote_dynamic_tracked_boxes_to_global_overlay else "off",
            self.dynamic_tracked_box_timeout_s,
            self.global_pointcloud_overlay_blind_zone_hold_ttl_s,
            self.global_pointcloud_overlay_blind_zone_radius_m,
            self.global_pointcloud_overlay_max_range_m,
            self.global_pointcloud_overlay_lookahead_m,
            self.global_pointcloud_overlay_corridor_margin_m,
            "on" if self.global_pointcloud_overlay_far_field_candidate_blocking_enabled else "off",
            self.global_pointcloud_overlay_far_field_candidate_min_distance_m,
            "on" if self.known_map_subtraction_enabled else "off",
            self.known_map_subtraction_radius_m,
        )

    @staticmethod
    def _make_box(min_x, max_x, min_y, max_y):
        raw_min_x = float(min_x)
        raw_max_x = float(max_x)
        raw_min_y = float(min_y)
        raw_max_y = float(max_y)
        min_x = float(min(raw_min_x, raw_max_x))
        max_x = float(max(raw_min_x, raw_max_x))
        min_y = float(min(raw_min_y, raw_max_y))
        max_y = float(max(raw_min_y, raw_max_y))
        return {
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "x": 0.5 * (min_x + max_x),
            "y": 0.5 * (min_y + max_y),
            "size_x": max(0.0, max_x - min_x),
            "size_y": max(0.0, max_y - min_y),
        }

    @classmethod
    def _make_memory_entry(cls, box, seen_sec, hits=1, locked=False, lock_time=0.0):
        entry = cls._make_box(
            box["min_x"], box["max_x"], box["min_y"], box["max_y"]
        )
        entry.update(
            {
                "last_seen": float(seen_sec),
                "hits": int(max(1, hits)),
                "locked": bool(locked),
                "lock_time": float(lock_time),
                "dynamic": bool(box.get("dynamic", False)),
            }
        )
        return entry

    @classmethod
    def _expand_box(cls, box, margin_m):
        margin_m = max(0.0, float(margin_m))
        return cls._make_box(
            box["min_x"] - margin_m,
            box["max_x"] + margin_m,
            box["min_y"] - margin_m,
            box["max_y"] + margin_m,
        )

    @staticmethod
    def _box_radius_m(box):
        return 0.5 * math.hypot(float(box["size_x"]), float(box["size_y"]))

    @staticmethod
    def _boxes_match(box_a, box_b, margin_m):
        margin_m = max(0.0, float(margin_m))
        return not (
            float(box_a["max_x"]) < (float(box_b["min_x"]) - margin_m)
            or float(box_a["min_x"]) > (float(box_b["max_x"]) + margin_m)
            or float(box_a["max_y"]) < (float(box_b["min_y"]) - margin_m)
            or float(box_a["min_y"]) > (float(box_b["max_y"]) + margin_m)
        )

    @classmethod
    def _average_box_into_entry(cls, entry, box, prev_hits):
        denom = float(max(1, prev_hits) + 1)
        return cls._make_box(
            (float(entry["min_x"]) * prev_hits + float(box["min_x"])) / denom,
            (float(entry["max_x"]) * prev_hits + float(box["max_x"])) / denom,
            (float(entry["min_y"]) * prev_hits + float(box["min_y"])) / denom,
            (float(entry["max_y"]) * prev_hits + float(box["max_y"])) / denom,
        )

    @staticmethod
    def _box_center_distance_sq(box, wx, wy):
        dx = float(box["x"]) - float(wx)
        dy = float(box["y"]) - float(wy)
        return dx * dx + dy * dy

    @staticmethod
    def _grid_bounds_for_box(g, box):
        res = float(g.info.resolution)
        ox = float(g.info.origin.position.x)
        oy = float(g.info.origin.position.y)
        gx0 = int(math.floor((float(box["min_x"]) - ox) / res))
        gx1 = int(math.floor((float(box["max_x"]) - ox) / res))
        gy0 = int(math.floor((float(box["min_y"]) - oy) / res))
        gy1 = int(math.floor((float(box["max_y"]) - oy) / res))
        return gx0, gx1, gy0, gy1

    @staticmethod
    def _local_box_to_map_box(local_min_x, local_max_x, local_min_y, local_max_y, transform_fn):
        corners = (
            transform_fn(local_min_x, local_min_y),
            transform_fn(local_min_x, local_max_y),
            transform_fn(local_max_x, local_min_y),
            transform_fn(local_max_x, local_max_y),
        )
        xs = [pt[0] for pt in corners]
        ys = [pt[1] for pt in corners]
        return GlobalObstacleOverlayPublisher._make_box(min(xs), max(xs), min(ys), max(ys))

    @staticmethod
    def _box_marker(frame_id, stamp, marker_id, box, min_size_m):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = "global_obstacle_overlay_boxes"
        marker.id = int(marker_id)
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = float(box["x"])
        marker.pose.position.y = float(box["y"])
        marker.pose.position.z = 0.06
        marker.scale.x = max(float(min_size_m), float(box["size_x"]))
        marker.scale.y = max(float(min_size_m), float(box["size_y"]))
        marker.scale.z = 0.16
        if box.get("locked", False):
            marker.color.a = 0.60
            marker.color.r = 1.0
            marker.color.g = 0.55
            marker.color.b = 0.05
        else:
            marker.color.a = 0.38
            marker.color.r = 1.0
            marker.color.g = 0.22
            marker.color.b = 0.15
        return marker

    @staticmethod
    def _delete_all_boxes_marker(frame_id, stamp):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.action = Marker.DELETEALL
        return marker

    @staticmethod
    def _clone_box_entry(item):
        return {
            "min_x": float(item["min_x"]),
            "max_x": float(item["max_x"]),
            "min_y": float(item["min_y"]),
            "max_y": float(item["max_y"]),
            "last_seen": float(item["last_seen"]),
            "x": float(item["x"]),
            "y": float(item["y"]),
            "size_x": float(item["size_x"]),
            "size_y": float(item["size_y"]),
            "hits": int(item["hits"]),
            "locked": bool(item.get("locked", False)),
            "lock_time": float(item.get("lock_time", 0.0)),
            "dynamic": bool(item.get("dynamic", False)),
        }

    @staticmethod
    def _quat_to_roll_pitch_yaw(q):
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.odom_x = float(p.x)
        self.odom_y = float(p.y)
        roll, pitch, yaw = self._quat_to_roll_pitch_yaw(q)
        max_abs = self.slope_compensation_max_abs_rad
        if max_abs > 0.0:
            roll = max(-max_abs, min(max_abs, roll))
            pitch = max(-max_abs, min(max_abs, pitch))
        self.odom_roll = float(roll)
        self.odom_pitch = float(pitch)
        self.odom_yaw = float(yaw)
        self.have_odom = True
        self._record_travel_history_point(self.odom_x, self.odom_y)

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

    def tracked_objects_callback(self, msg):
        stamp_sec = msg.header.stamp.to_sec()
        if stamp_sec <= 0.0:
            stamp_sec = rospy.get_time()
        self.tracked_objects_stamp_sec = float(stamp_sec)
        self.tracked_objects = list(msg.objects)

    def global_path_callback(self, msg):
        self.global_path = msg

    def drivable_grid_callback(self, msg):
        self.drivable_grid = msg

    def cloud_callback(self, msg):
        if not self.have_odom:
            return

        cluster_counts = {}
        cluster_bounds = {}
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

                cell = self._pointcloud_cluster_cell(x, y)
                cluster_counts[cell] = cluster_counts.get(cell, 0) + 1
                min_x, max_x, min_y, max_y = cluster_bounds.get(cell, (x, x, y, y))
                cluster_bounds[cell] = (
                    min(min_x, x),
                    max(max_x, x),
                    min(min_y, y),
                    max(max_y, y),
                )

            accepted_cells = set()
            for (cx, cy), count in cluster_counts.items():
                support = 0
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        support += cluster_counts.get((cx + dx, cy + dy), 0)
                if support >= self.pointcloud_min_cluster_points:
                    accepted_cells.add((cx, cy))

            current_boxes_map = []
            visited = set()
            for start_cell in accepted_cells:
                if start_cell in visited:
                    continue
                stack = [start_cell]
                visited.add(start_cell)
                local_min_x = float("inf")
                local_max_x = float("-inf")
                local_min_y = float("inf")
                local_max_y = float("-inf")
                component_points = 0
                while stack:
                    cx, cy = stack.pop()
                    count = int(cluster_counts.get((cx, cy), 0))
                    if count <= 0:
                        continue
                    component_points += count
                    cell_min_x, cell_max_x, cell_min_y, cell_max_y = cluster_bounds[(cx, cy)]
                    local_min_x = min(local_min_x, cell_min_x)
                    local_max_x = max(local_max_x, cell_max_x)
                    local_min_y = min(local_min_y, cell_min_y)
                    local_max_y = max(local_max_y, cell_max_y)
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            nbr = (cx + dx, cy + dy)
                            if nbr in accepted_cells and nbr not in visited:
                                visited.add(nbr)
                                stack.append(nbr)
                if component_points <= 0 or not math.isfinite(local_min_x):
                    continue
                box = self._local_box_to_map_box(
                    local_min_x,
                    local_max_x,
                    local_min_y,
                    local_max_y,
                    self._local_to_map,
                )
                current_boxes_map.append(
                    self._expand_box(box, self.global_pointcloud_overlay_static_box_margin_m)
                )

            stamp_sec = msg.header.stamp.to_sec()
            if stamp_sec <= 0.0:
                stamp_sec = rospy.Time.now().to_sec()

            candidates = self._select_global_overlay_candidate_boxes(current_boxes_map)
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
    def _tracked_object_speed_mps(obj):
        return math.hypot(float(obj.twist.linear.x), float(obj.twist.linear.y))

    @staticmethod
    def _is_recent_dynamic_label(label):
        return (label or "").lower().startswith("recent_")

    def _fresh_dynamic_tracked_boxes(self):
        if (
            (not self.suppress_dynamic_tracked_boxes)
            or (not self.tracked_objects_topic)
            or self.tracked_objects_stamp_sec <= 0.0
            or self.dynamic_tracked_box_timeout_s <= 0.0
        ):
            return []
        if (rospy.get_time() - self.tracked_objects_stamp_sec) > self.dynamic_tracked_box_timeout_s:
            return []

        boxes = []
        for obj in self.tracked_objects:
            label = str(getattr(obj, "label", "") or "")
            speed_mps = self._tracked_object_speed_mps(obj)
            if (
                speed_mps < self.dynamic_tracked_speed_thresh_mps
                and (not self._is_recent_dynamic_label(label))
            ):
                continue
            half_x = max(0.10, 0.5 * abs(float(obj.size.x))) + self.dynamic_tracked_box_margin_m
            half_y = max(0.10, 0.5 * abs(float(obj.size.y))) + self.dynamic_tracked_box_margin_m
            boxes.append(
                self._make_box(
                    float(obj.pose.position.x) - half_x,
                    float(obj.pose.position.x) + half_x,
                    float(obj.pose.position.y) - half_y,
                    float(obj.pose.position.y) + half_y,
                )
            )
        return boxes

    def _box_overlaps_dynamic_object(self, box, dynamic_boxes):
        for dynamic_box in dynamic_boxes:
            if self._boxes_match(box, dynamic_box, 0.0):
                return True
        return False

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

    @staticmethod
    def _project_point_to_segment(px, py, x0, y0, x1, y1):
        vx = x1 - x0
        vy = y1 - y0
        seg_len_sq = vx * vx + vy * vy
        if seg_len_sq <= 1e-9:
            dx = px - x0
            dy = py - y0
            return x0, y0, 0.0, dx * dx + dy * dy

        t = ((px - x0) * vx + (py - y0) * vy) / seg_len_sq
        t = max(0.0, min(1.0, t))
        proj_x = x0 + t * vx
        proj_y = y0 + t * vy
        dx = px - proj_x
        dy = py - proj_y
        return proj_x, proj_y, t, dx * dx + dy * dy

    @staticmethod
    def _world_to_grid_cell(g, x, y):
        res = float(g.info.resolution)
        gx = int(math.floor((float(x) - float(g.info.origin.position.x)) / res))
        gy = int(math.floor((float(y) - float(g.info.origin.position.y)) / res))
        return gx, gy

    @staticmethod
    def _grid_cell_is_drivable_free(g, gx, gy):
        if gx < 0 or gy < 0 or gx >= int(g.info.width) or gy >= int(g.info.height):
            return False
        idx = gy * int(g.info.width) + gx
        return int(g.data[idx]) == 0

    def _world_cell_is_drivable_free(self, g, x, y):
        gx, gy = self._world_to_grid_cell(g, x, y)
        return self._grid_cell_is_drivable_free(g, gx, gy)

    def _box_overlaps_known_map_obstacle(self, box, margin_m=0.0):
        if (not self.known_map_subtraction_enabled) or self.drivable_grid is None:
            return False

        g = self.drivable_grid
        expanded = self._expand_box(box, max(margin_m, self.known_map_subtraction_radius_m))
        gx0, gx1, gy0, gy1 = self._grid_bounds_for_box(g, expanded)
        width = int(g.info.width)
        for gy in range(gy0, gy1 + 1):
            for gx in range(gx0, gx1 + 1):
                if gx < 0 or gy < 0 or gx >= width or gy >= int(g.info.height):
                    continue
                idx = gy * width + gx
                if int(g.data[idx]) == 0:
                    continue
                cx = float(g.info.origin.position.x) + (gx + 0.5) * float(g.info.resolution)
                cy = float(g.info.origin.position.y) + (gy + 0.5) * float(g.info.resolution)
                if (
                    float(expanded["min_x"]) <= cx <= float(expanded["max_x"])
                    and float(expanded["min_y"]) <= cy <= float(expanded["max_y"])
                ):
                    return True
        return False

    def _has_drivable_grid_line_of_sight(self, g, x0, y0, x1, y1):
        gx0, gy0 = self._world_to_grid_cell(g, x0, y0)
        gx1, gy1 = self._world_to_grid_cell(g, x1, y1)
        if not self._grid_cell_is_drivable_free(g, gx0, gy0):
            return False
        if not self._grid_cell_is_drivable_free(g, gx1, gy1):
            return False

        dx = abs(gx1 - gx0)
        dy = abs(gy1 - gy0)
        sx = 1 if gx0 < gx1 else -1
        sy = 1 if gy0 < gy1 else -1
        err = dx - dy
        cx = gx0
        cy = gy0
        while True:
            if not self._grid_cell_is_drivable_free(g, cx, cy):
                return False
            if cx == gx1 and cy == gy1:
                return True
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cx += sx
            if e2 < dx:
                err += dx
                cy += sy

    def _global_overlay_path_slice(self):
        pts = self._global_path_points()
        if len(pts) < 2:
            return []

        i0 = self._nearest_idx(pts, self.odom_x, self.odom_y)
        ig = self._accum_distance(pts, i0, self.global_pointcloud_overlay_lookahead_m)
        path_slice = pts[i0 : ig + 1]
        if len(path_slice) < 2:
            return []
        return path_slice

    def _box_is_valid_overlay_candidate(self, box, path_slice, corridor_half_width_m):
        if not path_slice:
            return False

        wx = float(box["x"])
        wy = float(box["y"])
        radius_m = self._box_radius_m(box)
        dx = wx - self.odom_x
        dy = wy - self.odom_y
        if (dx * dx + dy * dy) > (
            (self.global_pointcloud_overlay_max_range_m + radius_m) ** 2
        ):
            return False

        best_proj = None
        best_dist_sq = float("inf")
        corridor_limit_sq = (corridor_half_width_m + radius_m) ** 2
        for idx in range(len(path_slice) - 1):
            x0, y0 = path_slice[idx]
            x1, y1 = path_slice[idx + 1]
            proj_x, proj_y, _, dist_sq = self._project_point_to_segment(
                wx, wy, x0, y0, x1, y1
            )
            if dist_sq <= corridor_limit_sq and dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_proj = (proj_x, proj_y)

        if best_proj is None:
            return False
        if self.drivable_grid is None:
            return True
        far_field_relaxed = (
            self.global_pointcloud_overlay_far_field_candidate_blocking_enabled
            and math.hypot(dx, dy)
            >= self.global_pointcloud_overlay_far_field_candidate_min_distance_m
        )
        overlaps_known_map_obstacle = self._box_overlaps_known_map_obstacle(box)
        center_is_drivable_free = self._world_cell_is_drivable_free(self.drivable_grid, wx, wy)
        if (
            (not overlaps_known_map_obstacle)
            and center_is_drivable_free
            and self._has_drivable_grid_line_of_sight(
                self.drivable_grid, best_proj[0], best_proj[1], wx, wy
            )
        ):
            return True
        if not far_field_relaxed:
            return False
        return self._world_cell_is_drivable_free(
            self.drivable_grid, best_proj[0], best_proj[1]
        )

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

    def _select_global_overlay_candidate_boxes(self, current_boxes_map):
        if not current_boxes_map:
            return []

        path_slice = self._global_overlay_path_slice()
        if len(path_slice) < 2:
            return []

        corridor_half_width_m = self._pointcloud_corridor_half_width_m(
            self.global_pointcloud_overlay_corridor_margin_m
        )
        dynamic_boxes = self._fresh_dynamic_tracked_boxes()

        selected = []
        suppressed_dynamic = 0
        for box in current_boxes_map:
            if dynamic_boxes and self._box_overlaps_dynamic_object(box, dynamic_boxes):
                suppressed_dynamic += 1
                continue
            if self._box_is_valid_overlay_candidate(
                box, path_slice, corridor_half_width_m
            ):
                selected.append(box)
        promoted_dynamic = 0
        if self.promote_dynamic_tracked_boxes_to_global_overlay and dynamic_boxes:
            for dynamic_box in dynamic_boxes:
                candidate = self._expand_box(
                    dynamic_box, self.global_pointcloud_overlay_static_box_margin_m
                )
                candidate["dynamic"] = True
                if not self._box_is_valid_overlay_candidate(
                    candidate, path_slice, corridor_half_width_m
                ):
                    continue
                if any(
                    self._boxes_match(
                        existing,
                        candidate,
                        self.global_pointcloud_overlay_merge_radius_m,
                    )
                    for existing in selected
                ):
                    continue
                selected.append(candidate)
                promoted_dynamic += 1
        if suppressed_dynamic > 0:
            rospy.loginfo_throttle(
                1.0,
                "global_obstacle_overlay: suppressed %d box candidates that overlapped dynamic tracked objects",
                suppressed_dynamic,
            )
        if promoted_dynamic > 0:
            rospy.loginfo_throttle(
                1.0,
                "global_obstacle_overlay: promoted %d tracked dynamic boxes into global overlay candidates",
                promoted_dynamic,
            )
        return selected

    def _prune_global_obstacle_overlay_memory(self, now_sec):
        if self.global_pointcloud_overlay_ttl_s <= 0.0 or self.global_pointcloud_overlay_max_points <= 0:
            self.global_obstacle_overlay_memory = []
            self.global_obstacle_overlay_boxes_map = []
            return []

        max_range_sq = self.global_pointcloud_overlay_max_range_m * self.global_pointcloud_overlay_max_range_m
        static_keep_range_sq = (
            self.global_pointcloud_overlay_static_lock_keep_range_m
            * self.global_pointcloud_overlay_static_lock_keep_range_m
        )
        path_slice = self._global_overlay_path_slice()
        corridor_half_width_m = self._pointcloud_corridor_half_width_m(
            self.global_pointcloud_overlay_corridor_margin_m
        )
        dynamic_boxes = self._fresh_dynamic_tracked_boxes()
        kept = []
        for entry in self.global_obstacle_overlay_memory:
            wx = float(entry["x"])
            wy = float(entry["y"])
            seen_sec = float(entry["last_seen"])
            hits = int(entry["hits"])
            locked = bool(entry.get("locked", False))
            effective_ttl_s = self.global_pointcloud_overlay_ttl_s
            if self._in_global_overlay_blind_zone(wx, wy):
                effective_ttl_s = max(
                    effective_ttl_s,
                    self.global_pointcloud_overlay_blind_zone_hold_ttl_s,
                )
            if locked:
                effective_ttl_s = max(
                    effective_ttl_s,
                    self.global_pointcloud_overlay_static_lock_ttl_s,
                )
            if (now_sec - seen_sec) > effective_ttl_s:
                continue
            dx = wx - self.odom_x
            dy = wy - self.odom_y
            keep_range_sq = static_keep_range_sq if locked else max_range_sq
            if (dx * dx + dy * dy) > keep_range_sq:
                continue
            if dynamic_boxes and self._box_overlaps_dynamic_object(entry, dynamic_boxes):
                if not (
                    self.promote_dynamic_tracked_boxes_to_global_overlay
                    and bool(entry.get("dynamic", False))
                ):
                    continue
            if path_slice and (
                not self._box_is_valid_overlay_candidate(
                    entry, path_slice, corridor_half_width_m
                )
            ):
                continue
            if (not path_slice) and self._box_overlaps_known_map_obstacle(entry):
                continue
            kept.append(entry)

        kept.sort(
            key=lambda item: (
                1 if item.get("locked", False) else 0,
                int(item["hits"]),
                float(item["last_seen"]),
            ),
            reverse=True,
        )
        if len(kept) > self.global_pointcloud_overlay_max_points:
            kept = kept[: self.global_pointcloud_overlay_max_points]

        self.global_obstacle_overlay_memory = kept
        confirmed = [
            self._clone_box_entry(item)
            for item in kept
            if item.get("locked", False)
            or int(item["hits"]) >= self.global_pointcloud_overlay_persistence_frames
        ]
        self.global_obstacle_overlay_boxes_map = confirmed
        return confirmed

    def _update_global_obstacle_overlay_memory(self, candidates_map, now_sec):
        confirmed = self._prune_global_obstacle_overlay_memory(now_sec)
        if not candidates_map:
            return confirmed

        memory = list(self.global_obstacle_overlay_memory)
        for box in candidates_map:
            best_idx = None
            best_d2 = 1e18
            for idx, entry in enumerate(memory):
                if not self._boxes_match(entry, box, self.global_pointcloud_overlay_merge_radius_m):
                    continue
                d2 = self._box_center_distance_sq(entry, box["x"], box["y"])
                if d2 < best_d2:
                    best_d2 = d2
                    best_idx = idx
            if best_idx is None:
                initial_hits = 1
                if bool(box.get("dynamic", False)) and self.dynamic_tracked_overlay_confirm_immediately:
                    initial_hits = max(
                        initial_hits, self.global_pointcloud_overlay_persistence_frames
                    )
                memory.append(self._make_memory_entry(box, now_sec, hits=initial_hits))
            else:
                entry = dict(memory[best_idx])
                prev_hits = max(1, int(entry["hits"]))
                hits = min(
                    prev_hits + 1,
                    self.global_pointcloud_overlay_static_lock_frames + 8,
                )
                locked = bool(entry.get("locked", False))
                if not locked:
                    entry.update(self._average_box_into_entry(entry, box, prev_hits))
                    if hits >= self.global_pointcloud_overlay_static_lock_frames:
                        locked = True
                        entry["lock_time"] = float(now_sec)
                        rospy.loginfo(
                            "global_obstacle_overlay: locked static obstacle box at (%.2f, %.2f) size=%.2fx%.2fm after %d hits",
                            float(entry["x"]),
                            float(entry["y"]),
                            float(entry["size_x"]),
                            float(entry["size_y"]),
                            hits,
                        )
                entry["hits"] = hits
                entry["last_seen"] = float(now_sec)
                entry["locked"] = locked
                entry["dynamic"] = bool(entry.get("dynamic", False) or box.get("dynamic", False))
                memory[best_idx] = entry

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
        for box in self.global_obstacle_overlay_boxes_map:
            gx0, gx1, gy0, gy1 = self._grid_bounds_for_box(base, box)
            gx0 = max(0, gx0)
            gy0 = max(0, gy0)
            gx1 = min(w - 1, gx1)
            gy1 = min(h - 1, gy1)
            if gx0 > gx1 or gy0 > gy1:
                continue
            for gy in range(gy0, gy1 + 1):
                row_offset = gy * w
                for gx in range(gx0, gx1 + 1):
                    data[row_offset + gx] = 100
        out.data = data
        self.pub_global_obstacle_overlay.publish(out)
        self._publish_global_obstacle_overlay_boxes_marker(
            out.header.frame_id, out.header.stamp, base.info.resolution
        )

    def _publish_global_obstacle_overlay_boxes_marker(self, frame_id, stamp, min_size_m):
        marker_array = MarkerArray()
        marker_array.markers.append(self._delete_all_boxes_marker(frame_id, stamp))
        for idx, box in enumerate(self.global_obstacle_overlay_boxes_map):
            marker_array.markers.append(
                self._box_marker(frame_id, stamp, idx, box, min_size_m)
            )
        self.pub_global_obstacle_overlay_boxes.publish(marker_array)

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
            "global_obstacle_overlay: travel_history marker/path points=%d topic=%s path_topic=%s",
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
        if self.pub_travel_history is None:
            return

        x = float(x)
        y = float(y)
        if self.travel_history_points:
            last_x, last_y = self.travel_history_points[-1]
            if math.hypot(x - last_x, y - last_y) < self.travel_history_spacing_m:
                return
        self.travel_history_points.append((x, y))
        self._publish_travel_history_marker()
        self._publish_travel_history_path()


if __name__ == "__main__":
    rospy.init_node("global_obstacle_overlay", anonymous=False)
    GlobalObstacleOverlayPublisher()
    rospy.spin()
