#!/usr/bin/env python3

import math

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header


class NovelObstacleCloudFilter:
    def __init__(self):
        self.input_cloud_topic = str(
            rospy.get_param("~input_cloud_topic", "/lio_localizer/localization/cloud_deskewed")
        )
        self.global_map_topic = str(
            rospy.get_param("~global_map_topic", "/lio_localizer/localization/global_map")
        )
        self.pose_topic = str(
            rospy.get_param("~pose_topic", "/lio_localizer/odometry/optimization")
        )
        self.output_cloud_topic = str(
            rospy.get_param("~output_cloud_topic", "/perception/novel_obstacle_cloud")
        )
        self.output_map_cloud_topic = str(
            rospy.get_param("~output_map_cloud_topic", "/perception/novel_obstacle_cloud_map")
        )

        self.min_z = float(rospy.get_param("~min_z", -0.35))
        self.max_z = float(rospy.get_param("~max_z", 2.20))
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
        self.max_range_m = max(0.1, float(rospy.get_param("~max_range_m", 20.0)))
        self.input_downsample = max(1, int(rospy.get_param("~input_downsample", 2)))
        self.map_downsample = max(1, int(rospy.get_param("~map_downsample", 1)))
        self.map_voxel_size_m = max(0.05, float(rospy.get_param("~map_voxel_size_m", 0.15)))
        self.match_xy_radius_m = max(
            0.0, float(rospy.get_param("~match_xy_radius_m", 0.30))
        )
        self.match_z_radius_m = max(
            0.0, float(rospy.get_param("~match_z_radius_m", 0.45))
        )
        self.cluster_cell_size_m = max(
            0.05, float(rospy.get_param("~cluster_cell_size_m", 0.20))
        )
        self.cluster_min_support_points = max(
            1, int(rospy.get_param("~cluster_min_support_points", 3))
        )
        self.cluster_z_layer_size_m = max(
            0.03, float(rospy.get_param("~cluster_z_layer_size_m", 0.18))
        )
        self.cluster_min_z_layers = max(
            1, int(rospy.get_param("~cluster_min_z_layers", 2))
        )
        self.cluster_min_height_m = max(
            0.0, float(rospy.get_param("~cluster_min_height_m", 0.12))
        )
        self.vertical_support_min_span_m = max(
            0.0, float(rospy.get_param("~vertical_support_min_span_m", 0.45))
        )
        self.reject_wide_low_clusters = bool(
            rospy.get_param("~reject_wide_low_clusters", True)
        )
        self.wide_low_min_span_m = max(
            0.0, float(rospy.get_param("~wide_low_min_span_m", 0.90))
        )
        self.wide_low_min_area_m2 = max(
            0.0, float(rospy.get_param("~wide_low_min_area_m2", 0.45))
        )
        self.wide_low_max_height_m = max(
            0.0, float(rospy.get_param("~wide_low_max_height_m", 0.12))
        )
        self.output_voxel_size_m = max(
            0.0, float(rospy.get_param("~output_voxel_size_m", 0.08))
        )
        self.blind_zone_radius_m = max(
            0.0, float(rospy.get_param("~blind_zone_radius_m", 0.0))
        )
        self.debug_log_period_s = max(
            0.0, float(rospy.get_param("~debug_log_period_s", 1.0))
        )

        self.have_pose = False
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_z = 0.0
        self.qx = 0.0
        self.qy = 0.0
        self.qz = 0.0
        self.qw = 1.0
        self.odom_roll = 0.0
        self.odom_pitch = 0.0
        self.odom_yaw = 0.0

        self.have_map = False
        self.global_map_frame_id = "map"
        self.map_voxels = set()
        self.match_xy_cells = max(
            0, int(math.ceil(self.match_xy_radius_m / self.map_voxel_size_m))
        )
        self.match_z_cells = max(
            0, int(math.ceil(self.match_z_radius_m / self.map_voxel_size_m))
        )
        self.range_sq = self.max_range_m * self.max_range_m
        self.blind_zone_radius_sq = self.blind_zone_radius_m * self.blind_zone_radius_m

        self.pub_cloud = rospy.Publisher(
            self.output_cloud_topic, PointCloud2, queue_size=1
        )
        self.pub_cloud_map = rospy.Publisher(
            self.output_map_cloud_topic, PointCloud2, queue_size=1
        )

        self.sub_pose = rospy.Subscriber(
            self.pose_topic, Odometry, self.pose_callback, queue_size=20
        )
        self.sub_map = rospy.Subscriber(
            self.global_map_topic, PointCloud2, self.global_map_callback, queue_size=1
        )
        self.sub_cloud = rospy.Subscriber(
            self.input_cloud_topic, PointCloud2, self.cloud_callback, queue_size=1
        )

        rospy.loginfo(
            "novel_obstacle_cloud_filter started | in=%s map=%s pose=%s out=%s out_map=%s "
            "z=[%.2f, %.2f] slope_comp=%s max_tilt=%.1fdeg ground_band=%s lidar_h=%.2fm "
            "ground=[%.2f, %.2f] range=%.1fm downsample=%d map_voxel=%.2fm match_xy=%.2fm "
            "match_z=%.2fm cluster_cell=%.2fm support>=%d z_layers>=%d layer=%.2fm min_h=%.2fm "
            "wide_low=%s span>=%.2fm area>=%.2fm2 h<=%.2fm output_voxel=%.2fm",
            self.input_cloud_topic,
            self.global_map_topic,
            self.pose_topic,
            self.output_cloud_topic,
            self.output_map_cloud_topic,
            self.min_z,
            self.max_z,
            "on" if self.enable_slope_compensation else "off",
            math.degrees(self.slope_compensation_max_abs_rad),
            "on" if self.enable_ground_band_rejection else "off",
            self.lidar_height_m,
            self.ground_reject_min_m,
            self.ground_reject_max_m,
            self.max_range_m,
            self.input_downsample,
            self.map_voxel_size_m,
            self.match_xy_radius_m,
            self.match_z_radius_m,
            self.cluster_cell_size_m,
            self.cluster_min_support_points,
            self.cluster_min_z_layers,
            self.cluster_z_layer_size_m,
            self.cluster_min_height_m,
            "on" if self.reject_wide_low_clusters else "off",
            self.wide_low_min_span_m,
            self.wide_low_min_area_m2,
            self.wide_low_max_height_m,
            self.output_voxel_size_m,
        )

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

    def pose_callback(self, msg):
        self.odom_x = float(msg.pose.pose.position.x)
        self.odom_y = float(msg.pose.pose.position.y)
        self.odom_z = float(msg.pose.pose.position.z)
        self.qx = float(msg.pose.pose.orientation.x)
        self.qy = float(msg.pose.pose.orientation.y)
        self.qz = float(msg.pose.pose.orientation.z)
        self.qw = float(msg.pose.pose.orientation.w)
        roll, pitch, yaw = self._quat_to_roll_pitch_yaw(msg.pose.pose.orientation)
        max_abs = self.slope_compensation_max_abs_rad
        if max_abs > 0.0:
            roll = max(-max_abs, min(max_abs, roll))
            pitch = max(-max_abs, min(max_abs, pitch))
        self.odom_roll = float(roll)
        self.odom_pitch = float(pitch)
        self.odom_yaw = float(yaw)
        self.have_pose = True

    def _voxel_key(self, x, y, z, voxel_size_m):
        return (
            int(math.floor(float(x) / voxel_size_m)),
            int(math.floor(float(y) / voxel_size_m)),
            int(math.floor(float(z) / voxel_size_m)),
        )

    def _local_to_map_xyz(self, x, y, z):
        qx = self.qx
        qy = self.qy
        qz = self.qz
        qw = self.qw

        r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
        r01 = 2.0 * (qx * qy - qz * qw)
        r02 = 2.0 * (qx * qz + qy * qw)
        r10 = 2.0 * (qx * qy + qz * qw)
        r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
        r12 = 2.0 * (qy * qz - qx * qw)
        r20 = 2.0 * (qx * qz - qy * qw)
        r21 = 2.0 * (qy * qz + qx * qw)
        r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

        mx = self.odom_x + r00 * x + r01 * y + r02 * z
        my = self.odom_y + r10 * x + r11 * y + r12 * z
        mz = self.odom_z + r20 * x + r21 * y + r22 * z
        return mx, my, mz

    def _leveled_z(self, x, y, z):
        if (not self.enable_slope_compensation) or (not self.have_pose):
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

    def global_map_callback(self, msg):
        voxels = set()
        total_points = 0
        kept_points = 0
        for i, p in enumerate(
            point_cloud2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True
            )
        ):
            total_points += 1
            if self.map_downsample > 1 and (i % self.map_downsample != 0):
                continue
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            if z < self.min_z or z > self.max_z:
                continue
            voxels.add(self._voxel_key(x, y, z, self.map_voxel_size_m))
            kept_points += 1

        self.map_voxels = voxels
        self.have_map = bool(voxels)
        self.global_map_frame_id = str(msg.header.frame_id or "map")

        rospy.loginfo(
            "novel_obstacle_cloud_filter: global map ready | frame=%s raw_pts=%d kept_pts=%d voxels=%d",
            self.global_map_frame_id,
            total_points,
            kept_points,
            len(self.map_voxels),
        )

    def _has_map_support(self, mx, my, mz):
        if not self.map_voxels:
            return False
        kx, ky, kz = self._voxel_key(mx, my, mz, self.map_voxel_size_m)
        for dx in range(-self.match_xy_cells, self.match_xy_cells + 1):
            for dy in range(-self.match_xy_cells, self.match_xy_cells + 1):
                for dz in range(-self.match_z_cells, self.match_z_cells + 1):
                    if (kx + dx, ky + dy, kz + dz) in self.map_voxels:
                        return True
        return False

    def _support_filter_pairs(self, pairs):
        if (not pairs) or self.cluster_min_support_points <= 1:
            return list(pairs)

        cell_counts = {}
        point_cells = []
        res = self.cluster_cell_size_m
        for local_pt, map_pt in pairs:
            cell = (
                int(math.floor(float(local_pt[0]) / res)),
                int(math.floor(float(local_pt[1]) / res)),
            )
            cell_counts[cell] = cell_counts.get(cell, 0) + 1
            point_cells.append((cell, local_pt, map_pt))

        kept = []
        for cell, local_pt, map_pt in point_cells:
            support = 0
            cx, cy = cell
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    support += cell_counts.get((cx + dx, cy + dy), 0)
            if support >= self.cluster_min_support_points:
                kept.append((local_pt, map_pt))
        return kept

    def _component_filter_pairs(self, pairs):
        if not pairs:
            return [], {
                "components": 0,
                "kept_components": 0,
                "rejected_wide_low": 0,
                "rejected_flat": 0,
            }

        res = self.cluster_cell_size_m
        cell_pairs = {}
        for local_pt, map_pt in pairs:
            cell = (
                int(math.floor(float(local_pt[0]) / res)),
                int(math.floor(float(local_pt[1]) / res)),
            )
            cell_pairs.setdefault(cell, []).append((local_pt, map_pt))

        kept = []
        visited = set()
        rejected_wide_low = 0
        rejected_flat = 0
        components = 0
        kept_components = 0

        for start_cell in cell_pairs.keys():
            if start_cell in visited:
                continue
            components += 1
            stack = [start_cell]
            visited.add(start_cell)
            component_pairs = []
            min_x = float("inf")
            max_x = float("-inf")
            min_y = float("inf")
            max_y = float("-inf")
            min_z_eval = float("inf")
            max_z_eval = float("-inf")
            z_layers = set()

            while stack:
                cell = stack.pop()
                for local_pt, map_pt in cell_pairs.get(cell, []):
                    component_pairs.append((local_pt, map_pt))
                    lx = float(local_pt[0])
                    ly = float(local_pt[1])
                    lz = float(local_pt[2])
                    z_eval = self._leveled_z(lx, ly, lz)
                    min_x = min(min_x, lx)
                    max_x = max(max_x, lx)
                    min_y = min(min_y, ly)
                    max_y = max(max_y, ly)
                    min_z_eval = min(min_z_eval, z_eval)
                    max_z_eval = max(max_z_eval, z_eval)
                    z_layers.add(
                        int(math.floor(z_eval / self.cluster_z_layer_size_m))
                    )
                cx, cy = cell
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nbr = (cx + dx, cy + dy)
                        if nbr in cell_pairs and nbr not in visited:
                            visited.add(nbr)
                            stack.append(nbr)

            if not component_pairs:
                continue

            span_x = max(0.0, max_x - min_x)
            span_y = max(0.0, max_y - min_y)
            span_xy = max(span_x, span_y)
            area_xy = max(span_x, self.cluster_cell_size_m) * max(
                span_y, self.cluster_cell_size_m
            )
            height_m = max(0.0, max_z_eval - min_z_eval)
            z_layer_count = len(z_layers)
            has_vertical_support = (
                z_layer_count >= self.cluster_min_z_layers
                or height_m >= self.cluster_min_height_m
            )

            if (
                self.reject_wide_low_clusters
                and span_xy >= self.wide_low_min_span_m
                and area_xy >= self.wide_low_min_area_m2
                and height_m <= self.wide_low_max_height_m
                and z_layer_count < self.cluster_min_z_layers
            ):
                rejected_wide_low += 1
                continue

            if (not has_vertical_support) and span_xy >= self.vertical_support_min_span_m:
                rejected_flat += 1
                continue

            kept.extend(component_pairs)
            kept_components += 1

        return kept, {
            "components": components,
            "kept_components": kept_components,
            "rejected_wide_low": rejected_wide_low,
            "rejected_flat": rejected_flat,
        }

    def _voxelize_pairs(self, pairs):
        if (not pairs) or self.output_voxel_size_m <= 0.0:
            return list(pairs)

        kept = []
        seen = set()
        for local_pt, map_pt in pairs:
            key = self._voxel_key(
                float(local_pt[0]),
                float(local_pt[1]),
                float(local_pt[2]),
                self.output_voxel_size_m,
            )
            if key in seen:
                continue
            seen.add(key)
            kept.append((local_pt, map_pt))
        return kept

    def _publish_empty(self, header):
        self.pub_cloud.publish(point_cloud2.create_cloud_xyz32(header, []))
        map_header = Header()
        map_header.stamp = header.stamp if header.stamp.to_sec() > 0.0 else rospy.Time.now()
        map_header.frame_id = self.global_map_frame_id
        self.pub_cloud_map.publish(point_cloud2.create_cloud_xyz32(map_header, []))

    def cloud_callback(self, msg):
        if not self.have_pose:
            rospy.logwarn_throttle(
                1.0, "novel_obstacle_cloud_filter: waiting for pose"
            )
            self._publish_empty(msg.header)
            return
        if not self.have_map:
            rospy.logwarn_throttle(
                1.0, "novel_obstacle_cloud_filter: waiting for global 3D map"
            )
            self._publish_empty(msg.header)
            return

        candidate_pairs = []
        in_points = 0
        z_filtered = 0
        ground_filtered = 0
        map_matched = 0
        for i, p in enumerate(
            point_cloud2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True
            )
        ):
            if self.input_downsample > 1 and (i % self.input_downsample != 0):
                continue
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            in_points += 1
            if self.enable_ground_band_rejection:
                ground_h = self._ground_relative_height(x, y, z)
                if self.ground_reject_min_m <= ground_h <= self.ground_reject_max_m:
                    ground_filtered += 1
                    continue

            z_eval = self._leveled_z(x, y, z)
            if z_eval < self.min_z or z_eval > self.max_z:
                continue
            z_filtered += 1
            if (x * x + y * y) > self.range_sq:
                continue
            if self.blind_zone_radius_sq > 0.0 and (x * x + y * y) < self.blind_zone_radius_sq:
                continue

            mx, my, mz = self._local_to_map_xyz(x, y, z)
            if self._has_map_support(mx, my, mz):
                map_matched += 1
                continue
            candidate_pairs.append(((x, y, z), (mx, my, mz)))

        supported_pairs = self._support_filter_pairs(candidate_pairs)
        validated_pairs, component_stats = self._component_filter_pairs(supported_pairs)
        output_pairs = self._voxelize_pairs(validated_pairs)

        local_points = [local_pt for local_pt, _ in output_pairs]
        map_points = [map_pt for _, map_pt in output_pairs]

        self.pub_cloud.publish(point_cloud2.create_cloud_xyz32(msg.header, local_points))

        map_header = Header()
        map_header.stamp = msg.header.stamp if msg.header.stamp.to_sec() > 0.0 else rospy.Time.now()
        map_header.frame_id = self.global_map_frame_id
        self.pub_cloud_map.publish(point_cloud2.create_cloud_xyz32(map_header, map_points))

        if self.debug_log_period_s > 0.0:
            rospy.loginfo_throttle(
                self.debug_log_period_s,
                "novel_obstacle_cloud_filter: in=%d ground_drop=%d z_ok=%d novel=%d support=%d comps=%d keep=%d reject_wide_low=%d reject_flat=%d out=%d map_match=%d",
                in_points,
                ground_filtered,
                z_filtered,
                len(candidate_pairs),
                len(supported_pairs),
                int(component_stats.get("components", 0)),
                int(component_stats.get("kept_components", 0)),
                int(component_stats.get("rejected_wide_low", 0)),
                int(component_stats.get("rejected_flat", 0)),
                len(output_pairs),
                map_matched,
            )


def main():
    rospy.init_node("novel_obstacle_cloud_filter", anonymous=False)
    NovelObstacleCloudFilter()
    rospy.spin()


if __name__ == "__main__":
    main()
