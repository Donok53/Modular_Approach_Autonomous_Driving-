#!/usr/bin/env python3
import math

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header


class TebObstacleCloudFilter:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/ouster/points")
        self.output_topic = rospy.get_param("~output_topic", "/move_base/filtered_obstacles")
        self.odom_topic = str(
            rospy.get_param("~odom_topic", "/lio_localizer/odometry/planning")
        ).strip()
        self.min_z = float(rospy.get_param("~min_z", -0.05))
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
            0.0, float(rospy.get_param("~lidar_height_m", 0.525))
        )
        self.ground_reject_min_m = float(
            rospy.get_param("~ground_reject_min_m", -0.20)
        )
        self.ground_reject_max_m = float(
            rospy.get_param("~ground_reject_max_m", 0.04)
        )
        self.min_range = max(0.0, float(rospy.get_param("~min_range", 0.10)))
        self.max_range = max(self.min_range + 0.1, float(rospy.get_param("~max_range", 4.5)))
        self.self_filter_radius_x = max(0.0, float(rospy.get_param("~self_filter_radius_x", 0.45)))
        self.self_filter_radius_y = max(0.0, float(rospy.get_param("~self_filter_radius_y", 0.40)))
        self.self_filter_padding_m = max(
            0.0, float(rospy.get_param("~self_filter_padding_m", 0.05))
        )
        self.rear_limit_x = float(rospy.get_param("~rear_limit_x", -0.10))
        self.lateral_limit_y = max(0.2, float(rospy.get_param("~lateral_limit_y", 2.0)))
        self.voxel_size = max(0.01, float(rospy.get_param("~voxel_size", 0.15)))
        self.have_level_pose = False
        self.level_roll = 0.0
        self.level_pitch = 0.0

        self.pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=1)
        self.sub = rospy.Subscriber(self.input_topic, PointCloud2, self.cloud_callback, queue_size=1, buff_size=2**24)
        self.sub_odom = None
        if self.enable_slope_compensation and self.odom_topic:
            self.sub_odom = rospy.Subscriber(
                self.odom_topic, Odometry, self.odom_callback, queue_size=20
            )

        rospy.loginfo(
            "teb_obstacle_cloud_filter started | in=%s out=%s odom=%s slope_comp=%s max_tilt=%.1fdeg ground_band=%s lidar_h=%.2fm ground=[%.2f, %.2f] z=[%.2f, %.2f] self=%.2fx%.2fm pad=%.2fm rear>=%.2fm lateral<=%.2fm voxel=%.2fm range=%.1f..%.1fm",
            self.input_topic,
            self.output_topic,
            self.odom_topic if self.odom_topic else "-",
            "on" if self.enable_slope_compensation else "off",
            math.degrees(self.slope_compensation_max_abs_rad),
            "on" if self.enable_ground_band_rejection else "off",
            self.lidar_height_m,
            self.ground_reject_min_m,
            self.ground_reject_max_m,
            self.min_z,
            self.max_z,
            2.0 * self.self_filter_radius_x,
            2.0 * self.self_filter_radius_y,
            self.self_filter_padding_m,
            self.rear_limit_x,
            self.lateral_limit_y,
            self.voxel_size,
            self.min_range,
            self.max_range,
        )

    @staticmethod
    def _quat_to_roll_pitch(q):
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)
        return roll, pitch

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        roll, pitch = self._quat_to_roll_pitch(q)
        max_abs = self.slope_compensation_max_abs_rad
        if max_abs > 0.0:
            roll = max(-max_abs, min(max_abs, roll))
            pitch = max(-max_abs, min(max_abs, pitch))
        self.level_roll = float(roll)
        self.level_pitch = float(pitch)
        self.have_level_pose = True

    def _leveled_z(self, x, y, z):
        if (not self.enable_slope_compensation) or (not self.have_level_pose):
            return z

        cr = math.cos(self.level_roll)
        sr = math.sin(self.level_roll)
        cp = math.cos(self.level_pitch)
        sp = math.sin(self.level_pitch)

        y1 = cr * y - sr * z
        z1 = sr * y + cr * z
        return (-sp * x) + (cp * z1)

    def _ground_relative_height(self, x, y, z):
        return self._leveled_z(x, y, z) + self.lidar_height_m

    def _safe_publish(self, msg):
        if rospy.is_shutdown():
            return
        try:
            self.pub.publish(msg)
        except rospy.ROSException:
            pass

    def cloud_callback(self, msg):
        points = []
        occupied = set()
        min_range_sq = self.min_range * self.min_range
        max_range_sq = self.max_range * self.max_range
        voxel = self.voxel_size

        raw_count = 0
        kept_count = 0

        for x, y, z in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            raw_count += 1

            x = float(x)
            y = float(y)
            z = float(z)
            z_eval = self._leveled_z(x, y, z)
            if self.enable_ground_band_rejection:
                ground_h = self._ground_relative_height(x, y, z)
                if self.ground_reject_min_m <= ground_h <= self.ground_reject_max_m:
                    continue
            if z_eval < self.min_z or z_eval > self.max_z:
                continue

            if x < self.rear_limit_x or abs(y) > self.lateral_limit_y:
                continue

            dist_sq = x * x + y * y + z * z
            if dist_sq < min_range_sq or dist_sq > max_range_sq:
                continue

            if (
                abs(x) <= (self.self_filter_radius_x + self.self_filter_padding_m)
                and abs(y) <= (self.self_filter_radius_y + self.self_filter_padding_m)
            ):
                continue

            key = (
                int(math.floor(x / voxel)),
                int(math.floor(y / voxel)),
                int(math.floor(z / voxel)),
            )
            if key in occupied:
                continue
            occupied.add(key)
            points.append((x, y, z))
            kept_count += 1

        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = msg.header.frame_id
        cloud = pc2.create_cloud_xyz32(header, points)
        self._safe_publish(cloud)

        rospy.loginfo_throttle(
            1.0,
            "teb_obstacle_cloud_filter: raw=%d kept=%d frame=%s",
            raw_count,
            kept_count,
            msg.header.frame_id or "<empty>",
        )


if __name__ == "__main__":
    rospy.init_node("teb_obstacle_cloud_filter")
    TebObstacleCloudFilter()
    rospy.spin()
