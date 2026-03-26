#!/usr/bin/env python3
import math

import rospy
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header


class TebObstacleCloudFilter:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/ouster/points")
        self.output_topic = rospy.get_param("~output_topic", "/move_base/filtered_obstacles")
        self.min_z = float(rospy.get_param("~min_z", -0.05))
        self.max_z = float(rospy.get_param("~max_z", 1.50))
        self.min_range = max(0.0, float(rospy.get_param("~min_range", 0.10)))
        self.max_range = max(self.min_range + 0.1, float(rospy.get_param("~max_range", 4.5)))
        self.self_filter_radius_x = max(0.0, float(rospy.get_param("~self_filter_radius_x", 0.45)))
        self.self_filter_radius_y = max(0.0, float(rospy.get_param("~self_filter_radius_y", 0.40)))
        self.rear_limit_x = float(rospy.get_param("~rear_limit_x", -0.10))
        self.lateral_limit_y = max(0.2, float(rospy.get_param("~lateral_limit_y", 2.0)))
        self.voxel_size = max(0.01, float(rospy.get_param("~voxel_size", 0.15)))

        self.pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=1)
        self.sub = rospy.Subscriber(self.input_topic, PointCloud2, self.cloud_callback, queue_size=1, buff_size=2**24)

        rospy.loginfo(
            "teb_obstacle_cloud_filter started | in=%s out=%s z=[%.2f, %.2f] self=%.2fx%.2fm rear>=%.2fm lateral<=%.2fm voxel=%.2fm range=%.1f..%.1fm",
            self.input_topic,
            self.output_topic,
            self.min_z,
            self.max_z,
            2.0 * self.self_filter_radius_x,
            2.0 * self.self_filter_radius_y,
            self.rear_limit_x,
            self.lateral_limit_y,
            self.voxel_size,
            self.min_range,
            self.max_range,
        )

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

            if z < self.min_z or z > self.max_z:
                continue

            if x < self.rear_limit_x or abs(y) > self.lateral_limit_y:
                continue

            dist_sq = x * x + y * y + z * z
            if dist_sq < min_range_sq or dist_sq > max_range_sq:
                continue

            if abs(x) <= self.self_filter_radius_x and abs(y) <= self.self_filter_radius_y:
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
        self.pub.publish(cloud)

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
