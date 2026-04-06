#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
from collections import deque

import numpy as np
import rospy
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64, String


def clamp(value, low, high):
    return max(low, min(high, value))


class LidarHeightEstimator:
    def __init__(self):
        self.input_topic = str(rospy.get_param("~input_topic", "/ouster/points")).strip()
        self.process_every_n = max(1, int(rospy.get_param("~process_every_n", 3)))
        self.input_decimation = max(1, int(rospy.get_param("~input_decimation", 6)))
        self.min_range_m = max(0.0, float(rospy.get_param("~min_range_m", 0.6)))
        self.max_range_m = max(self.min_range_m + 0.5, float(rospy.get_param("~max_range_m", 12.0)))
        self.min_ground_z_m = float(rospy.get_param("~min_ground_z_m", -2.0))
        self.max_ground_z_m = float(rospy.get_param("~max_ground_z_m", -0.03))
        self.self_exclusion_half_length_m = max(
            0.0, float(rospy.get_param("~self_exclusion_half_length_m", 0.45))
        )
        self.self_exclusion_half_width_m = max(
            0.0, float(rospy.get_param("~self_exclusion_half_width_m", 0.40))
        )
        self.max_abs_x_m = max(0.0, float(rospy.get_param("~max_abs_x_m", 8.0)))
        self.max_abs_y_m = max(0.0, float(rospy.get_param("~max_abs_y_m", 8.0)))
        self.voxel_size_m = max(0.01, float(rospy.get_param("~voxel_size_m", 0.05)))
        self.ransac_iterations = max(20, int(rospy.get_param("~ransac_iterations", 180)))
        self.ransac_inlier_threshold_m = max(
            0.005, float(rospy.get_param("~ransac_inlier_threshold_m", 0.025))
        )
        self.max_plane_tilt_deg = max(
            0.5, float(rospy.get_param("~max_plane_tilt_deg", 8.0))
        )
        self.min_candidate_points = max(
            50, int(rospy.get_param("~min_candidate_points", 250))
        )
        self.min_inlier_points = max(30, int(rospy.get_param("~min_inlier_points", 120)))
        self.history_size = max(3, int(rospy.get_param("~history_size", 15)))
        self.stable_std_threshold_m = max(
            0.001, float(rospy.get_param("~stable_std_threshold_m", 0.01))
        )
        self.exit_when_stable = bool(rospy.get_param("~exit_when_stable", False))
        self.lowest_vertical_angle_deg = rospy.get_param(
            "~lowest_vertical_angle_deg", None
        )
        self.publish_ground_cloud = bool(rospy.get_param("~publish_ground_cloud", True))

        self.max_plane_tilt_rad = math.radians(self.max_plane_tilt_deg)
        self.height_history = deque(maxlen=self.history_size)
        self.cloud_counter = 0
        self.last_status_text = ""

        self.pub_height = rospy.Publisher("~height_m", Float64, queue_size=10)
        self.pub_height_median = rospy.Publisher("~height_median_m", Float64, queue_size=10)
        self.pub_height_std = rospy.Publisher("~height_std_m", Float64, queue_size=10)
        self.pub_observed_ground_start = rospy.Publisher(
            "~observed_ground_start_m", Float64, queue_size=10
        )
        self.pub_theoretical_ground_start = rospy.Publisher(
            "~theoretical_ground_start_m", Float64, queue_size=10
        )
        self.pub_status = rospy.Publisher("~status_text", String, queue_size=10)
        self.pub_ground_cloud = None
        if self.publish_ground_cloud:
            self.pub_ground_cloud = rospy.Publisher(
                "~ground_points", PointCloud2, queue_size=1
            )

        self.sub = rospy.Subscriber(
            self.input_topic,
            PointCloud2,
            self.cloud_callback,
            queue_size=1,
            buff_size=2**24,
        )

        rospy.loginfo(
            "lidar_height_estimator started | topic=%s every_n=%d decimation=%d range=%.2f..%.2fm z=%.2f..%.2fm self=%.2fx%.2fm voxel=%.3fm ransac=%d thresh=%.3fm tilt<=%.1fdeg history=%d",
            self.input_topic,
            self.process_every_n,
            self.input_decimation,
            self.min_range_m,
            self.max_range_m,
            self.min_ground_z_m,
            self.max_ground_z_m,
            2.0 * self.self_exclusion_half_length_m,
            2.0 * self.self_exclusion_half_width_m,
            self.voxel_size_m,
            self.ransac_iterations,
            self.ransac_inlier_threshold_m,
            self.max_plane_tilt_deg,
            self.history_size,
        )

    def _publish_status(self, text):
        if text != self.last_status_text:
            self.last_status_text = text
            self.pub_status.publish(String(data=text))

    def _extract_points(self, msg):
        points = []
        for idx, point in enumerate(
            pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        ):
            if idx % self.input_decimation != 0:
                continue

            x = float(point[0])
            y = float(point[1])
            z = float(point[2])

            if abs(x) > self.max_abs_x_m or abs(y) > self.max_abs_y_m:
                continue

            if (
                abs(x) <= self.self_exclusion_half_length_m
                and abs(y) <= self.self_exclusion_half_width_m
            ):
                continue

            if z < self.min_ground_z_m or z > self.max_ground_z_m:
                continue

            horiz_range = math.hypot(x, y)
            if horiz_range < self.min_range_m or horiz_range > self.max_range_m:
                continue

            points.append((x, y, z))

        if not points:
            return np.empty((0, 3), dtype=np.float64)
        return np.asarray(points, dtype=np.float64)

    def _voxel_downsample(self, points):
        if points.shape[0] == 0:
            return points
        voxel = self.voxel_size_m
        keys = np.floor(points / voxel).astype(np.int64)
        unique_keys, unique_indices = np.unique(keys, axis=0, return_index=True)
        del unique_keys
        return points[np.sort(unique_indices)]

    @staticmethod
    def _plane_from_three_points(p0, p1, p2):
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            return None
        normal = normal / norm
        if normal[2] < 0.0:
            normal = -normal
        d = -float(np.dot(normal, p0))
        if d < 0.0:
            normal = -normal
            d = -d
        return normal, d

    def _plane_is_ground_like(self, normal, d):
        if normal is None:
            return False
        tilt = math.acos(clamp(float(normal[2]), -1.0, 1.0))
        if tilt > self.max_plane_tilt_rad:
            return False
        return d > 0.0

    def _fit_plane_ransac(self, points):
        point_count = points.shape[0]
        if point_count < 3:
            return None

        best_mask = None
        best_normal = None
        best_d = None
        best_inliers = 0

        for _ in range(self.ransac_iterations):
            sample_idx = random.sample(range(point_count), 3)
            plane = self._plane_from_three_points(
                points[sample_idx[0]], points[sample_idx[1]], points[sample_idx[2]]
            )
            if plane is None:
                continue
            normal, d = plane
            if not self._plane_is_ground_like(normal, d):
                continue

            distances = np.abs(points.dot(normal) + d)
            mask = distances <= self.ransac_inlier_threshold_m
            inlier_count = int(np.count_nonzero(mask))
            if inlier_count > best_inliers:
                best_inliers = inlier_count
                best_mask = mask
                best_normal = normal
                best_d = d

        if best_mask is None or best_inliers < self.min_inlier_points:
            return None
        return best_normal, best_d, best_mask

    def _refit_plane(self, inliers):
        centroid = np.mean(inliers, axis=0)
        centered = inliers - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1, :]
        normal = normal / max(np.linalg.norm(normal), 1e-9)
        if normal[2] < 0.0:
            normal = -normal
        d = -float(np.dot(normal, centroid))
        if d < 0.0:
            normal = -normal
            d = -d
        return normal, d

    def _publish_ground_cloud(self, header, ground_points):
        if self.pub_ground_cloud is None or ground_points.shape[0] == 0:
            return
        msg = pc2.create_cloud_xyz32(header, ground_points.tolist())
        self.pub_ground_cloud.publish(msg)

    def _maybe_publish_theoretical_ground_start(self, height):
        if self.lowest_vertical_angle_deg is None:
            return
        try:
            angle_deg = float(self.lowest_vertical_angle_deg)
        except (TypeError, ValueError):
            return
        if angle_deg >= 0.0:
            return
        angle_rad = math.radians(abs(angle_deg))
        if angle_rad <= 1e-6:
            return
        distance = height / math.tan(angle_rad)
        self.pub_theoretical_ground_start.publish(Float64(data=float(distance)))

    def cloud_callback(self, msg):
        self.cloud_counter += 1
        if self.cloud_counter % self.process_every_n != 0:
            return

        points = self._extract_points(msg)
        if points.shape[0] < self.min_candidate_points:
            status = "too_few_candidates: {}".format(points.shape[0])
            self._publish_status(status)
            rospy.logwarn_throttle(
                2.0,
                "lidar_height_estimator: %s (need >= %d)",
                status,
                self.min_candidate_points,
            )
            return

        downsampled = self._voxel_downsample(points)
        plane = self._fit_plane_ransac(downsampled)
        if plane is None:
            status = "ground_plane_not_found"
            self._publish_status(status)
            rospy.logwarn_throttle(2.0, "lidar_height_estimator: %s", status)
            return

        _, _, inlier_mask = plane
        inliers = downsampled[inlier_mask]
        normal, d = self._refit_plane(inliers)
        if not self._plane_is_ground_like(normal, d):
            status = "refit_plane_invalid"
            self._publish_status(status)
            rospy.logwarn_throttle(2.0, "lidar_height_estimator: %s", status)
            return

        residuals = np.abs(inliers.dot(normal) + d)
        height_m = float(d)
        height_std_m = 0.0
        observed_ground_ranges = np.linalg.norm(inliers[:, :2], axis=1)
        observed_ground_start_m = float(np.percentile(observed_ground_ranges, 2.0))

        self.height_history.append(height_m)
        median_height_m = float(np.median(np.asarray(self.height_history)))
        if len(self.height_history) >= 2:
            height_std_m = float(np.std(np.asarray(self.height_history)))

        self.pub_height.publish(Float64(data=height_m))
        self.pub_height_median.publish(Float64(data=median_height_m))
        self.pub_height_std.publish(Float64(data=height_std_m))
        self.pub_observed_ground_start.publish(Float64(data=observed_ground_start_m))
        self._maybe_publish_theoretical_ground_start(median_height_m)
        self._publish_ground_cloud(msg.header, inliers)

        status = (
            "ok height={:.3f}m median={:.3f}m std={:.3f}m "
            "ground_start={:.3f}m inliers={} residual_med={:.3f}m tilt={:.2f}deg"
        ).format(
            height_m,
            median_height_m,
            height_std_m,
            observed_ground_start_m,
            int(inliers.shape[0]),
            float(np.median(residuals)),
            math.degrees(math.acos(clamp(float(normal[2]), -1.0, 1.0))),
        )
        self._publish_status(status)
        rospy.loginfo_throttle(1.0, "lidar_height_estimator: %s", status)

        if (
            self.exit_when_stable
            and len(self.height_history) >= self.history_size
            and height_std_m <= self.stable_std_threshold_m
        ):
            rospy.loginfo(
                "lidar_height_estimator: stable estimate reached (median=%.3fm std=%.3fm), shutting down",
                median_height_m,
                height_std_m,
            )
            rospy.signal_shutdown("stable height estimate reached")


def main():
    rospy.init_node("lidar_height_estimator", anonymous=False)
    random.seed(0)
    LidarHeightEstimator()
    rospy.spin()


if __name__ == "__main__":
    main()
