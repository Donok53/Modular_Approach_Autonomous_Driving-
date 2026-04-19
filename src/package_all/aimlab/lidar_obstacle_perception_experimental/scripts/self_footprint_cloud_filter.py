#!/usr/bin/env python3
import math

import rospy
from sensor_msgs.msg import Imu, PointCloud2
from sensor_msgs import point_cloud2
from tf.transformations import euler_from_quaternion


class SelfFootprintCloudFilter:
    def __init__(self):
        self.input_topic = rospy.get_param(
            "~input_topic", "/planning/linefit_ground/non_ground_cloud_raw"
        )
        self.output_topic = rospy.get_param(
            "~output_topic", "/planning/linefit_ground/non_ground_cloud"
        )
        self.base_frame = rospy.get_param("~base_frame", "base_link")

        robot_length_m = max(0.01, float(rospy.get_param("~robot_length_m", 0.612)))
        robot_width_m = max(0.01, float(rospy.get_param("~robot_width_m", 0.58)))
        padding_m = max(0.0, float(rospy.get_param("~padding_m", 0.04)))

        default_front = 0.5 * robot_length_m + padding_m
        default_rear = 0.5 * robot_length_m + padding_m
        default_half_width = 0.5 * robot_width_m + padding_m

        self.front_m = max(0.0, float(rospy.get_param("~front_m", default_front)))
        self.rear_m = max(0.0, float(rospy.get_param("~rear_m", default_rear)))
        self.half_width_m = max(
            0.0, float(rospy.get_param("~half_width_m", default_half_width))
        )
        self.center_x_m = float(rospy.get_param("~center_x_m", 0.0))
        self.center_y_m = float(rospy.get_param("~center_y_m", 0.0))
        self.min_z_m = float(rospy.get_param("~min_z_m", -10.0))
        self.max_z_m = float(rospy.get_param("~max_z_m", 10.0))
        self.drop_zero_points = bool(rospy.get_param("~drop_zero_points", True))

        self.enable_pitch_footprint_adjustment = bool(
            rospy.get_param("~enable_pitch_footprint_adjustment", False)
        )
        self.pitch_uphill_sign = 1.0
        if float(rospy.get_param("~pitch_uphill_sign", -1.0)) < 0.0:
            self.pitch_uphill_sign = -1.0
        self.pitch_adjust_deadband_deg = abs(
            float(rospy.get_param("~pitch_adjust_deadband_deg", 2.0))
        )
        self.pitch_adjust_primary_gain_m_per_deg = max(
            0.0, float(rospy.get_param("~pitch_adjust_primary_gain_m_per_deg", 0.02))
        )
        self.pitch_adjust_opposite_gain_m_per_deg = max(
            0.0, float(rospy.get_param("~pitch_adjust_opposite_gain_m_per_deg", 0.0))
        )
        self.pitch_adjust_max_extension_m = max(
            0.0, float(rospy.get_param("~pitch_adjust_max_extension_m", 0.25))
        )

        self.enable_slope_residual_filter = bool(
            rospy.get_param("~enable_slope_residual_filter", False)
        )
        self.use_imu_slope_gate = bool(rospy.get_param("~use_imu_slope_gate", True))
        self.imu_topic = rospy.get_param("~imu_topic", "/imu/data")
        self.slope_enter_pitch_deg = abs(
            float(rospy.get_param("~slope_enter_pitch_deg", 4.5))
        )
        self.slope_exit_pitch_deg = abs(
            float(rospy.get_param("~slope_exit_pitch_deg", 3.0))
        )
        self.slope_filter_max_range_m = max(
            0.0, float(rospy.get_param("~slope_filter_max_range_m", 3.0))
        )
        self.slope_filter_cell_size_m = max(
            0.03, float(rospy.get_param("~slope_filter_cell_size_m", 0.18))
        )
        self.slope_filter_min_z_span_m = max(
            0.0, float(rospy.get_param("~slope_filter_min_z_span_m", 0.14))
        )
        self.slope_filter_remove_sparse_cells = bool(
            rospy.get_param("~slope_filter_remove_sparse_cells", True)
        )
        self.slope_filter_min_points_per_cell = max(
            1, int(rospy.get_param("~slope_filter_min_points_per_cell", 2))
        )
        self.slope_filter_keep_front_close_m = max(
            0.0, float(rospy.get_param("~slope_filter_keep_front_close_m", 0.35))
        )
        self._last_pitch_deg = 0.0
        self._slope_mode = False

        self.log_period_s = max(0.0, float(rospy.get_param("~log_period_s", 1.0)))
        self._last_log_time = rospy.Time(0)

        self.pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=1)
        self.sub_imu = None
        if self.enable_pitch_footprint_adjustment or (
            self.enable_slope_residual_filter and self.use_imu_slope_gate
        ):
            self.sub_imu = rospy.Subscriber(
                self.imu_topic, Imu, self.imu_callback, queue_size=20
            )
        self.sub = rospy.Subscriber(
            self.input_topic,
            PointCloud2,
            self.cloud_callback,
            queue_size=1,
            buff_size=2**24,
        )

        rospy.loginfo(
            "self_footprint_cloud_filter started | input=%s output=%s frame=%s "
            "mask_center=(%.2f, %.2f) mask=[x %.2f..%.2f, |y|<=%.2f, z %.2f..%.2f] "
            "drop_zero=%s",
            self.input_topic,
            self.output_topic,
            self.base_frame,
            self.center_x_m,
            self.center_y_m,
            -self.rear_m,
            self.front_m,
            self.half_width_m,
            self.min_z_m,
            self.max_z_m,
            str(self.drop_zero_points).lower(),
        )
        if self.enable_pitch_footprint_adjustment:
            rospy.loginfo(
                "self_footprint_cloud_filter pitch footprint adjustment | "
                "uphill_sign=%.0f deadband=%.2fdeg primary_gain=%.3fm/deg "
                "opposite_gain=%.3fm/deg max_extra=%.2fm",
                self.pitch_uphill_sign,
                self.pitch_adjust_deadband_deg,
                self.pitch_adjust_primary_gain_m_per_deg,
                self.pitch_adjust_opposite_gain_m_per_deg,
                self.pitch_adjust_max_extension_m,
            )
        if self.enable_slope_residual_filter:
            rospy.loginfo(
                "self_footprint_cloud_filter slope residual filter | imu_gate=%s "
                "enter=%.2fdeg exit=%.2fdeg range=%.2fm cell=%.2fm z_span<=%.2fm "
                "min_pts=%d sparse=%s",
                str(self.use_imu_slope_gate).lower(),
                self.slope_enter_pitch_deg,
                self.slope_exit_pitch_deg,
                self.slope_filter_max_range_m,
                self.slope_filter_cell_size_m,
                self.slope_filter_min_z_span_m,
                self.slope_filter_min_points_per_cell,
                str(self.slope_filter_remove_sparse_cells).lower(),
            )

    def imu_callback(self, msg):
        q = msg.orientation
        try:
            _roll, pitch, _yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        except (ValueError, TypeError):
            return
        self._last_pitch_deg = math.degrees(pitch)
        abs_pitch = abs(self._last_pitch_deg)
        if self._slope_mode:
            if abs_pitch <= self.slope_exit_pitch_deg:
                self._slope_mode = False
        elif abs_pitch >= self.slope_enter_pitch_deg:
            self._slope_mode = True

    def _pitch_adjusted_extents(self):
        front_m = self.front_m
        rear_m = self.rear_m
        if not self.enable_pitch_footprint_adjustment:
            return front_m, rear_m

        signed_pitch = self._last_pitch_deg * self.pitch_uphill_sign
        abs_effective_pitch = abs(signed_pitch) - self.pitch_adjust_deadband_deg
        if abs_effective_pitch <= 0.0:
            return front_m, rear_m

        primary_extra = min(
            self.pitch_adjust_max_extension_m,
            abs_effective_pitch * self.pitch_adjust_primary_gain_m_per_deg,
        )
        opposite_extra = min(
            self.pitch_adjust_max_extension_m,
            abs_effective_pitch * self.pitch_adjust_opposite_gain_m_per_deg,
        )

        if signed_pitch > 0.0:
            # Uphill: the robot nose/front is more likely to appear in the cloud.
            front_m += primary_extra
            rear_m += opposite_extra
        else:
            # Downhill: mirror the same idea toward the rear.
            rear_m += primary_extra
            front_m += opposite_extra
        return front_m, rear_m

    def _inside_self_mask(self, x, y, z):
        if self.drop_zero_points and abs(x) < 1e-6 and abs(y) < 1e-6 and abs(z) < 1e-6:
            return True
        if z < self.min_z_m or z > self.max_z_m:
            return False
        local_x = x - self.center_x_m
        local_y = y - self.center_y_m
        front_m, rear_m = self._pitch_adjusted_extents()
        return -rear_m <= local_x <= front_m and abs(local_y) <= self.half_width_m

    def _slope_filter_active(self):
        if not self.enable_slope_residual_filter:
            return False
        if not self.use_imu_slope_gate:
            return True
        return self._slope_mode

    def _remove_slope_residuals(self, points):
        if not points or self.slope_filter_max_range_m <= 0.0:
            return points, 0

        cell_size = self.slope_filter_cell_size_m
        max_range_sq = self.slope_filter_max_range_m * self.slope_filter_max_range_m
        cells = {}
        for idx, (x, y, z) in enumerate(points):
            range_sq = x * x + y * y
            if range_sq > max_range_sq:
                continue
            # Do not erase the immediate front strip; this is where a foot or bumper contact matters.
            if 0.0 <= x <= self.slope_filter_keep_front_close_m and abs(y) <= self.half_width_m:
                continue
            key = (int(math.floor(x / cell_size)), int(math.floor(y / cell_size)))
            cell = cells.get(key)
            if cell is None:
                cells[key] = [idx, idx, 1, z, z]
            else:
                cell[1] = idx
                cell[2] += 1
                if z < cell[3]:
                    cell[3] = z
                if z > cell[4]:
                    cell[4] = z

        remove_cells = set()
        for key, (_first_idx, _last_idx, count, z_min, z_max) in cells.items():
            z_span = z_max - z_min
            if self.slope_filter_remove_sparse_cells and count < self.slope_filter_min_points_per_cell:
                remove_cells.add(key)
                continue
            if count >= self.slope_filter_min_points_per_cell and z_span <= self.slope_filter_min_z_span_m:
                remove_cells.add(key)

        if not remove_cells:
            return points, 0

        filtered = []
        removed = 0
        for x, y, z in points:
            range_sq = x * x + y * y
            if range_sq <= max_range_sq:
                key = (int(math.floor(x / cell_size)), int(math.floor(y / cell_size)))
                if key in remove_cells:
                    removed += 1
                    continue
            filtered.append((x, y, z))
        return filtered, removed

    def cloud_callback(self, msg):
        if msg.header.frame_id != self.base_frame:
            rospy.logwarn_throttle(
                2.0,
                "self_footprint_cloud_filter: expected frame %s but got %s; "
                "publishing unfiltered cloud",
                self.base_frame,
                msg.header.frame_id,
            )
            self.pub.publish(msg)
            return

        kept_points = []
        total = 0
        removed_self = 0
        for x, y, z in point_cloud2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True
        ):
            total += 1
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue
            if self._inside_self_mask(float(x), float(y), float(z)):
                removed_self += 1
                continue
            kept_points.append((float(x), float(y), float(z)))

        slope_active = self._slope_filter_active()
        removed_slope = 0
        if slope_active:
            kept_points, removed_slope = self._remove_slope_residuals(kept_points)

        out = point_cloud2.create_cloud_xyz32(msg.header, kept_points)
        self.pub.publish(out)

        if self.log_period_s > 0.0:
            now = rospy.Time.now()
            if (now - self._last_log_time).to_sec() >= self.log_period_s:
                self._last_log_time = now
                rospy.loginfo(
                    "self_footprint_cloud_filter: kept=%d removed_self=%d "
                    "removed_slope=%d total=%d slope=%s pitch=%.2fdeg "
                    "front=%.2f rear=%.2f",
                    len(kept_points),
                    removed_self,
                    removed_slope,
                    total,
                    "on" if slope_active else "off",
                    self._last_pitch_deg,
                    *self._pitch_adjusted_extents(),
                )


def main():
    rospy.init_node("self_footprint_cloud_filter", anonymous=False)
    SelfFootprintCloudFilter()
    rospy.spin()


if __name__ == "__main__":
    main()
