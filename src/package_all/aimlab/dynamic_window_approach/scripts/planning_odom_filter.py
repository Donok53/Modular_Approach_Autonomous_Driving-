#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rospy
from nav_msgs.msg import Odometry


def clamp(value, low, high):
    return max(low, min(high, value))


def wrap_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def angle_diff(target, source):
    return wrap_angle(target - source)


def alpha_from_tau(dt, tau_s):
    tau_s = max(1e-3, float(tau_s))
    return 1.0 - math.exp(-dt / tau_s)


def quat_to_euler(q):
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


def euler_to_quat(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


class PlanningOdomFilter:
    def __init__(self):
        self.input_topic = rospy.get_param(
            "~input_topic", "/lio_localizer/odometry/optimization"
        )
        self.twist_topic = str(
            rospy.get_param("~twist_topic", "/lio_localizer/odometry/lidar_incremental")
        ).strip()
        self.twist_linear_frame = str(
            rospy.get_param("~twist_linear_frame", "auto")
        ).strip().lower()
        self.output_topic = rospy.get_param(
            "~output_topic", "/lio_localizer/odometry/planning"
        )
        self.twist_timeout_s = max(
            0.0, float(rospy.get_param("~twist_timeout_s", 0.30))
        )
        self.output_child_frame_id = str(
            rospy.get_param("~output_child_frame_id", "base_link")
        ).strip()
        self.forward_tau_s = max(0.01, float(rospy.get_param("~forward_tau_s", 0.18)))
        self.lateral_tau_s = max(0.01, float(rospy.get_param("~lateral_tau_s", 0.70)))
        self.turning_lateral_tau_s = max(
            0.01, float(rospy.get_param("~turning_lateral_tau_s", 0.20))
        )
        self.yaw_tau_s = max(0.01, float(rospy.get_param("~yaw_tau_s", 0.45)))
        self.turning_yaw_tau_s = max(
            0.01, float(rospy.get_param("~turning_yaw_tau_s", 0.18))
        )
        self.twist_tau_s = max(0.01, float(rospy.get_param("~twist_tau_s", 0.25)))
        self.lateral_deadband_m = max(
            0.0, float(rospy.get_param("~lateral_deadband_m", 0.03))
        )
        self.yaw_deadband_rad = math.radians(
            max(0.0, float(rospy.get_param("~yaw_deadband_deg", 1.5)))
        )
        self.straight_speed_min_mps = max(
            0.0, float(rospy.get_param("~straight_speed_min_mps", 0.10))
        )
        self.turning_yaw_rate_threshold_rps = max(
            0.0,
            float(rospy.get_param("~turning_yaw_rate_threshold_rps", 0.20)),
        )
        self.turning_yaw_error_threshold_rad = math.radians(
            max(0.0, float(rospy.get_param("~turning_yaw_error_deg", 10.0)))
        )
        self.max_straight_lateral_correction_mps = max(
            0.01,
            float(rospy.get_param("~max_straight_lateral_correction_mps", 0.18)),
        )
        self.reinit_gap_s = max(0.2, float(rospy.get_param("~reinit_gap_s", 1.0)))

        self.have_state = False
        self.last_stamp_s = None
        self.fx = 0.0
        self.fy = 0.0
        self.fz = 0.0
        self.froll = 0.0
        self.fpitch = 0.0
        self.fyaw = 0.0
        self.fvx = 0.0
        self.fvy = 0.0
        self.fvz = 0.0
        self.fwz = 0.0
        self.latest_twist_msg = None
        self.latest_twist_stamp_s = 0.0

        self.pub = rospy.Publisher(self.output_topic, Odometry, queue_size=20)
        self.sub = rospy.Subscriber(
            self.input_topic, Odometry, self.odom_callback, queue_size=50
        )
        self.sub_twist = None
        if self.twist_topic and self.twist_topic != self.input_topic:
            self.sub_twist = rospy.Subscriber(
                self.twist_topic, Odometry, self.twist_callback, queue_size=50
            )

        rospy.loginfo(
            "planning_odom_filter started | pose=%s twist=%s twist_frame=%s out=%s twist_timeout=%.2fs child=%s tau(fwd=%.2f lat=%.2f turn_lat=%.2f yaw=%.2f turn_yaw=%.2f twist=%.2f)",
            self.input_topic,
            self.twist_topic if self.twist_topic else "-",
            self.twist_linear_frame,
            self.output_topic,
            self.twist_timeout_s,
            self.output_child_frame_id if self.output_child_frame_id else "<inherit>",
            self.forward_tau_s,
            self.lateral_tau_s,
            self.turning_lateral_tau_s,
            self.yaw_tau_s,
            self.turning_yaw_tau_s,
            self.twist_tau_s,
        )

    @staticmethod
    def _stamp_to_sec(msg):
        stamp_s = msg.header.stamp.to_sec()
        if stamp_s <= 0.0:
            return rospy.Time.now().to_sec()
        return stamp_s

    def twist_callback(self, msg):
        self.latest_twist_msg = msg
        self.latest_twist_stamp_s = self._stamp_to_sec(msg)

    def _select_twist_msg(self, pose_msg, pose_stamp_s):
        if self.twist_topic and self.twist_topic != self.input_topic:
            if self.latest_twist_msg is not None:
                if self.twist_timeout_s <= 0.0:
                    return self.latest_twist_msg
                if abs(pose_stamp_s - self.latest_twist_stamp_s) <= self.twist_timeout_s:
                    return self.latest_twist_msg
        return pose_msg

    def _resolved_child_frame_id(self, pose_msg, twist_msg):
        if self.output_child_frame_id:
            return self.output_child_frame_id
        pose_child = str(pose_msg.child_frame_id).strip()
        if pose_child:
            return pose_child
        if twist_msg is not None:
            twist_child = str(twist_msg.child_frame_id).strip()
            if twist_child:
                return twist_child
        return "base_link"

    def _resolve_twist_linear_frame(self, pose_msg, twist_msg):
        frame_mode = self.twist_linear_frame
        if frame_mode in ("child", "body", "base"):
            return "child"
        if frame_mode in ("world", "map", "odom", "global", "pose"):
            return "world"

        pose_frame = str(pose_msg.header.frame_id).strip()
        twist_frame = str(twist_msg.header.frame_id).strip()
        twist_child = str(twist_msg.child_frame_id).strip()
        output_child = self._resolved_child_frame_id(pose_msg, twist_msg)

        if twist_child and output_child and twist_child == output_child:
            return "child"
        if pose_frame and twist_frame and pose_frame == twist_frame:
            return "world"
        return "child"

    def _extract_body_twist(self, pose_msg, twist_msg, yaw):
        raw_vx = float(twist_msg.twist.twist.linear.x)
        raw_vy = float(twist_msg.twist.twist.linear.y)
        raw_vz = float(twist_msg.twist.twist.linear.z)
        raw_wz = float(twist_msg.twist.twist.angular.z)

        if self._resolve_twist_linear_frame(pose_msg, twist_msg) == "world":
            cy = math.cos(yaw)
            sy = math.sin(yaw)
            body_vx = cy * raw_vx + sy * raw_vy
            body_vy = -sy * raw_vx + cy * raw_vy
            return body_vx, body_vy, raw_vz, raw_wz

        return raw_vx, raw_vy, raw_vz, raw_wz

    def _reset_state(self, pose_msg, raw_roll, raw_pitch, raw_yaw, twist_msg):
        msg = pose_msg
        p = msg.pose.pose.position
        raw_vx, raw_vy, raw_vz, raw_wz = self._extract_body_twist(
            pose_msg, twist_msg, raw_yaw
        )
        self.fx = float(p.x)
        self.fy = float(p.y)
        self.fz = float(p.z)
        self.froll = raw_roll
        self.fpitch = raw_pitch
        self.fyaw = raw_yaw
        self.fvx = raw_vx
        self.fvy = raw_vy
        self.fvz = raw_vz
        self.fwz = raw_wz
        self.have_state = True

    def _publish_filtered(self, pose_msg, twist_msg):
        out = Odometry()
        out.header = pose_msg.header
        out.child_frame_id = self._resolved_child_frame_id(pose_msg, twist_msg)
        out.pose = pose_msg.pose
        out.twist = twist_msg.twist

        out.pose.pose.position.x = self.fx
        out.pose.pose.position.y = self.fy
        out.pose.pose.position.z = self.fz
        qx, qy, qz, qw = euler_to_quat(self.froll, self.fpitch, self.fyaw)
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw
        out.twist.twist.linear.x = self.fvx
        out.twist.twist.linear.y = self.fvy
        out.twist.twist.linear.z = self.fvz
        out.twist.twist.angular.z = self.fwz
        self.pub.publish(out)

    def odom_callback(self, pose_msg):
        stamp_s = self._stamp_to_sec(pose_msg)
        twist_msg = self._select_twist_msg(pose_msg, stamp_s)

        raw_roll, raw_pitch, raw_yaw = quat_to_euler(pose_msg.pose.pose.orientation)

        if (not self.have_state) or (self.last_stamp_s is None):
            self._reset_state(pose_msg, raw_roll, raw_pitch, raw_yaw, twist_msg)
            self.last_stamp_s = stamp_s
            self._publish_filtered(pose_msg, twist_msg)
            return

        dt = stamp_s - self.last_stamp_s
        self.last_stamp_s = stamp_s
        if dt <= 0.0 or dt > self.reinit_gap_s:
            self._reset_state(pose_msg, raw_roll, raw_pitch, raw_yaw, twist_msg)
            self._publish_filtered(pose_msg, twist_msg)
            return

        raw_x = float(pose_msg.pose.pose.position.x)
        raw_y = float(pose_msg.pose.pose.position.y)
        raw_z = float(pose_msg.pose.pose.position.z)
        raw_vx, raw_vy, raw_vz, raw_wz = self._extract_body_twist(
            pose_msg, twist_msg, raw_yaw
        )
        raw_speed = math.hypot(raw_vx, raw_vy)
        yaw_err = angle_diff(raw_yaw, self.fyaw)
        turning = (
            abs(raw_wz) >= self.turning_yaw_rate_threshold_rps
            or abs(yaw_err) >= self.turning_yaw_error_threshold_rad
        )

        alpha_fwd = alpha_from_tau(dt, self.forward_tau_s)
        alpha_lat = alpha_from_tau(
            dt, self.turning_lateral_tau_s if turning else self.lateral_tau_s
        )
        alpha_yaw = alpha_from_tau(
            dt, self.turning_yaw_tau_s if turning else self.yaw_tau_s
        )
        alpha_twist = alpha_from_tau(dt, self.twist_tau_s)

        dx = raw_x - self.fx
        dy = raw_y - self.fy
        cy = math.cos(self.fyaw)
        sy = math.sin(self.fyaw)
        fwd_err = cy * dx + sy * dy
        lat_err = -sy * dx + cy * dy

        if (not turning) and raw_speed >= self.straight_speed_min_mps:
            if abs(lat_err) < self.lateral_deadband_m:
                lat_err = 0.0
            if abs(yaw_err) < self.yaw_deadband_rad:
                yaw_err = 0.0

        fwd_step = alpha_fwd * fwd_err
        lat_step = alpha_lat * lat_err

        if (not turning) and raw_speed >= self.straight_speed_min_mps:
            max_lat_step = self.max_straight_lateral_correction_mps * dt
            lat_step = clamp(lat_step, -max_lat_step, max_lat_step)

        self.fx += cy * fwd_step - sy * lat_step
        self.fy += sy * fwd_step + cy * lat_step
        self.fz += alpha_fwd * (raw_z - self.fz)
        self.froll = raw_roll
        self.fpitch = raw_pitch
        self.fyaw = wrap_angle(self.fyaw + alpha_yaw * yaw_err)
        self.fvx += alpha_twist * (raw_vx - self.fvx)
        self.fvy += alpha_twist * (raw_vy - self.fvy)
        self.fvz += alpha_twist * (raw_vz - self.fvz)
        self.fwz += alpha_twist * (raw_wz - self.fwz)

        self._publish_filtered(pose_msg, twist_msg)


def main():
    rospy.init_node("planning_odom_filter", anonymous=False)
    PlanningOdomFilter()
    rospy.spin()


if __name__ == "__main__":
    main()
