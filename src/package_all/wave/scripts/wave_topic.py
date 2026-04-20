#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob

import rospy
import serial
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


class WaveTopicBridge(object):
    def __init__(self):
        self.port_name = rospy.get_param("~port_name", "").strip()
        self.baud_rate = int(rospy.get_param("~baud_rate", 19200))
        self.timeout_s = float(rospy.get_param("~timeout_s", 1.0))
        self.log_period_s = max(0.2, float(rospy.get_param("~log_period_s", 1.0)))
        self.linear_scale = float(rospy.get_param("~linear_scale", 1.0))
        self.angular_scale = float(rospy.get_param("~angular_scale", 1.0))
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.emergency_stop_topic = rospy.get_param("~emergency_stop_topic", "/planning/emergency_stop")
        self.emergency_stop_timeout_s = max(
            0.05, float(rospy.get_param("~emergency_stop_timeout_s", 0.35))
        )
        self.emergency_zero_hz = max(
            5.0, float(rospy.get_param("~emergency_zero_hz", 20.0))
        )
        self.port_candidates = rospy.get_param(
            "~port_candidates",
            ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0", "/dev/ttyACM1"],
        )

        self.alive_count = 0
        self.emergency_stop_active = False
        self.emergency_stop_last_stamp = rospy.Time(0)
        resolved_port = self._resolve_port()
        if not resolved_port:
            rospy.logfatal(
                "wave_topic: no serial port found. checked=%s",
                ", ".join(self.port_candidates),
            )
            raise rospy.ROSInitException("wave_topic serial port not found")

        try:
            self.ser = serial.Serial(resolved_port, self.baud_rate, timeout=self.timeout_s)
        except serial.SerialException as exc:
            rospy.logfatal(
                "wave_topic: failed to open serial port %s at %d baud: %s",
                resolved_port,
                self.baud_rate,
                str(exc),
            )
            raise rospy.ROSInitException("wave_topic serial open failed")

        self.port_name = resolved_port
        self.sub = rospy.Subscriber(self.cmd_vel_topic, Twist, self.callback, queue_size=10)
        self.sub_emergency = rospy.Subscriber(
            self.emergency_stop_topic, Bool, self.emergency_stop_callback, queue_size=5
        )
        self.emergency_timer = rospy.Timer(
            rospy.Duration(1.0 / self.emergency_zero_hz),
            self.emergency_timer_callback,
        )
        rospy.loginfo(
            "wave_topic started | port=%s baud=%d timeout=%.2fs cmd=%s linear_scale=%.2f angular_scale=%.2f emergency=%s",
            self.port_name,
            self.baud_rate,
            self.timeout_s,
            self.cmd_vel_topic,
            self.linear_scale,
            self.angular_scale,
            self.emergency_stop_topic,
        )

    def _resolve_port(self):
        if self.port_name:
            return self.port_name

        for candidate in self.port_candidates:
            matches = sorted(glob.glob(candidate))
            if matches:
                return matches[0]
        return ""

    def twist_to_serial_data(self, twist):
        self.alive_count = (self.alive_count + 1) % 256
        linear = float(twist.linear.x) * self.linear_scale
        angular = float(twist.angular.z) * self.angular_scale
        return f"{linear},{angular},0,{self.alive_count}\n"

    def emergency_stop_fresh(self):
        if not self.emergency_stop_active:
            return False
        if self.emergency_stop_last_stamp.to_sec() <= 0.0:
            return False
        return (rospy.Time.now() - self.emergency_stop_last_stamp).to_sec() <= self.emergency_stop_timeout_s

    def write_twist(self, twist):
        serial_data = self.twist_to_serial_data(twist)
        try:
            self.ser.write(serial_data.encode())
        except serial.SerialException as exc:
            rospy.logerr_throttle(
                self.log_period_s,
                "wave_topic: serial communication error on %s: %s",
                self.port_name,
                str(exc),
            )

    def write_zero_stop(self):
        self.write_twist(Twist())

    def emergency_stop_callback(self, msg):
        self.emergency_stop_active = bool(msg.data)
        self.emergency_stop_last_stamp = rospy.Time.now()
        if self.emergency_stop_active:
            rospy.logwarn_throttle(
                self.log_period_s,
                "wave_topic: emergency stop active -> forcing serial zero",
            )
            self.write_zero_stop()

    def emergency_timer_callback(self, _event):
        if self.emergency_stop_fresh():
            self.write_zero_stop()

    def callback(self, data):
        out = Twist() if self.emergency_stop_fresh() else data
        scaled_linear = float(out.linear.x) * self.linear_scale
        scaled_angular = float(out.angular.z) * self.angular_scale
        rospy.loginfo_throttle(
            self.log_period_s,
            "wave_topic: tx cmd | ros(v=%.3f w=%.3f) scaled(v=%.3f w=%.3f) port=%s estop=%s",
            out.linear.x,
            out.angular.z,
            scaled_linear,
            scaled_angular,
            self.port_name,
            "on" if self.emergency_stop_fresh() else "off",
        )
        self.write_twist(out)

    def close(self):
        if hasattr(self, "ser") and self.ser and self.ser.is_open:
            self.ser.close()


def listener():
    rospy.init_node("wave_topic", anonymous=False)
    bridge = WaveTopicBridge()
    rospy.on_shutdown(bridge.close)
    rospy.spin()


if __name__ == "__main__":
    try:
        listener()
    except (rospy.ROSInterruptException, rospy.ROSInitException):
        pass
