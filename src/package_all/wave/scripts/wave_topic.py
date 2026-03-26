#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob

import rospy
import serial
from geometry_msgs.msg import Twist


class WaveTopicBridge(object):
    def __init__(self):
        self.port_name = rospy.get_param("~port_name", "").strip()
        self.baud_rate = int(rospy.get_param("~baud_rate", 19200))
        self.timeout_s = float(rospy.get_param("~timeout_s", 1.0))
        self.log_period_s = max(0.2, float(rospy.get_param("~log_period_s", 1.0)))
        self.port_candidates = rospy.get_param(
            "~port_candidates",
            ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0", "/dev/ttyACM1"],
        )

        self.alive_count = 0
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
        self.sub = rospy.Subscriber("cmd_vel", Twist, self.callback, queue_size=10)
        rospy.loginfo(
            "wave_topic started | port=%s baud=%d timeout=%.2fs",
            self.port_name,
            self.baud_rate,
            self.timeout_s,
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
        return f"{twist.linear.x},{twist.angular.z},0,{self.alive_count}\n"

    def callback(self, data):
        rospy.loginfo_throttle(
            self.log_period_s,
            "wave_topic: tx cmd | v=%.3f w=%.3f port=%s",
            data.linear.x,
            data.angular.z,
            self.port_name,
        )
        serial_data = self.twist_to_serial_data(data)
        try:
            self.ser.write(serial_data.encode())
        except serial.SerialException as exc:
            rospy.logerr_throttle(
                self.log_period_s,
                "wave_topic: serial communication error on %s: %s",
                self.port_name,
                str(exc),
            )

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
