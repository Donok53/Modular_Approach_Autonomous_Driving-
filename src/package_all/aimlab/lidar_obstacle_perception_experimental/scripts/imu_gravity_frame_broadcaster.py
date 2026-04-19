#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math

import rospy
import tf
from sensor_msgs.msg import Imu


class ImuGravityFrameBroadcaster:
    def __init__(self):
        self.imu_topic = rospy.get_param("~imu_topic", "/imu/data")
        self.raw_frame = str(rospy.get_param("~raw_frame", "os_sensor")).strip() or "os_sensor"
        self.gravity_frame = (
            str(rospy.get_param("~gravity_frame", self.raw_frame + "_gravity")).strip()
            or (self.raw_frame + "_gravity")
        )
        self.zero_yaw = bool(rospy.get_param("~zero_yaw", True))

        self.br = tf.TransformBroadcaster()
        self.sub = rospy.Subscriber(self.imu_topic, Imu, self.imu_callback, queue_size=50)

        rospy.loginfo(
            "imu_gravity_frame_broadcaster started | imu=%s raw_frame=%s gravity_frame=%s zero_yaw=%s",
            self.imu_topic,
            self.raw_frame,
            self.gravity_frame,
            str(self.zero_yaw).lower(),
        )

    def imu_callback(self, msg):
        q = msg.orientation
        norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        if norm < 1e-6:
            return

        roll, pitch, yaw = tf.transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w]
        )
        target_yaw = 0.0 if self.zero_yaw else -yaw
        quat = tf.transformations.quaternion_from_euler(-roll, -pitch, target_yaw)
        stamp = msg.header.stamp if msg.header.stamp.to_sec() > 0.0 else rospy.Time.now()
        self.br.sendTransform(
            (0.0, 0.0, 0.0),
            quat,
            stamp,
            self.gravity_frame,
            self.raw_frame,
        )


def main():
    rospy.init_node("imu_gravity_frame_broadcaster", anonymous=False)
    ImuGravityFrameBroadcaster()
    rospy.spin()


if __name__ == "__main__":
    main()
