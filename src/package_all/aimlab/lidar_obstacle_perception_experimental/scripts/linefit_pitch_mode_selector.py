#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import message_filters
import rospy
import tf.transformations as transformations
from sensor_msgs.msg import Imu, PointCloud2
from std_msgs.msg import String


class LinefitPitchModeSelector:
    def __init__(self):
        self.imu_topic = rospy.get_param("~imu_topic", "/imu/data")
        self.flat_ground_topic = rospy.get_param("~flat_ground_topic")
        self.flat_non_ground_topic = rospy.get_param("~flat_non_ground_topic")
        self.slope_ground_topic = rospy.get_param("~slope_ground_topic")
        self.slope_non_ground_topic = rospy.get_param("~slope_non_ground_topic")
        self.output_ground_topic = rospy.get_param("~output_ground_topic")
        self.output_non_ground_topic = rospy.get_param("~output_non_ground_topic")
        self.mode_topic = rospy.get_param(
            "~mode_topic", "/experimental/linefit_ground/mode"
        )

        self.enter_slope_pitch_deg = float(
            rospy.get_param("~enter_slope_pitch_deg", 4.5)
        )
        self.exit_slope_pitch_deg = float(
            rospy.get_param("~exit_slope_pitch_deg", 3.0)
        )
        self.sync_slop_s = float(rospy.get_param("~sync_slop_s", 0.03))
        self.queue_size = int(rospy.get_param("~queue_size", 8))

        self.current_pitch_deg = 0.0
        self.current_mode = "flat"

        self.pub_ground = rospy.Publisher(
            self.output_ground_topic, PointCloud2, queue_size=1
        )
        self.pub_non_ground = rospy.Publisher(
            self.output_non_ground_topic, PointCloud2, queue_size=1
        )
        self.pub_mode = rospy.Publisher(self.mode_topic, String, queue_size=1)

        self.sub_imu = rospy.Subscriber(
            self.imu_topic, Imu, self.imu_callback, queue_size=50
        )

        self.sub_flat_ground = message_filters.Subscriber(
            self.flat_ground_topic, PointCloud2
        )
        self.sub_flat_non_ground = message_filters.Subscriber(
            self.flat_non_ground_topic, PointCloud2
        )
        self.sub_slope_ground = message_filters.Subscriber(
            self.slope_ground_topic, PointCloud2
        )
        self.sub_slope_non_ground = message_filters.Subscriber(
            self.slope_non_ground_topic, PointCloud2
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [
                self.sub_flat_ground,
                self.sub_flat_non_ground,
                self.sub_slope_ground,
                self.sub_slope_non_ground,
            ],
            queue_size=self.queue_size,
            slop=self.sync_slop_s,
        )
        self.sync.registerCallback(self.synced_clouds_callback)

        rospy.loginfo(
            "linefit_pitch_mode_selector started | imu=%s enter=%.2fdeg exit=%.2fdeg sync_slop=%.3fs",
            self.imu_topic,
            self.enter_slope_pitch_deg,
            self.exit_slope_pitch_deg,
            self.sync_slop_s,
        )

    def imu_callback(self, msg):
        q = msg.orientation
        _, pitch, _ = transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.current_pitch_deg = math.degrees(float(pitch))

    def _select_mode(self):
        pitch_abs = abs(self.current_pitch_deg)
        if self.current_mode == "flat" and pitch_abs >= self.enter_slope_pitch_deg:
            self.current_mode = "slope"
        elif self.current_mode == "slope" and pitch_abs <= self.exit_slope_pitch_deg:
            self.current_mode = "flat"
        return self.current_mode

    def synced_clouds_callback(
        self, flat_ground, flat_non_ground, slope_ground, slope_non_ground
    ):
        mode = self._select_mode()
        if mode == "slope":
            out_ground = slope_ground
            out_non_ground = slope_non_ground
        else:
            out_ground = flat_ground
            out_non_ground = flat_non_ground

        self.pub_ground.publish(out_ground)
        self.pub_non_ground.publish(out_non_ground)
        self.pub_mode.publish(String(data=mode))
        rospy.loginfo_throttle(
            1.0,
            "linefit_pitch_mode_selector: mode=%s pitch=%.2fdeg",
            mode,
            self.current_pitch_deg,
        )


def main():
    rospy.init_node("linefit_pitch_mode_selector", anonymous=False)
    LinefitPitchModeSelector()
    rospy.spin()


if __name__ == "__main__":
    main()
