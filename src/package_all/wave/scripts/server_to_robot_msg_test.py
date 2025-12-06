#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist
from wave.msg import robot_to_server,server_to_robot
import math

# 서버로부터 수신받을 데이터 변수
ser_Robot_id = 0
ser_status_cmd = 2
ser_Dest_point = 0
ser_dest_lat = 35.17918888541
ser_dest_lon = 126.90607244451
ser_vel_ctr = 1
ser_Velcmd = 1.2
ser_Anvelcmd = 36000.0 * math.pi / 180.0  # [rad/s]
ser_Alivecnt = 0

pub_server_to_robot = rospy.Publisher('server_to_robot_topic', server_to_robot, queue_size=10)


def listener():
    rospy.init_node('server_to_robot_msg_test', anonymous=True)
    rate = rospy.Rate(2)  # 1Hz

    while not rospy.is_shutdown():
        # server_to_robot 메시지 발행
        msg = server_to_robot()
        msg.Robot_ID = ser_Robot_id
        msg.Cmd_drive_mode = ser_status_cmd
        msg.Cmd_dest_index = ser_Dest_point
        msg.Cmd_dest_lat = ser_dest_lat
        msg.Cmd_dest_lon = ser_dest_lon
        msg.Cmd_use_vel_control = ser_vel_ctr
        msg.Cmd_linear_velocity = ser_Velcmd
        msg.Cmd_angular_velocity = ser_Anvelcmd
        msg.Alive = ser_Alivecnt
        pub_server_to_robot.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass