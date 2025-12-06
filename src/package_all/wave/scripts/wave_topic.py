#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist
import serial
import time

#시리얼 포트 설정
port_name = "/dev/ttyUSB0"
baud_rate = 19200

#시리얼 연결
ser = serial.Serial(port_name, baud_rate, timeout=1)

#Alive 카운트
alive_count = 0

def twist_to_serial_data(twist):
    global alive_count

    linear_speed = twist.linear.x
    angular_speed = twist.angular.z
    etc = 0
    alive_count = (alive_count + 1) % 256

    serial_data = f"{linear_speed},{angular_speed},{etc},{alive_count}\n"

    return serial_data

def callback(data):
    rospy.loginfo("Received velocity command: linear_speed=%s, angular_speed=%s", data.linear.x, data.angular.z)
    serial_data = twist_to_serial_data(data)
    try:
        ser.write(serial_data.encode())

    except serial.SerialException as e:
        rospy.logerr("Serial communication error: %s", str(e))

def listener():
    rospy.init_node('robot_command_listener', anonymous=True)
    rospy.Subscriber("cmd_vel", Twist, callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass
    finally:
        ser.close()