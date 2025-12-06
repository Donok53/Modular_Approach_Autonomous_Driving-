#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from wave.msg import robot_to_server
import random

alive_count = 0

def publish_robot_to_server():
    global alive_count
    # ROS 퍼블리셔 초기화
    pub = rospy.Publisher('robot_to_server_topic', robot_to_server, queue_size=10)
    rospy.init_node('robot_to_server_publisher', anonymous=True)
    rate = rospy.Rate(2)  # 1Hz
    alive_count = (alive_count + 1) % 256

    while not rospy.is_shutdown():

        # robot_to_server 메시지 생성 및 데이터 할당
        msg = robot_to_server()
        msg.ID = 1
        msg.Cur_state = 2
        msg.Cur_node_index = 3
        msg.Dest_node_index = 4
        msg.Cur_latitude = random.uniform(-90.0, 90.0)
        msg.Cur_longitude = random.uniform(-180.0, 180.0)
        msg.Cur_linear_velocity = random.uniform(0.0, 10.0)
        msg.Cur_angular_velocity = random.uniform(-5.0, 5.0)
        msg.Dist_to_destination = random.uniform(0.0, 1000.0)
        msg.Remain_dist_to_destination = random.uniform(0.0, 500.0)
        msg.Alive_count = alive_count

        # 로그 메시지 출력 및 토픽 발행
        rospy.loginfo("Publishing robot_to_server message: ID=%d, State=%d, Latitude=%.2f, Longitude=%.2f, Alive_count=%.2f", 
                      msg.ID, msg.Cur_state, msg.Cur_latitude, msg.Cur_longitude,alive_count)
        pub.publish(msg)

        rate.sleep()

if __name__ == '__main__':
    try:
        publish_robot_to_server()
    except rospy.ROSInterruptException:
        pass
