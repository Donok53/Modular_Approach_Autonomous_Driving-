#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
import requests

SERVER_URL = "http://localhost:5000/robot/location/update"  # 실제 서버 IP로 교체


def odom_callback(msg):
    # x가 위도, y가 경도인지 반드시 ROS 토픽 메시지와 실측 좌표를 확인하세요.
    lat = msg.pose.pose.position.x
    lon = msg.pose.pose.position.y
    timestamp = str(rospy.Time.now().to_sec())
    data = {
        "latitude": lat,
        "longitude": lon,
        "timestamp": timestamp
    }
    try:
        requests.post(SERVER_URL, json=data, timeout=0.5)
    except Exception as e:
        rospy.logwarn("Failed to send robot location: %s" % e)

if __name__ == '__main__':
    rospy.init_node('robot_location_sender')
    rospy.Subscriber('/lio_localizer/odometry/optimization_wgs84', Odometry, odom_callback, queue_size=1)
    rospy.spin()

