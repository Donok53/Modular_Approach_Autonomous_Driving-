#!/usr/bin/env python3
import rospy
import datetime
from std_msgs.msg import String

# pose msg
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix

# Velocity
from geometry_msgs.msg import Twist

# Path info
from std_msgs.msg import Int32MultiArray, Float32MultiArray

# Server 
from wave.msg import robot_to_server, server_to_robot

import requests

# Global variable to store message data
publish_data = robot_to_server()
publish_data.ID = 1
reach_goal_flag = False
path_node_id_list = []

def server_to_robot_callback(data):
    global publish_data, reach_goal_flag
    if reach_goal_flag == False:
        publish_data.Cur_state = data.Cmd_drive_mode
    else:
        publish_data.Cur_state = 3

def path_node_id_list_callback(data):
    global path_node_id_list
    path_node_id_list = data.data
    print("len(path_node_id_list): ", len(path_node_id_list))


def traj_info_callback(data):
    global publish_data, reach_goal_flag
    # print("len(path_node_id_list): ", len(path_node_id_list), ", int(data.data[2]): ", int(data.data[2]))
    if len(path_node_id_list) > 0 and len(path_node_id_list) >= int(data.data[2]) - 1:
        cur_node_index_value = (int) (path_node_id_list[int(data.data[2])])
        dest_node_index_value = (int) (path_node_id_list[-1])
        if cur_node_index_value >= 0 and dest_node_index_value >= 0:
            publish_data.Cur_node_index = (int) (path_node_id_list[int(data.data[2])])
            publish_data.Dest_node_index = (int) (path_node_id_list[-1])

    publish_data.Dist_to_destination = data.data[0]
    publish_data.Remain_dist_to_destination = data.data[0] - data.data[1]
    reach_goal_flag = data.data[3]
    if reach_goal_flag == True:
        publish_data.Cur_state = 3
    
def pose_wgs84_callback(data):
    global publish_data
    publish_data.Cur_latitude = data.pose.pose.position.x
    publish_data.Cur_longitude = data.pose.pose.position.y

    # POST to Flask
    robot_location = {
        "latitude": float(publish_data.Cur_latitude),
        "longitude": float(publish_data.Cur_longitude),
        "timestamp": str(datetime.datetime.utcnow())
    }
    try:
        requests.post('http://10.0.2.2:5000/robot/location', json=robot_location, timeout=0.5)
    except Exception as e:
        print("Failed to post robot location: ", e)


#def gnss_callback(data):
#    global publish_data
#    publish_data.Cur_latitude = data.latitude
#    publish_data.Cur_longitude = data.longitude
    
    # rospy.loginfo("GNSS Received lat: %f, lon: %f", publish_data.Cur_latitude, publish_data.Cur_longitude)

def cmd_vel_callback(data):
    global publish_data
    # Handle another subscription
    publish_data.Cur_linear_velocity = data.linear.x
    publish_data.Cur_angular_velocity = data.angular.z
    
    # rospy.loginfo("cmd vel Received x: %f, z: %f", publish_data.Cur_linear_velocity, publish_data.Cur_angular_velocity)

# def path_info_callback(data):


# robot_to_server 메시지 생성 및 데이터 할당
#publish_data = robot_to_server()
#publish_data.ID = 1 
#publish_data.Cur_state = 2
#publish_data.Cur_node_index = 3
#publish_data.Dest_node_index = 4
#publish_data.Cur_latitude = random.uniform(-90.0, 90.0)
#publish_data.Cur_longitude = random.uniform(-180.0, 180.0)
#publish_data.Cur_linear_velocity = random.uniform(0.0, 10.0)
#publish_data.Cur_angular_velocity = random.uniform(-5.0, 5.0)
#publish_data.Dist_to_destination = random.uniform(0.0, 1000.0)
#publish_data.Remain_dist_to_destination = random.uniform(0.0, 500.0)
#publish_data.Alive_count = alive_count

def timer_callback(event):
    global publish_data
    # Publish data every 0.5 seconds (2Hz)
    pub.publish(publish_data)
    print("ID:", publish_data.ID)
    print("Cur_state:", publish_data.Cur_state)
    print("Cur_node_index:", publish_data.Cur_node_index)
    print("Dest_node_index:", publish_data.Dest_node_index)
    print("Cur_latitude:", publish_data.Cur_latitude)
    print("Cur_longitude:", publish_data.Cur_longitude)
    print("Cur_linear_velocity:", publish_data.Cur_linear_velocity)
    print("Cur_angular_velocity:", publish_data.Cur_angular_velocity)
    print("Dist_to_destination:", publish_data.Dist_to_destination)
    print("Remain_dist_to_destination:", publish_data.Remain_dist_to_destination)
    print("Alive_count:", publish_data.Alive_count)

def listener():
    rospy.init_node('multi_listener_publisher', anonymous=True)

    # Subscribers
    rospy.Subscriber('/server_to_robot_topic',server_to_robot, server_to_robot_callback)
    rospy.Subscriber('lio_localizer/odometry/optimization_wgs84',Odometry, pose_wgs84_callback)
    rospy.Subscriber('/astar/path_node_id_list',Int32MultiArray, path_node_id_list_callback)
    rospy.Subscriber('/traj_info',Float32MultiArray, traj_info_callback)
    # rospy.Subscriber("/smc_2000/fix", NavSatFix, gnss_callback)
    rospy.Subscriber("/cmd_vel", Twist, cmd_vel_callback)

    # Publisher
    global pub
    pub = rospy.Publisher('robot_to_server_topic', robot_to_server, queue_size=10)

    # Timer to publish messages at 2Hz
    rospy.Timer(rospy.Duration(0.5), timer_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
