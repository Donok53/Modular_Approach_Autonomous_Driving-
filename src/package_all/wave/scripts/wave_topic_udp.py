#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist
from wave.msg import robot_to_server,server_to_robot
import serial
import time
import socket
import struct
import threading

# 시리얼 포트 설정 및 연결
port_name = "/dev/ttyUSB0"
baud_rate = 19200
ser = serial.Serial(port_name, baud_rate, timeout=1)

# 서버 IP 주소 및 포트
server_ip = '121.148.234.22'
server_port = 9101

# UDP 소켓 생성
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 전송할 데이터 초기화
robot_id = 0
robot_State = 0
robot_Cur_index = 0
robot_Dest_index = 0
robot_Cur_lat = 0
robot_Cur_long = 0
robot_Cur_lin_vel = 0
robot_Cur_ang_vel = 0
Dist_destination = 0
Remain_Dist = 0
alive_count = 0

# 서버로부터 수신받을 데이터 변수
ser_Robot_id = 0
ser_status_cmd = 0
ser_Dest_point = 0
ser_vel_ctr = 0
ser_Velcmd = 0.0
ser_Anvelcmd = 0.0
ser_Alivecnt = 0

# 데이터 패킹 형식
send_data_format = '>BBBBddffffB'
receive_data_format = '>BBBBffB'

pub_server_to_robot = rospy.Publisher('server_to_robot_topic', server_to_robot, queue_size=10)

def twist_to_serial_data(twist):
    global CurrVel, CurrAnvel
    linear_speed = twist.linear.x
    angular_speed = twist.angular.z
    CurrVel = linear_speed
    CurrAnvel = angular_speed
    etc = 0
    
    serial_data = f"{linear_speed},{angular_speed},{etc},{alive_count}\n"
    return serial_data

def callback(data):
    serial_data = twist_to_serial_data(data)
    try:
        ser.write(serial_data.encode())
    except serial.SerialException as e:
        rospy.logerr("Serial communication error: %s", str(e))

def robot_to_server_callback(robot_data):
    global robot_id, robot_State, robot_Cur_index, robot_Dest_index, robot_Cur_lat, robot_Cur_long, robot_Cur_lin_vel,robot_Cur_ang_vel,Dist_destination,Remain_Dist, alive_count
    
    robot_id = robot_data.ID
    robot_State = robot_data.Cur_state
    robot_Cur_index = robot_data.Cur_node_index
    robot_Dest_index = robot_data.Dest_node_index
    robot_Cur_lat = robot_data.Cur_latitude
    robot_Cur_long = robot_data.Cur_longitude
    robot_Cur_lin_vel = robot_data.Cur_linear_velocity
    robot_Cur_ang_vel = robot_data.Cur_angular_velocity
    Dist_destination = robot_data.Dist_to_destination
    Remain_Dist = robot_data.Remain_dist_to_destination
    # robot_alive_cnt = robot_data.Alive_count
    
    # alive_count = (alive_count + 1) % 256  
    # rospy.loginfo("Received robot_to_server message: ID=%d, State=%d, Cur_Node_Index=%d, Dest_Node_Index=%d, Latitude=%f, Longitude=%f, Linear_Vel=%f, Angular_Vel=%f, Dist_to_dest=%f, Remain_dist_to_dest=%f,robot_alive_cnt=%d",
    #               robot_id, robot_State, robot_Cur_index, robot_Dest_index, robot_Cur_lat, robot_Cur_long, robot_Cur_lin_vel, robot_Cur_ang_vel, Dist_destination, Remain_Dist, robot_alive_cnt)


def server_communication():

    global alive_count, CurrVel,CurrAnvel,ser_Robot_id, ser_status_cmd, ser_Dest_point,ser_vel_ctr, ser_Velcmd, ser_Anvelcmd, ser_Alivecnt,robot_id, robot_State ,robot_Cur_index ,robot_Dest_index ,robot_Cur_lat ,robot_Cur_long ,robot_Cur_lin_vel ,robot_Cur_ang_vel ,Dist_destination ,Remain_Dist
    
    try:
        while not rospy.is_shutdown():
            
            packed_data = struct.pack(send_data_format, robot_id, robot_State ,robot_Cur_index ,robot_Dest_index ,robot_Cur_lat ,robot_Cur_long ,robot_Cur_lin_vel ,robot_Cur_ang_vel ,Dist_destination ,Remain_Dist,alive_count)
            sock.sendto(packed_data, (server_ip, server_port))
            rospy.loginfo("Message sent to %s:%s with robot_Cur_lat: %s,robot_Cur_long: %s, robot_State: %s, alive_count: %s", server_ip, server_port, robot_Cur_lat, robot_Cur_long, robot_State,alive_count)
            alive_count = (alive_count + 1) % 256
            
            try:
                sock.settimeout(0.5)
                data, server = sock.recvfrom(1024)
                unpacked_data = struct.unpack(receive_data_format, data)
                # 수신 데이터 글로벌 변수 업데이트
                ser_Robot_id, ser_status_cmd, ser_Dest_point, ser_vel_ctr, ser_Velcmd, ser_Anvelcmd, ser_Alivecnt = unpacked_data
                # rospy.loginfo("Received data updated: Robot ID: %d, Status Cmd: %d, Dest Point: %d, Vel cmd use: %d, Vel Cmd: %.2f, Anvel Cmd: %.2f, Alive Count: %d", ser_Robot_id, ser_status_cmd, ser_Dest_point, ser_vel_ctr, ser_Velcmd, ser_Anvelcmd, ser_Alivecnt )

                # server_to_robot 메시지 발행
                msg = server_to_robot()
                msg.Robot_ID = ser_Robot_id
                msg.Cmd_drive_mode = ser_status_cmd
                msg.Cmd_dest_index = ser_Dest_point
                msg.Cmd_use_vel_control = ser_vel_ctr
                msg.Cmd_linear_velocity = ser_Velcmd
                msg.Cmd_angular_velocity = ser_Anvelcmd
                msg.Alive = ser_Alivecnt
                pub_server_to_robot.publish(msg)
            except socket.timeout:
                rospy.loginfo("Timed out waiting for a response from the server")
            
            time.sleep(0.5)
    except KeyboardInterrupt:
        rospy.loginfo("Server communication stopped")
    finally:
        sock.close()

def listener():
    rospy.init_node('robot_command_listener', anonymous=True)
    rospy.Subscriber("cmd_vel", Twist, callback)
    rospy.Subscriber("robot_to_server_topic", robot_to_server, robot_to_server_callback)

    # 서버 통신을 별도의 스레드로 실행
    server_thread = threading.Thread(target=server_communication)
    server_thread.daemon = True 
    server_thread.start()

    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass
    finally:
        ser.close()  
