cmake_minimum_required(VERSION 3.0.2)
project(lidar_height_estimator)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy sensor_msgs std_msgs
)

catkin_install_python(PROGRAMS
  scripts/lidar_height_estimator.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
