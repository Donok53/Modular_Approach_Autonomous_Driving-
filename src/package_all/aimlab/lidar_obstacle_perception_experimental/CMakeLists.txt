cmake_minimum_required(VERSION 3.0.2)
project(lidar_obstacle_perception_experimental)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  rospy
  sensor_msgs
  std_msgs
  visualization_msgs
  tf
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs rospy sensor_msgs std_msgs visualization_msgs tf
)

catkin_install_python(PROGRAMS
  scripts/lidar_obstacle_perception_experimental.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
