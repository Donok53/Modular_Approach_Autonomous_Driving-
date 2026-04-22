cmake_minimum_required(VERSION 3.0.2)
project(autonomy_explainability_bridge)

find_package(catkin REQUIRED COMPONENTS
  dynamic_window_approach
  geometry_msgs
  nav_msgs
  rospy
  sensor_msgs
  std_msgs
  visualization_msgs
)

catkin_package(
  CATKIN_DEPENDS
    dynamic_window_approach
    geometry_msgs
    nav_msgs
    rospy
    sensor_msgs
    std_msgs
    visualization_msgs
)

catkin_install_python(PROGRAMS
  scripts/explainability_topic_bridge.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch docs
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
