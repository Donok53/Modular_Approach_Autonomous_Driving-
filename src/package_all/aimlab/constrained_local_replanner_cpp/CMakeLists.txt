cmake_minimum_required(VERSION 3.0.2)
project(constrained_local_replanner_cpp)

# C++14 keeps us compatible with ROS Noetic stock toolchain.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  nav_msgs
  sensor_msgs
  std_msgs
  geometry_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  pcl_ros
  pcl_conversions
  eigen_conversions
  dynamic_window_approach
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
    roscpp
    nav_msgs
    sensor_msgs
    std_msgs
    geometry_msgs
    visualization_msgs
    tf2
    tf2_ros
    tf2_geometry_msgs
    pcl_ros
    pcl_conversions
    dynamic_window_approach
  DEPENDS EIGEN3 PCL
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/grid.cpp
  src/path_ops.cpp
  src/cluster.cpp
  src/avoidance_planner.cpp
  src/avoidance_state.cpp
  src/branch_search.cpp
  src/viz.cpp
)
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

add_executable(${PROJECT_NAME}_node
  src/node.cpp
  src/main.cpp
)
target_link_libraries(${PROJECT_NAME}_node
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)
set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
