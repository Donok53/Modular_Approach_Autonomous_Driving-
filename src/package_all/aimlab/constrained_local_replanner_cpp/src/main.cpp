#include <ros/ros.h>

#include "constrained_local_replanner_cpp/node.hpp"

int main(int argc, char** argv) {
  ros::init(argc, argv, "constrained_local_replanner_cpp");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  clr::ReplannerNode node(nh, pnh);
  ros::spin();
  return 0;
}
