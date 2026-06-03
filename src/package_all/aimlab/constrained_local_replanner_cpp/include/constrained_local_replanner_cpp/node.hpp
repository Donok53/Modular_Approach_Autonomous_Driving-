#pragma once

#include <mutex>
#include <string>

#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>

#include "constrained_local_replanner_cpp/avoidance_planner.hpp"
#include "constrained_local_replanner_cpp/types.hpp"

namespace clr {

// Minimal ROS wrapper that demonstrates the C++ planner end-to-end. It
// subscribes to the same inputs the Python replanner uses and publishes a
// candidate avoidance path on a parallel topic so the Python and C++
// implementations can run side by side during validation. **Not** a drop-in
// replacement for the Python node yet — the state machine, trigger
// debouncing, locked-static memory, branch search, etc. are still living in
// Python. See README.md for the full scoreboard.
class ReplannerNode {
 public:
  ReplannerNode(ros::NodeHandle nh, ros::NodeHandle pnh);

 private:
  void timerCB(const ros::TimerEvent&);
  void cloudCB(const sensor_msgs::PointCloud2::ConstPtr& msg);
  void gridCB(const nav_msgs::OccupancyGrid::ConstPtr& msg);
  void odomCB(const nav_msgs::Odometry::ConstPtr& msg);
  void globalPathCB(const nav_msgs::Path::ConstPtr& msg);

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  ros::Subscriber sub_cloud_;
  ros::Subscriber sub_grid_;
  ros::Subscriber sub_odom_;
  ros::Subscriber sub_global_path_;
  ros::Publisher pub_local_path_;
  ros::Timer timer_;

  std::mutex state_mu_;
  std::vector<WorldXY> latest_obstacle_points_;
  ros::Time latest_obstacle_stamp_;
  nav_msgs::OccupancyGrid::ConstPtr latest_grid_;
  nav_msgs::Odometry::ConstPtr latest_odom_;
  nav_msgs::Path::ConstPtr latest_global_path_;

  PlannerParams params_;
  std::string output_local_path_topic_{"/planning/local_path_cpp"};
  double cloud_z_min_{0.10};
  double cloud_z_max_{1.30};
  double cloud_voxel_m_{0.10};
  double loop_period_s_{0.10};
};

}  // namespace clr
