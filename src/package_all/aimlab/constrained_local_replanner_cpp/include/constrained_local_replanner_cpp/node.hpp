#pragma once

#include <memory>
#include <mutex>
#include <string>

#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "constrained_local_replanner_cpp/avoidance_planner.hpp"
#include "constrained_local_replanner_cpp/avoidance_state.hpp"
#include "constrained_local_replanner_cpp/types.hpp"

namespace clr {

// ROS wrapper. Two run modes selectable via the ~primary_mode parameter:
//   primary_mode=false (default): publishes candidate output on the
//     shadow topics (~local_path_topic, ~path_mode_topic). Python remains
//     authoritative.
//   primary_mode=true: publishes onto the same topics the DWA controller
//     consumes (/planning/local_path and /planning/path_mode). The Python
//     constrained_local_replanner must be disabled in the launch file to
//     avoid topic contention.
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
  ros::Publisher pub_path_mode_;
  ros::Publisher pub_avoidance_path_;
  ros::Publisher pub_path_history_;
  ros::Publisher pub_travel_history_;
  ros::Publisher pub_travel_history_path_;
  ros::Publisher pub_recognized_obstacles_;
  ros::Publisher pub_blocking_obstacles_;
  ros::Publisher pub_global_overlay_;
  ros::Publisher pub_debug_text_;
  ros::Publisher pub_explainability_;
  ros::Timer timer_;

  std::vector<WorldXY> travel_history_;

  // tf2 listener used to transform incoming point clouds from the sensor
  // frame (e.g. os_sensor) into the planning frame (e.g. map). Without this
  // every cloud point was overlaid at the wrong cell.
  tf2_ros::Buffer tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  std::string target_frame_{"map"};
  std::string base_frame_{"base_link"};
  double tf_wait_s_{0.05};

  std::mutex state_mu_;
  std::vector<WorldXY> latest_obstacle_points_;
  ros::Time latest_obstacle_stamp_;
  nav_msgs::OccupancyGrid::ConstPtr latest_grid_;
  nav_msgs::Odometry::ConstPtr latest_odom_;
  nav_msgs::Path::ConstPtr latest_global_path_;

  // Cached last-published avoidance so we can apply the
  // endpoint-distance / locked-static / drivable-revalidation guards the
  // Python node uses.
  std::vector<GridCell> last_avoid_grid_path_;
  std::vector<WorldXY>  last_avoid_world_path_;
  bool                   last_avoid_active_{false};

  PlannerParams params_;
  std::unique_ptr<AvoidanceStateMachine> sm_;
  std::string output_local_path_topic_{"/planning/local_path_cpp"};
  std::string output_path_mode_topic_{"/planning/path_mode_cpp"};
  std::string output_avoidance_path_topic_{"/planning/avoidance_path_cpp"};
  std::string output_path_history_topic_{"/planning/path_history_cpp"};
  std::string output_travel_history_topic_{"/planning/travel_history_cpp"};
  std::string output_travel_history_path_topic_{"/planning/travel_history_path_cpp"};
  std::string output_recognized_obstacles_topic_{"/planning/recognized_obstacles_cpp"};
  std::string output_blocking_obstacles_topic_{"/planning/blocking_obstacles_cpp"};
  std::string output_global_overlay_topic_{"/planning/global_obstacle_overlay_cpp"};
  std::string output_debug_text_topic_{"/planning/local_replanner_debug_text_cpp"};
  std::string output_explainability_topic_{"/planning/explainability_cpp"};
  bool primary_mode_{false};
  // Tracks the path mode published on the previous tick so we can fire an
  // ExplainabilityEvent only on transitions, matching Python's behaviour.
  PathMode last_published_mode_{PathMode::FOLLOW_LOCAL};
  bool first_mode_publish_{true};
  double cloud_z_min_{0.04};
  double cloud_z_max_{1.30};
  double cloud_world_z_min_{0.03};
  double cloud_voxel_m_{0.10};
  double loop_period_s_{0.10};
  bool self_filter_enabled_{true};
  double self_filter_front_m_{0.54};
  double self_filter_rear_m_{0.46};
  double self_filter_half_width_m_{0.44};
  double self_filter_log_period_s_{1.0};
  double last_self_filter_log_sec_{0.0};
  int global_progress_backtrack_points_{20};
  bool have_global_progress_{false};
  std::size_t last_global_near_idx_{0};
  bool have_global_signature_{false};
  std::size_t last_global_path_size_{0};
  WorldXY last_global_start_{};
  WorldXY last_global_goal_{};
  int branch_max_expand_{4500};
  double branch_time_budget_s_{0.45};
  int branch_max_rejoin_candidates_{12};
  double locked_static_hold_radius_m_{1.2};
};

}  // namespace clr
