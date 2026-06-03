#include "constrained_local_replanner_cpp/node.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/utils.h>

#include "constrained_local_replanner_cpp/cluster.hpp"
#include "constrained_local_replanner_cpp/grid.hpp"
#include "constrained_local_replanner_cpp/path_ops.hpp"

namespace clr {

ReplannerNode::ReplannerNode(ros::NodeHandle nh, ros::NodeHandle pnh)
    : nh_(nh), pnh_(pnh) {
  // Topic names mirror the Python defaults so the C++ node can drop in next
  // to the existing pipeline.
  std::string cloud_topic = "/ouster/points";
  std::string grid_topic = "/lio_sam/drivable_area/grid";
  std::string odom_topic = "/lio_localizer/odometry/optimization";
  std::string global_path_topic = "/astar/path";

  pnh_.param("cloud_topic", cloud_topic, cloud_topic);
  pnh_.param("drivable_grid_topic", grid_topic, grid_topic);
  pnh_.param("odom_topic", odom_topic, odom_topic);
  pnh_.param("global_path_topic", global_path_topic, global_path_topic);
  pnh_.param("local_path_topic", output_local_path_topic_, output_local_path_topic_);

  pnh_.param("loop_period_s", loop_period_s_, loop_period_s_);
  pnh_.param("cloud_z_min_m", cloud_z_min_, cloud_z_min_);
  pnh_.param("cloud_z_max_m", cloud_z_max_, cloud_z_max_);
  pnh_.param("cloud_voxel_m", cloud_voxel_m_, cloud_voxel_m_);

  pnh_.param("lookahead_m", params_.lookahead_m, params_.lookahead_m);
  pnh_.param("avoidance_trigger_ahead_m", params_.avoidance_trigger_ahead_m,
             params_.avoidance_trigger_ahead_m);
  pnh_.param("obstacle_block_margin_m", params_.obstacle_block_margin_m,
             params_.obstacle_block_margin_m);
  pnh_.param("sidestep_min_offset_m", params_.sidestep_min_offset_m,
             params_.sidestep_min_offset_m);
  pnh_.param("sidestep_max_offset_m", params_.sidestep_max_offset_m,
             params_.sidestep_max_offset_m);
  pnh_.param("sidestep_preview_m", params_.sidestep_preview_m,
             params_.sidestep_preview_m);
  pnh_.param("rejoin_min_distance_m", params_.rejoin_min_distance_m,
             params_.rejoin_min_distance_m);
  pnh_.param("grid_unknown_is_occupied", params_.grid_unknown_is_occupied,
             params_.grid_unknown_is_occupied);

  pub_local_path_ = nh_.advertise<nav_msgs::Path>(output_local_path_topic_, 2, true);
  sub_cloud_ = nh_.subscribe(cloud_topic, 2, &ReplannerNode::cloudCB, this);
  sub_grid_ = nh_.subscribe(grid_topic, 2, &ReplannerNode::gridCB, this);
  sub_odom_ = nh_.subscribe(odom_topic, 5, &ReplannerNode::odomCB, this);
  sub_global_path_ = nh_.subscribe(global_path_topic, 2, &ReplannerNode::globalPathCB, this);

  timer_ = nh_.createTimer(ros::Duration(loop_period_s_), &ReplannerNode::timerCB, this);
  ROS_INFO("constrained_local_replanner_cpp started | cloud=%s grid=%s odom=%s "
           "global=%s out=%s",
           cloud_topic.c_str(), grid_topic.c_str(), odom_topic.c_str(),
           global_path_topic.c_str(), output_local_path_topic_.c_str());
}

void ReplannerNode::cloudCB(const sensor_msgs::PointCloud2::ConstPtr& msg) {
  if (!msg) return;
  // Lazy conversion: only keep XY of points inside the configured z-band.
  pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
  pcl::fromROSMsg(*msg, pcl_cloud);
  std::vector<WorldXY> pts;
  pts.reserve(pcl_cloud.size() / 4);
  for (const auto& p : pcl_cloud.points) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
    if (p.z < cloud_z_min_ || p.z > cloud_z_max_) continue;
    pts.push_back(WorldXY{static_cast<double>(p.x), static_cast<double>(p.y)});
  }
  auto down = voxelDownsample2D(pts, cloud_voxel_m_);
  std::lock_guard<std::mutex> lk(state_mu_);
  latest_obstacle_points_ = std::move(down);
  latest_obstacle_stamp_ = msg->header.stamp;
}

void ReplannerNode::gridCB(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
  std::lock_guard<std::mutex> lk(state_mu_);
  latest_grid_ = msg;
}

void ReplannerNode::odomCB(const nav_msgs::Odometry::ConstPtr& msg) {
  std::lock_guard<std::mutex> lk(state_mu_);
  latest_odom_ = msg;
}

void ReplannerNode::globalPathCB(const nav_msgs::Path::ConstPtr& msg) {
  std::lock_guard<std::mutex> lk(state_mu_);
  latest_global_path_ = msg;
}

void ReplannerNode::timerCB(const ros::TimerEvent&) {
  nav_msgs::OccupancyGrid::ConstPtr grid;
  nav_msgs::Odometry::ConstPtr odom;
  nav_msgs::Path::ConstPtr global;
  std::vector<WorldXY> obstacle_pts;
  {
    std::lock_guard<std::mutex> lk(state_mu_);
    grid = latest_grid_;
    odom = latest_odom_;
    global = latest_global_path_;
    obstacle_pts = latest_obstacle_points_;
  }
  if (!grid || !odom || !global || global->poses.size() < 2) {
    return;
  }

  const OccupancyView g = fromOccupancyGrid(*grid);
  auto blocked = baseBlockedMask(g, params_.grid_unknown_is_occupied);
  // Overlay obstacle points (inflated by the block margin) onto the mask.
  overlayPoints(g, blocked, obstacle_pts,
                params_.obstacle_block_margin_m + params_.footprint.half_width_m);

  // Convert the global path to grid cells for the lookahead window.
  std::vector<WorldXY> global_world;
  global_world.reserve(global->poses.size());
  for (const auto& ps : global->poses) {
    global_world.push_back(
        WorldXY{ps.pose.position.x, ps.pose.position.y});
  }
  auto nominal_cells = worldPointsToGridPath(global_world, g);
  if (nominal_cells.size() < 2) return;

  const double rx = odom->pose.pose.position.x;
  const double ry = odom->pose.pose.position.y;
  double yaw = 0.0;
  {
    tf2::Quaternion q(odom->pose.pose.orientation.x,
                      odom->pose.pose.orientation.y,
                      odom->pose.pose.orientation.z,
                      odom->pose.pose.orientation.w);
    double roll, pitch;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
  }
  const GridCell start = g.world_to_cell(rx, ry);

  // Is the nominal path blocked ahead?
  const bool blocked_ahead =
      pathBlockedAhead(nominal_cells, blocked, g.width, g.height, start,
                       g.resolution_m, params_.avoidance_trigger_ahead_m);
  if (!blocked_ahead) {
    // Publish the nominal path itself (truncated to lookahead).
    nav_msgs::Path out;
    out.header.stamp = ros::Time::now();
    out.header.frame_id = grid->header.frame_id;
    double accum = 0.0;
    for (std::size_t i = 0; i < global_world.size() && accum < params_.lookahead_m; ++i) {
      geometry_msgs::PoseStamped ps;
      ps.header = out.header;
      ps.pose.position.x = global_world[i].x;
      ps.pose.position.y = global_world[i].y;
      ps.pose.orientation.w = 1.0;
      out.poses.push_back(ps);
      if (i > 0) {
        accum += std::hypot(global_world[i].x - global_world[i - 1].x,
                            global_world[i].y - global_world[i - 1].y);
      }
    }
    pub_local_path_.publish(out);
    return;
  }

  // Pick the obstacle centroid nearest to the nominal path as the blocker
  // for the sidestep planner. This is a placeholder until the full
  // memory/locked-static logic is ported.
  WorldXY blocker_world{rx + 2.0 * std::cos(yaw), ry + 2.0 * std::sin(yaw)};
  if (!obstacle_pts.empty()) {
    auto clusters = clusterPoints2D(obstacle_pts, params_.pointcloud_cluster_resolution_m);
    double best = std::numeric_limits<double>::infinity();
    for (const auto& cl : clusters) {
      const double dxy = std::hypot(cl.centroid.x - rx, cl.centroid.y - ry);
      if (dxy < best) {
        best = dxy;
        blocker_world = cl.centroid;
      }
    }
  }

  const AvoidanceResult res = buildSidestepAvoidance(
      nominal_cells, blocked, g, start, blocker_world, params_, yaw, /*pref*/ 0);

  nav_msgs::Path out;
  out.header.stamp = ros::Time::now();
  out.header.frame_id = grid->header.frame_id;
  if (res.found) {
    for (const auto& wp : res.waypoints) {
      geometry_msgs::PoseStamped ps;
      ps.header = out.header;
      ps.pose.position.x = wp.x;
      ps.pose.position.y = wp.y;
      ps.pose.orientation.w = 1.0;
      out.poses.push_back(ps);
    }
    ROS_INFO_THROTTLE(1.0,
        "cpp sidestep found | side=%d offset_or_curv=%.3f waypoints=%zu",
        res.side, res.max_curvature, res.waypoints.size());
  } else {
    ROS_WARN_THROTTLE(1.0,
        "cpp sidestep no solution — falling back to empty local path");
    // Empty path lets the Python pipeline retain authority.
  }
  pub_local_path_.publish(out);
}

}  // namespace clr
