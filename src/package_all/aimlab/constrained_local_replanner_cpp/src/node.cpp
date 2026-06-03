#include "constrained_local_replanner_cpp/node.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <std_msgs/String.h>
#include <tf2/utils.h>

#include "constrained_local_replanner_cpp/branch_search.hpp"
#include "constrained_local_replanner_cpp/cluster.hpp"
#include "constrained_local_replanner_cpp/grid.hpp"
#include "constrained_local_replanner_cpp/path_ops.hpp"

namespace clr {

namespace {

// Truncate the global path to roughly `len_m` meters ahead of `start_idx`.
std::vector<WorldXY> truncateWorld(const std::vector<WorldXY>& world,
                                   std::size_t start_idx, double len_m) {
  std::vector<WorldXY> out;
  if (world.empty()) return out;
  out.push_back(world[start_idx]);
  double accum = 0.0;
  for (std::size_t i = start_idx + 1; i < world.size(); ++i) {
    accum += std::hypot(world[i].x - world[i - 1].x, world[i].y - world[i - 1].y);
    out.push_back(world[i]);
    if (accum >= len_m) break;
  }
  return out;
}

std::size_t nearestIndex(const std::vector<WorldXY>& world, double rx, double ry) {
  std::size_t best = 0;
  double bd = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < world.size(); ++i) {
    const double d = std::hypot(world[i].x - rx, world[i].y - ry);
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}

bool cellIsBlockedAt(const std::vector<uint8_t>& blocked, int width, int height,
                     int gx, int gy) {
  if (gx < 0 || gy < 0 || gx >= width || gy >= height) return true;
  return blocked[static_cast<std::size_t>(gy) * static_cast<std::size_t>(width) +
                 static_cast<std::size_t>(gx)] != 0;
}

// Down-sample a grid path to ~`step_m` spacing so the emitted local path is
// not unnecessarily dense (the DWA controller does its own resampling but a
// 5 cm grid path explodes RViz markers).
std::vector<WorldXY> samplePath(const std::vector<GridCell>& grid_path,
                                const OccupancyView& g, double step_m) {
  std::vector<WorldXY> out;
  if (grid_path.empty()) return out;
  out.push_back(g.cell_to_world(grid_path.front().x, grid_path.front().y));
  double accum = 0.0;
  for (std::size_t i = 1; i < grid_path.size(); ++i) {
    const WorldXY w = g.cell_to_world(grid_path[i].x, grid_path[i].y);
    const double d = std::hypot(w.x - out.back().x, w.y - out.back().y);
    accum += d;
    if (accum >= step_m || i == grid_path.size() - 1) {
      out.push_back(w);
      accum = 0.0;
    }
  }
  return out;
}

}  // namespace

ReplannerNode::ReplannerNode(ros::NodeHandle nh, ros::NodeHandle pnh)
    : nh_(nh), pnh_(pnh) {
  std::string cloud_topic = "/ouster/points";
  std::string grid_topic = "/lio_sam/drivable_area/grid";
  std::string odom_topic = "/lio_localizer/odometry/optimization";
  std::string global_path_topic = "/astar/path";

  pnh_.param("cloud_topic", cloud_topic, cloud_topic);
  pnh_.param("drivable_grid_topic", grid_topic, grid_topic);
  pnh_.param("odom_topic", odom_topic, odom_topic);
  pnh_.param("global_path_topic", global_path_topic, global_path_topic);
  pnh_.param("primary_mode", primary_mode_, primary_mode_);
  if (primary_mode_) {
    output_local_path_topic_ = "/planning/local_path";
    output_path_mode_topic_  = "/planning/path_mode";
  }
  pnh_.param("local_path_topic", output_local_path_topic_, output_local_path_topic_);
  pnh_.param("path_mode_topic", output_path_mode_topic_, output_path_mode_topic_);

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
  pnh_.param("branch_max_expand", branch_max_expand_, branch_max_expand_);
  pnh_.param("branch_time_budget_s", branch_time_budget_s_, branch_time_budget_s_);
  pnh_.param("branch_max_rejoin_candidates", branch_max_rejoin_candidates_,
             branch_max_rejoin_candidates_);

  AvoidanceStateMachine::Config smc;
  pnh_.param("trigger_confirm_cycles", smc.trigger_confirm_cycles,
             smc.trigger_confirm_cycles);
  pnh_.param("clear_detour_hold_s", smc.clear_detour_hold_s,
             smc.clear_detour_hold_s);
  pnh_.param("keep_until_endpoint_distance_m", smc.keep_until_endpoint_distance_m,
             smc.keep_until_endpoint_distance_m);
  pnh_.param("locked_static_hit_radius_m", smc.locked_static_hit_radius_m,
             smc.locked_static_hit_radius_m);
  pnh_.param("locked_static_persistence_hits", smc.locked_static_persistence_hits,
             smc.locked_static_persistence_hits);
  pnh_.param("locked_static_hold_radius_m", smc.locked_static_hold_radius_m,
             smc.locked_static_hold_radius_m);
  sm_ = std::make_unique<AvoidanceStateMachine>(smc);

  pub_local_path_ = nh_.advertise<nav_msgs::Path>(output_local_path_topic_, 2, true);
  pub_path_mode_  = nh_.advertise<std_msgs::String>(output_path_mode_topic_, 2, true);
  sub_cloud_ = nh_.subscribe(cloud_topic, 2, &ReplannerNode::cloudCB, this);
  sub_grid_ = nh_.subscribe(grid_topic, 2, &ReplannerNode::gridCB, this);
  sub_odom_ = nh_.subscribe(odom_topic, 5, &ReplannerNode::odomCB, this);
  sub_global_path_ = nh_.subscribe(global_path_topic, 2, &ReplannerNode::globalPathCB, this);

  timer_ = nh_.createTimer(ros::Duration(loop_period_s_), &ReplannerNode::timerCB, this);
  ROS_INFO("constrained_local_replanner_cpp started | primary=%s "
           "local_path=%s path_mode=%s",
           primary_mode_ ? "yes" : "no",
           output_local_path_topic_.c_str(),
           output_path_mode_topic_.c_str());
}

void ReplannerNode::cloudCB(const sensor_msgs::PointCloud2::ConstPtr& msg) {
  if (!msg) return;
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
  if (!grid || !odom || !global || global->poses.size() < 2) return;

  const OccupancyView g = fromOccupancyGrid(*grid);
  auto blocked = baseBlockedMask(g, params_.grid_unknown_is_occupied);
  overlayPoints(g, blocked, obstacle_pts,
                params_.obstacle_block_margin_m + params_.footprint.half_width_m);

  std::vector<WorldXY> global_world;
  global_world.reserve(global->poses.size());
  for (const auto& ps : global->poses) {
    global_world.push_back(WorldXY{ps.pose.position.x, ps.pose.position.y});
  }

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

  // Truncate the global path to the lookahead window from the robot.
  const std::size_t near_idx = nearestIndex(global_world, rx, ry);
  const auto nominal_world = truncateWorld(global_world, near_idx, params_.lookahead_m);
  const auto nominal_cells = worldPointsToGridPath(nominal_world, g);
  if (nominal_cells.size() < 2) return;

  // Is the nominal path blocked ahead?
  const bool nominal_blocked =
      pathBlockedAhead(nominal_cells, blocked, g.width, g.height, start,
                       g.resolution_m, params_.avoidance_trigger_ahead_m);

  // Record candidate static blockers from the obstacle clusters.
  auto clusters = clusterPoints2D(obstacle_pts, params_.pointcloud_cluster_resolution_m);
  WorldXY blocker_world{rx + 2.0 * std::cos(yaw), ry + 2.0 * std::sin(yaw)};
  {
    double best = std::numeric_limits<double>::infinity();
    for (const auto& cl : clusters) {
      const double dxy = std::hypot(cl.centroid.x - rx, cl.centroid.y - ry);
      if (dxy < best) { best = dxy; blocker_world = cl.centroid; }
      sm_->recordStaticHit(cl.centroid);
    }
  }

  // Try sidestep first, then branch search as a fallback.
  AvoidanceResult candidate;
  std::vector<GridCell> candidate_cells;
  std::vector<WorldXY>  candidate_world;
  if (nominal_blocked) {
    candidate = buildSidestepAvoidance(nominal_cells, blocked, g, start,
                                       blocker_world, params_, yaw, 0);
    if (candidate.found) {
      candidate_cells = candidate.grid_path;
      candidate_world = candidate.waypoints;
    } else {
      // Build rejoin candidates from the lookahead nominal path past the
      // blocker — only cells that pass the drivable + dynamic-blocked check.
      std::vector<GridCell> rejoin_candidates;
      const std::size_t nominal_n = nominal_cells.size();
      const double res = g.resolution_m;
      double traveled = 0.0;
      for (std::size_t i = 1; i < nominal_n; ++i) {
        traveled += std::hypot(
            static_cast<double>(nominal_cells[i].x - nominal_cells[i - 1].x),
            static_cast<double>(nominal_cells[i].y - nominal_cells[i - 1].y)) * res;
        if (traveled < params_.rejoin_min_distance_m) continue;
        if (cellIsBlockedAt(blocked, g.width, g.height,
                            nominal_cells[i].x, nominal_cells[i].y)) continue;
        rejoin_candidates.push_back(nominal_cells[i]);
        if (static_cast<int>(rejoin_candidates.size()) >=
            branch_max_rejoin_candidates_) break;
      }
      if (!rejoin_candidates.empty()) {
        candidate_cells = aStarBranch(blocked, g.width, g.height, start,
                                      rejoin_candidates, branch_max_expand_,
                                      branch_time_budget_s_);
        if (candidate_cells.size() >= 2) {
          candidate_world = samplePath(candidate_cells, g, 0.20);
          candidate.found = true;
        }
      }
    }
  }

  // Validate the cached avoidance path against the *current* drivable + dynamic
  // mask before we lean on it for the release guard.
  bool cached_drivable = true;
  double endpoint_dist = std::numeric_limits<double>::infinity();
  if (last_avoid_active_ && !last_avoid_grid_path_.empty()) {
    for (const auto& c : last_avoid_grid_path_) {
      if (cellIsBlockedAt(blocked, g.width, g.height, c.x, c.y)) {
        cached_drivable = false;
        break;
      }
    }
    endpoint_dist = gridPathEndpointDistanceToXY(last_avoid_grid_path_, g, rx, ry);
  }

  const bool locked_static_nearby =
      sm_->lockedStaticNearby(WorldXY{rx, ry}, /*radius_m*/ 3.0);
  const PathMode mode = sm_->update(
      nominal_blocked, /*avoidance_candidate_available=*/candidate.found,
      endpoint_dist, cached_drivable, locked_static_nearby,
      ros::Time::now().toSec());

  // Choose what to publish based on the resolved mode.
  nav_msgs::Path out;
  out.header.stamp = ros::Time::now();
  out.header.frame_id = grid->header.frame_id;
  auto emit_world_path = [&out](const std::vector<WorldXY>& pts) {
    for (const auto& p : pts) {
      geometry_msgs::PoseStamped ps;
      ps.header = out.header;
      ps.pose.position.x = p.x;
      ps.pose.position.y = p.y;
      ps.pose.orientation.w = 1.0;
      out.poses.push_back(ps);
    }
  };

  switch (mode) {
    case PathMode::FOLLOW_AVOIDANCE: {
      if (candidate.found) {
        emit_world_path(candidate_world);
        last_avoid_grid_path_ = candidate_cells;
        last_avoid_world_path_ = candidate_world;
        last_avoid_active_ = true;
      } else if (last_avoid_active_ && cached_drivable) {
        emit_world_path(last_avoid_world_path_);
      } else {
        // No candidate, cached invalid: fall back to nominal and let the
        // state machine flip to HOLD next tick.
        emit_world_path(nominal_world);
      }
      break;
    }
    case PathMode::FOLLOW_LOCAL: {
      emit_world_path(nominal_world);
      last_avoid_active_ = false;
      last_avoid_grid_path_.clear();
      last_avoid_world_path_.clear();
      break;
    }
    case PathMode::HOLD: {
      // Empty path tells the downstream controller to halt.
      last_avoid_active_ = false;
      last_avoid_grid_path_.clear();
      last_avoid_world_path_.clear();
      break;
    }
  }

  pub_local_path_.publish(out);
  std_msgs::String mode_msg;
  mode_msg.data = pathModeToString(mode);
  pub_path_mode_.publish(mode_msg);

  ROS_INFO_THROTTLE(1.0,
      "cpp planner | mode=%s nominal_blocked=%d candidate=%d cached_drivable=%d "
      "endpoint=%.2f locked_static=%d clusters=%zu obs_pts=%zu",
      mode_msg.data.c_str(), static_cast<int>(nominal_blocked),
      static_cast<int>(candidate.found), static_cast<int>(cached_drivable),
      endpoint_dist, static_cast<int>(locked_static_nearby),
      clusters.size(), obstacle_pts.size());
}

}  // namespace clr
