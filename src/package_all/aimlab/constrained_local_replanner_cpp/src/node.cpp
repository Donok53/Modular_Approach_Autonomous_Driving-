#include "constrained_local_replanner_cpp/node.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <std_msgs/String.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/TransformStamped.h>
#include <dynamic_window_approach/ExplainabilityEvent.h>
#include <chrono>

#include "constrained_local_replanner_cpp/branch_search.hpp"
#include "constrained_local_replanner_cpp/cluster.hpp"
#include "constrained_local_replanner_cpp/grid.hpp"
#include "constrained_local_replanner_cpp/path_ops.hpp"
#include "constrained_local_replanner_cpp/viz.hpp"

namespace clr {

namespace {

bool cellIsBlockedAt(const std::vector<uint8_t>& blocked, int width, int height,
                     int gx, int gy);

std::vector<WorldXY> samplePath(const std::vector<GridCell>& grid_path,
                                const OccupancyView& g, double step_m);

void clearBlockedCircle(const OccupancyView& g, std::vector<uint8_t>& blocked,
                        GridCell center, double radius_m);

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

void appendDistinct(std::vector<WorldXY>& out, WorldXY p, double eps_m = 0.03) {
  if (!out.empty() && std::hypot(out.back().x - p.x, out.back().y - p.y) <= eps_m) {
    return;
  }
  out.push_back(p);
}

void appendGlobalTail(std::vector<WorldXY>& out,
                      const std::vector<WorldXY>& global,
                      std::size_t start_idx,
                      double max_total_len_m) {
  if (global.empty() || out.empty()) return;
  double total = polylineLengthM(out);
  for (std::size_t i = start_idx; i < global.size(); ++i) {
    const WorldXY next = global[i];
    const double seg = std::hypot(next.x - out.back().x, next.y - out.back().y);
    appendDistinct(out, next);
    total += seg;
    if (total >= max_total_len_m) break;
  }
}

struct ReturnPath {
  bool found{false};
  std::size_t rejoin_idx{0};
  std::vector<GridCell> grid_path;
  std::vector<WorldXY> world_path;
};

ReturnPath buildReturnToGlobalPath(const std::vector<WorldXY>& global,
                                   const std::vector<uint8_t>& blocked,
                                   const OccupancyView& g,
                                   GridCell start,
                                   double rx,
                                   double ry,
                                   const PlannerParams& params,
                                   int max_expand,
                                   double time_budget_s) {
  ReturnPath result;
  if (global.empty()) return result;
  struct RankedCandidate {
    GridCell cell;
    std::size_t idx;
    double score;
  };
  std::vector<RankedCandidate> ranked;
  ranked.reserve(global.size());
  for (std::size_t i = 0; i < global.size(); ++i) {
    const GridCell cell = g.world_to_cell(global[i].x, global[i].y);
    if (!g.in_bounds(cell.x, cell.y)) continue;
    if (cellIsBlockedAt(blocked, g.width, g.height, cell.x, cell.y)) continue;
    const double d = std::hypot(global[i].x - rx, global[i].y - ry);
    ranked.push_back(RankedCandidate{cell, i, d});
  }
  if (ranked.empty()) return result;
  std::sort(ranked.begin(), ranked.end(),
            [](const RankedCandidate& a, const RankedCandidate& b) {
              return a.score < b.score;
            });

  std::vector<GridCell> candidates;
  std::vector<RankedCandidate> used;
  const int limit = std::max(1, params.return_to_global_max_candidates);
  for (const auto& c : ranked) {
    const bool duplicate =
        std::any_of(candidates.begin(), candidates.end(),
                    [&c](const GridCell& existing) { return existing == c.cell; });
    if (duplicate) continue;
    candidates.push_back(c.cell);
    used.push_back(c);
    if (static_cast<int>(candidates.size()) >= limit) break;
  }

  auto path = aStarBranch(blocked, g.width, g.height, start, candidates,
                          max_expand, time_budget_s);
  if (path.size() < 2) return result;

  const GridCell end = path.back();
  auto match = std::find_if(used.begin(), used.end(),
                            [&end](const RankedCandidate& c) {
                              return c.cell == end;
                            });
  if (match == used.end()) return result;

  result.found = true;
  result.rejoin_idx = match->idx;
  result.grid_path = std::move(path);
  result.world_path = samplePath(result.grid_path, g, 0.20);
  appendGlobalTail(result.world_path, global, result.rejoin_idx + 1,
                   params.lookahead_m);
  result.grid_path = worldPointsToGridPath(result.world_path, g);
  return result;
}

bool cellIsBlockedAt(const std::vector<uint8_t>& blocked, int width, int height,
                     int gx, int gy) {
  if (gx < 0 || gy < 0 || gx >= width || gy >= height) return true;
  return blocked[static_cast<std::size_t>(gy) * static_cast<std::size_t>(width) +
                 static_cast<std::size_t>(gx)] != 0;
}

void clearBlockedCircle(const OccupancyView& g, std::vector<uint8_t>& blocked,
                        GridCell center, double radius_m) {
  if (radius_m <= 0.0 || blocked.empty()) return;
  const int radius_cells = std::max(
      0, static_cast<int>(std::ceil(radius_m / std::max(1e-6, g.resolution_m))));
  const int r2 = radius_cells * radius_cells;
  for (int dy = -radius_cells; dy <= radius_cells; ++dy) {
    for (int dx = -radius_cells; dx <= radius_cells; ++dx) {
      if (dx * dx + dy * dy > r2) continue;
      const int gx = center.x + dx;
      const int gy = center.y + dy;
      if (!g.in_bounds(gx, gy)) continue;
      blocked[static_cast<std::size_t>(gy) * static_cast<std::size_t>(g.width) +
              static_cast<std::size_t>(gx)] = 0;
    }
  }
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
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(tf_buffer_);
  pnh_.param("target_frame", target_frame_, target_frame_);
  pnh_.param("tf_wait_s", tf_wait_s_, tf_wait_s_);
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
    output_local_path_topic_            = "/planning/local_path";
    output_path_mode_topic_             = "/planning/path_mode";
    output_avoidance_path_topic_        = "/planning/avoidance_path";
    output_path_history_topic_          = "/planning/path_history";
    output_travel_history_topic_        = "/planning/travel_history";
    output_travel_history_path_topic_   = "/planning/travel_history_path";
    output_recognized_obstacles_topic_  = "/planning/recognized_obstacles";
    output_blocking_obstacles_topic_    = "/planning/blocking_obstacles";
    output_global_overlay_topic_        = "/planning/global_obstacle_overlay";
    output_debug_text_topic_            = "/planning/local_replanner_debug_text";
    output_explainability_topic_        = "/planning/explainability";
  }
  pnh_.param("local_path_topic", output_local_path_topic_, output_local_path_topic_);
  pnh_.param("path_mode_topic",  output_path_mode_topic_,  output_path_mode_topic_);
  pnh_.param("avoidance_path_topic",        output_avoidance_path_topic_,        output_avoidance_path_topic_);
  pnh_.param("path_history_topic",          output_path_history_topic_,          output_path_history_topic_);
  pnh_.param("travel_history_topic",        output_travel_history_topic_,        output_travel_history_topic_);
  pnh_.param("travel_history_path_topic",   output_travel_history_path_topic_,   output_travel_history_path_topic_);
  pnh_.param("recognized_obstacles_topic",  output_recognized_obstacles_topic_,  output_recognized_obstacles_topic_);
  pnh_.param("blocking_obstacles_topic",    output_blocking_obstacles_topic_,    output_blocking_obstacles_topic_);
  pnh_.param("global_obstacle_overlay_topic", output_global_overlay_topic_,      output_global_overlay_topic_);
  pnh_.param("debug_text_topic",            output_debug_text_topic_,            output_debug_text_topic_);
  pnh_.param("explainability_topic",        output_explainability_topic_,        output_explainability_topic_);

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
  pnh_.param("return_to_global_trigger_distance_m",
             params_.return_to_global_trigger_distance_m,
             params_.return_to_global_trigger_distance_m);
  pnh_.param("return_to_global_min_tail_m",
             params_.return_to_global_min_tail_m,
             params_.return_to_global_min_tail_m);
  pnh_.param("return_to_global_goal_tolerance_m",
             params_.return_to_global_goal_tolerance_m,
             params_.return_to_global_goal_tolerance_m);
  pnh_.param("return_to_global_max_candidates",
             params_.return_to_global_max_candidates,
             params_.return_to_global_max_candidates);
  pnh_.param("candidate_start_clearance_m",
             params_.candidate_start_clearance_m,
             params_.candidate_start_clearance_m);

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
  pnh_.param("locked_static_ttl_s", smc.locked_static_ttl_s,
             smc.locked_static_ttl_s);
  pnh_.param("hold_escape_timeout_s", smc.hold_escape_timeout_s,
             smc.hold_escape_timeout_s);
  pnh_.param("hold_without_candidate", smc.hold_without_candidate,
             smc.hold_without_candidate);
  sm_ = std::make_unique<AvoidanceStateMachine>(smc);

  pub_local_path_           = nh_.advertise<nav_msgs::Path>(output_local_path_topic_, 2, true);
  pub_path_mode_            = nh_.advertise<std_msgs::String>(output_path_mode_topic_, 4, true);
  pub_avoidance_path_       = nh_.advertise<nav_msgs::Path>(output_avoidance_path_topic_, 2, true);
  pub_path_history_         = nh_.advertise<visualization_msgs::MarkerArray>(output_path_history_topic_, 2, true);
  pub_travel_history_       = nh_.advertise<visualization_msgs::Marker>(output_travel_history_topic_, 2, true);
  pub_travel_history_path_  = nh_.advertise<nav_msgs::Path>(output_travel_history_path_topic_, 2, true);
  pub_recognized_obstacles_ = nh_.advertise<visualization_msgs::MarkerArray>(output_recognized_obstacles_topic_, 2);
  pub_blocking_obstacles_   = nh_.advertise<visualization_msgs::MarkerArray>(output_blocking_obstacles_topic_, 2);
  pub_global_overlay_       = nh_.advertise<nav_msgs::OccupancyGrid>(output_global_overlay_topic_, 1);
  pub_debug_text_           = nh_.advertise<std_msgs::String>(output_debug_text_topic_, 5);
  pub_explainability_       = nh_.advertise<dynamic_window_approach::ExplainabilityEvent>(output_explainability_topic_, 20);
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

  // Resolve sensor-frame -> planner-frame transform at the cloud stamp.
  // Fall back to the latest known transform if the exact stamp is not yet
  // in the buffer.
  geometry_msgs::TransformStamped tf;
  try {
    tf = tf_buffer_.lookupTransform(target_frame_, msg->header.frame_id,
                                    msg->header.stamp,
                                    ros::Duration(tf_wait_s_));
  } catch (const tf2::TransformException&) {
    try {
      tf = tf_buffer_.lookupTransform(target_frame_, msg->header.frame_id,
                                      ros::Time(0));
    } catch (const tf2::TransformException& ex2) {
      ROS_WARN_THROTTLE(1.0, "cpp planner: TF %s -> %s unavailable: %s",
                        msg->header.frame_id.c_str(), target_frame_.c_str(),
                        ex2.what());
      return;
    }
  }

  // Build a 2D rotation/translation from the transform — we drop z later.
  const double tx = tf.transform.translation.x;
  const double ty = tf.transform.translation.y;
  const double tz = tf.transform.translation.z;
  tf2::Quaternion q(tf.transform.rotation.x, tf.transform.rotation.y,
                    tf.transform.rotation.z, tf.transform.rotation.w);
  tf2::Matrix3x3 R(q);

  pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
  pcl::fromROSMsg(*msg, pcl_cloud);
  std::vector<WorldXY> pts;
  pts.reserve(pcl_cloud.size() / 4);
  for (const auto& p : pcl_cloud.points) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
    // z gating is in sensor frame (height above sensor); apply before
    // transform so the gating bounds match Python's behaviour.
    if (p.z < cloud_z_min_ || p.z > cloud_z_max_) continue;
    const double wx = R[0][0] * p.x + R[0][1] * p.y + R[0][2] * p.z + tx;
    const double wy = R[1][0] * p.x + R[1][1] * p.y + R[1][2] * p.z + ty;
    // Also drop ground-plane returns after transform: anything below ~5 cm
    // above the world floor is likely ground.
    const double wz = R[2][0] * p.x + R[2][1] * p.y + R[2][2] * p.z + tz;
    if (wz < 0.05) continue;
    pts.push_back(WorldXY{wx, wy});
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
  const auto t_tick_start = std::chrono::steady_clock::now();
  const double now_sec = ros::Time::now().toSec();
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
    const double tick_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_tick_start).count();
    ROS_INFO_THROTTLE(1.0,
        "cpp planner | skipped=missing_inputs grid=%d odom=%d global=%d gpts=%zu tick=%.1fms",
        static_cast<int>(grid != nullptr), static_cast<int>(odom != nullptr),
        static_cast<int>(global != nullptr),
        global ? global->poses.size() : 0u, tick_ms);
    return;
  }

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
  const bool start_blocked =
      cellIsBlockedAt(blocked, g.width, g.height, start.x, start.y);
  auto candidate_blocked = blocked;
  clearBlockedCircle(g, candidate_blocked, start,
                     params_.candidate_start_clearance_m);

  // Truncate the global path to the lookahead window from the robot.
  const std::size_t near_idx = nearestIndex(global_world, rx, ry);
  double near_dist = std::numeric_limits<double>::infinity();
  if (!global_world.empty()) {
    near_dist = std::hypot(global_world[near_idx].x - rx,
                           global_world[near_idx].y - ry);
  }
  const double goal_dist = std::hypot(global_world.back().x - rx,
                                      global_world.back().y - ry);
  std::vector<WorldXY> nominal_world =
      truncateWorld(global_world, near_idx, params_.lookahead_m);
  double nominal_tail_m = polylineLengthM(nominal_world);
  bool return_to_global = false;
  const bool short_tail =
      nominal_tail_m < params_.return_to_global_min_tail_m &&
      goal_dist > params_.return_to_global_goal_tolerance_m;
  const bool off_global_path =
      near_dist > params_.return_to_global_trigger_distance_m;
  if (short_tail || off_global_path) {
    ReturnPath return_path = buildReturnToGlobalPath(
        global_world, candidate_blocked, g, start, rx, ry, params_,
        branch_max_expand_, branch_time_budget_s_);
    if (return_path.found && return_path.world_path.size() >= 2) {
      nominal_world = std::move(return_path.world_path);
      nominal_tail_m = polylineLengthM(nominal_world);
      return_to_global = true;
      ROS_INFO_THROTTLE(
          1.0,
          "cpp planner | returning to global path rejoin_idx=%zu near_idx=%zu "
          "near_dist=%.2f goal_dist=%.2f tail=%.2f",
          return_path.rejoin_idx, near_idx, near_dist, goal_dist, nominal_tail_m);
    }
  }
  const auto nominal_cells = worldPointsToGridPath(nominal_world, g);
  if (nominal_cells.size() < 2) {
    const double tick_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_tick_start).count();
    ROS_INFO_THROTTLE(1.0,
        "cpp planner | skipped=at_goal_or_short_nominal near_idx=%zu "
        "nominal_cells=%zu near_dist=%.2f goal_dist=%.2f tail=%.2f tick=%.1fms",
        near_idx, nominal_cells.size(), near_dist, goal_dist, nominal_tail_m,
        tick_ms);
    return;
  }

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
      sm_->recordStaticHit(cl.centroid, now_sec);
    }
    sm_->pruneStaleStatic(now_sec);
  }

  // Try sidestep first, then branch search as a fallback.
  AvoidanceResult candidate;
  std::vector<GridCell> candidate_cells;
  std::vector<WorldXY>  candidate_world;
  if (nominal_blocked) {
    candidate = buildSidestepAvoidance(nominal_cells, candidate_blocked, g, start,
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
        if (cellIsBlockedAt(candidate_blocked, g.width, g.height,
                            nominal_cells[i].x, nominal_cells[i].y)) continue;
        rejoin_candidates.push_back(nominal_cells[i]);
        if (static_cast<int>(rejoin_candidates.size()) >=
            branch_max_rejoin_candidates_) break;
      }
      if (!rejoin_candidates.empty()) {
        candidate_cells = aStarBranch(candidate_blocked, g.width, g.height, start,
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
      if (cellIsBlockedAt(candidate_blocked, g.width, g.height, c.x, c.y)) {
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
      now_sec);

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

  // ---- ExplainabilityEvent on mode transitions (XAI parity with Python) ----
  if (first_mode_publish_ || mode != last_published_mode_) {
    dynamic_window_approach::ExplainabilityEvent ev;
    ev.header.stamp = out.header.stamp;
    ev.header.frame_id = grid->header.frame_id;
    ev.source_node = "constrained_local_replanner_cpp";
    ev.decision_layer = "local_planner";
    ev.local_planning_active = (mode == PathMode::FOLLOW_AVOIDANCE);
    ev.stop_commanded = (mode == PathMode::HOLD);
    ev.slowdown_commanded = false;
    ev.avoid_direction = (candidate.side > 0) ? "right" :
                          (candidate.side < 0) ? "left" : "";
    if (mode == PathMode::FOLLOW_AVOIDANCE) {
      ev.event_type = "AVOIDANCE_ACTIVE";
      ev.trigger_reason = "predicted_overlap";
      ev.action_taken = "follow_avoidance";
      ev.summary_text = "C++ planner entered avoidance: blocker detected on nominal path";
    } else if (mode == PathMode::HOLD) {
      ev.event_type = "HOLD";
      ev.trigger_reason = "no_valid_avoidance_candidate";
      ev.action_taken = "stop";
      ev.summary_text = "C++ planner holding: nominal blocked but no detour found";
    } else {
      ev.event_type = "FOLLOW_LOCAL";
      ev.trigger_reason = "nominal_clear";
      ev.action_taken = "follow_local";
      ev.summary_text = "C++ planner tracking nominal global path";
    }
    ev.closest_obstacle_dist_m =
        clusters.empty() ? -1.0f
                          : static_cast<float>(std::hypot(
                                blocker_world.x - rx, blocker_world.y - ry));
    ev.tracked_object_id = -1;
    pub_explainability_.publish(ev);
    last_published_mode_ = mode;
    first_mode_publish_ = false;
  }

  // ---- visualization mirror topics (RViz parity with Python) ----
  const std::string frame = grid->header.frame_id;
  const ros::Time now_stamp = out.header.stamp;

  // /planning/avoidance_path: same content as local_path while
  // FOLLOW_AVOIDANCE, empty otherwise.
  nav_msgs::Path avoid_msg;
  avoid_msg.header = out.header;
  if (mode == PathMode::FOLLOW_AVOIDANCE) avoid_msg.poses = out.poses;
  pub_avoidance_path_.publish(avoid_msg);

  // /planning/recognized_obstacles + /planning/blocking_obstacles markers.
  pub_recognized_obstacles_.publish(buildRecognizedObstaclesMarkers(
      sm_->lockedList(), clusters, frame, now_stamp,
      /*sphere_scale_m*/ 0.18, /*lifetime_s*/ 0.8));
  pub_blocking_obstacles_.publish(buildBlockingObstacleMarkers(
      blocker_world, /*active*/ nominal_blocked, frame, now_stamp,
      /*radius_m*/ params_.obstacle_block_margin_m + 0.20,
      /*lifetime_s*/ 0.8));

  // /planning/global_obstacle_overlay republishes the blocked mask so the
  // existing RViz overlay panel keeps working.
  pub_global_overlay_.publish(buildOverlayGrid(g, blocked, frame, now_stamp));

  // /planning/path_history: the current local path as a strip marker.
  std::vector<WorldXY> current_path;
  current_path.reserve(out.poses.size());
  for (const auto& ps : out.poses) {
    current_path.push_back(WorldXY{ps.pose.position.x, ps.pose.position.y});
  }
  pub_path_history_.publish(
      buildPathHistoryMarkers(current_path, frame, now_stamp));

  // /planning/travel_history + travel_history_path: accumulate the robot's
  // own positions over time.
  if (travel_history_.empty() ||
      std::hypot(rx - travel_history_.back().x, ry - travel_history_.back().y)
          > 0.10) {
    travel_history_.push_back(WorldXY{rx, ry});
    if (travel_history_.size() > 2000) {
      travel_history_.erase(travel_history_.begin(),
                            travel_history_.begin() + 500);
    }
  }
  {
    visualization_msgs::Marker tr;
    tr.header.frame_id = frame;
    tr.header.stamp = now_stamp;
    tr.ns = "travel_history";
    tr.id = 0;
    tr.type = visualization_msgs::Marker::LINE_STRIP;
    tr.action = visualization_msgs::Marker::ADD;
    tr.scale.x = 0.05;
    tr.color.a = 0.85;
    tr.color.r = 0.85;
    tr.color.g = 0.85;
    tr.color.b = 0.0;
    tr.pose.orientation.w = 1.0;
    for (const auto& p : travel_history_) {
      geometry_msgs::Point pt;
      pt.x = p.x;
      pt.y = p.y;
      pt.z = 0.02;
      tr.points.push_back(pt);
    }
    pub_travel_history_.publish(tr);

    nav_msgs::Path tp;
    tp.header.frame_id = frame;
    tp.header.stamp = now_stamp;
    for (const auto& p : travel_history_) {
      geometry_msgs::PoseStamped ps;
      ps.header = tp.header;
      ps.pose.position.x = p.x;
      ps.pose.position.y = p.y;
      ps.pose.orientation.w = 1.0;
      tp.poses.push_back(ps);
    }
    pub_travel_history_path_.publish(tp);
  }

  pub_debug_text_.publish(buildDebugText(
      mode, nominal_blocked, candidate.found, cached_drivable, endpoint_dist,
      clusters.size(), obstacle_pts.size()));

  const double tick_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t_tick_start)
          .count();
  ROS_INFO_THROTTLE(1.0,
      "cpp planner | mode=%s nominal_blocked=%d candidate=%d cached_drivable=%d "
      "endpoint=%.2f locked_static=%d start_blocked=%d return_global=%d near_dist=%.2f "
      "tail=%.2f clusters=%zu obs_pts=%zu tick=%.1fms",
      mode_msg.data.c_str(), static_cast<int>(nominal_blocked),
      static_cast<int>(candidate.found), static_cast<int>(cached_drivable),
      endpoint_dist, static_cast<int>(locked_static_nearby),
      static_cast<int>(start_blocked), static_cast<int>(return_to_global),
      near_dist, nominal_tail_m,
      clusters.size(), obstacle_pts.size(), tick_ms);
}

}  // namespace clr
