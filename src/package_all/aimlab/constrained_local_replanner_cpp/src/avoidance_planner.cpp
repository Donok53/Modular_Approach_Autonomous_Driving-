#include "constrained_local_replanner_cpp/avoidance_planner.hpp"

#define _USE_MATH_DEFINES
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "constrained_local_replanner_cpp/grid.hpp"
#include "constrained_local_replanner_cpp/path_ops.hpp"

namespace clr {

namespace {

// Quadratic Bezier-ish sampler used by the Python sidestep builder. We sample
// a control-point list to a dense waypoint list at ~resolution spacing.
std::vector<WorldXY> sampleWorldPoints(const std::vector<WorldXY>& ctrl,
                                       double step_m) {
  std::vector<WorldXY> out;
  if (ctrl.empty()) return out;
  out.push_back(ctrl.front());
  for (std::size_t i = 1; i < ctrl.size(); ++i) {
    const WorldXY& a = ctrl[i - 1];
    const WorldXY& b = ctrl[i];
    const double seg = std::hypot(b.x - a.x, b.y - a.y);
    const int n = std::max(1, static_cast<int>(std::ceil(seg / std::max(1e-3, step_m))));
    for (int k = 1; k <= n; ++k) {
      const double t = static_cast<double>(k) / static_cast<double>(n);
      out.push_back(WorldXY{a.x + t * (b.x - a.x), a.y + t * (b.y - a.y)});
    }
  }
  return out;
}

inline WorldXY localToWorld(double lx, double ly, double rx, double ry, double yaw) {
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  return WorldXY{rx + c * lx - s * ly, ry + s * lx + c * ly};
}

inline WorldXY worldToLocal(WorldXY p, double rx, double ry, double yaw) {
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  const double dx = p.x - rx;
  const double dy = p.y - ry;
  return WorldXY{c * dx + s * dy, -s * dx + c * dy};
}

std::vector<WorldXY> nominalPathWorld(const std::vector<GridCell>& nominal_path,
                                      const OccupancyView& g) {
  std::vector<WorldXY> out;
  out.reserve(nominal_path.size());
  for (const auto& cell : nominal_path) {
    out.push_back(g.cell_to_world(cell.x, cell.y));
  }
  return out;
}

std::vector<WorldXY> nominalPathLocal(const std::vector<WorldXY>& nominal_world,
                                      double rx,
                                      double ry,
                                      double yaw) {
  std::vector<WorldXY> out;
  out.reserve(nominal_world.size());
  for (const auto& p : nominal_world) {
    out.push_back(worldToLocal(p, rx, ry, yaw));
  }
  return out;
}

double nominalYAtX(const std::vector<WorldXY>& nominal_local, double lx) {
  if (nominal_local.empty()) return 0.0;
  double best_score = std::numeric_limits<double>::infinity();
  double best_y = nominal_local.front().y;

  for (std::size_t i = 1; i < nominal_local.size(); ++i) {
    const WorldXY a = nominal_local[i - 1];
    const WorldXY b = nominal_local[i];
    const double vx = b.x - a.x;
    const double vy = b.y - a.y;
    const double denom = vx * vx + vy * vy;
    double t = 0.0;
    if (denom > 1e-9) {
      t = std::max(0.0, std::min(1.0, ((lx - a.x) * vx) / denom));
    }
    const double px = a.x + t * vx;
    const double py = a.y + t * vy;
    const double score = std::abs(px - lx) + 0.05 * std::abs(t - 0.5);
    if (score < best_score) {
      best_score = score;
      best_y = py;
    }
  }
  return best_y;
}

double nominalSideHint(const std::vector<WorldXY>& nominal_local,
                       double from_x,
                       double to_x) {
  double weighted_sum = 0.0;
  double total_weight = 0.0;
  for (const auto& p : nominal_local) {
    if (p.x < from_x || p.x > to_x) continue;
    const double w = std::max(0.05, p.x - from_x + 0.05);
    weighted_sum += w * p.y;
    total_weight += w;
  }
  if (total_weight <= 1e-6) return nominalYAtX(nominal_local, to_x);
  return weighted_sum / total_weight;
}

double pointSegmentDistance(WorldXY p, WorldXY a, WorldXY b) {
  const double vx = b.x - a.x;
  const double vy = b.y - a.y;
  const double wx = p.x - a.x;
  const double wy = p.y - a.y;
  const double denom = vx * vx + vy * vy;
  if (denom <= 1e-9) return std::hypot(p.x - a.x, p.y - a.y);
  const double t = std::max(0.0, std::min(1.0, (wx * vx + wy * vy) / denom));
  const double px = a.x + t * vx;
  const double py = a.y + t * vy;
  return std::hypot(p.x - px, p.y - py);
}

double pointPolylineDistance(WorldXY p, const std::vector<WorldXY>& path) {
  if (path.empty()) return std::numeric_limits<double>::infinity();
  if (path.size() == 1) return std::hypot(p.x - path.front().x, p.y - path.front().y);
  double best = std::numeric_limits<double>::infinity();
  for (std::size_t i = 1; i < path.size(); ++i) {
    best = std::min(best, pointSegmentDistance(p, path[i - 1], path[i]));
  }
  return best;
}

double meanPolylineDistance(const std::vector<WorldXY>& pts,
                            const std::vector<WorldXY>& reference) {
  if (pts.empty() || reference.empty()) return 0.0;
  double sum = 0.0;
  for (const auto& p : pts) {
    sum += pointPolylineDistance(p, reference);
  }
  return sum / static_cast<double>(pts.size());
}

}  // namespace

AvoidanceResult buildSidestepAvoidance(const std::vector<GridCell>& nominal_path,
                                       const std::vector<uint8_t>& blocked,
                                       const OccupancyView& g,
                                       GridCell start_cell,
                                       WorldXY obstacle_world,
                                       const PlannerParams& params,
                                       double robot_yaw,
                                       int preferred_direction) {
  AvoidanceResult result;
  if (nominal_path.size() < 2) return result;

  // Local-frame x of the obstacle relative to the robot. The robot is at the
  // cell `start_cell`; we pull the world coordinate from the grid.
  const WorldXY robot_w = g.cell_to_world(start_cell.x, start_cell.y);
  const double dx = obstacle_world.x - robot_w.x;
  const double dy = obstacle_world.y - robot_w.y;
  const double c = std::cos(robot_yaw);
  const double s = std::sin(robot_yaw);
  const double obs_lx = c * dx + s * dy;
  const double obs_ly = -s * dx + c * dy;
  const std::vector<WorldXY> nominal_world = nominalPathWorld(nominal_path, g);
  const std::vector<WorldXY> nominal_local =
      nominalPathLocal(nominal_world, robot_w.x, robot_w.y, robot_yaw);

  if (obs_lx < 0.10) {
    // Obstacle is behind / next to the robot — sidestep is not appropriate.
    return result;
  }

  // Side preference: explicit override > obstacle bias > default right.
  std::array<int, 2> side_order{};
  if (preferred_direction < 0) {
    side_order = {{-1, +1}};
  } else if (preferred_direction > 0) {
    side_order = {{+1, -1}};
  } else {
    const double future_y = nominalSideHint(
        nominal_local, std::max(0.0, obs_lx - 0.20),
        obs_lx + std::max(0.60, params.rejoin_min_distance_m));
    if (std::abs(future_y) >= 0.12) {
      const int nominal_side = (future_y > 0.0) ? +1 : -1;
      side_order = {{nominal_side, -nominal_side}};
    } else if (obs_ly > 0.0) {
      side_order = {{-1, +1}};  // obstacle on left -> sidestep right
    } else {
      side_order = {{+1, -1}};
    }
  }

  const double start_x = std::max(0.0, obs_lx - 1.20);
  // Lateral clearance the path centerline needs from the obstacle centroid.
  // The overlay already inflates obstacles by (half_width + block_margin) so
  // the collision check below already enforces that buffer. Adding
  // block_margin again here double-counted the inflation and produced the
  // "way too wide" detour. Keep only the footprint + a small tracking nudge.
  const double clearance_y =
      params.footprint.half_width_m + params.footprint.padding_m +
      std::max(0.0, params.sidestep_clearance_extra_m);

  double best_score = std::numeric_limits<double>::infinity();
  for (const int side : side_order) {
    const double required_offset =
        (side > 0) ? std::max(0.0, obs_ly + clearance_y)
                   : std::max(0.0, -obs_ly + clearance_y);
    const std::array<double, 4> offsets{
        params.sidestep_min_offset_m,
        required_offset,
        required_offset + 0.15,
        required_offset + 0.30,
    };
    for (const double raw_off : offsets) {
      double offset_m = std::max(params.sidestep_min_offset_m, raw_off);
      if (offset_m > params.sidestep_max_offset_m + 1e-6) continue;
      double target_y = static_cast<double>(side) * offset_m;

      const double entry_lead = std::max(0.20, params.sidestep_entry_lead_m);
      const double entry_x = std::max(start_x + 0.15, obs_lx - entry_lead);
      const double pass_x = std::max(entry_x + 0.35,
                                     obs_lx + params.sidestep_forward_margin_m);
      // Rejoin curve: after passing the obstacle the path must come back to
      // the nominal corridor. Keep the early part close to the robot's current
      // heading; in doorways, following a sharply turning nominal path too
      // early clips the inner wall before the footprint is through the gap.
      const double rejoin_x = pass_x + params.rejoin_min_distance_m;
      // Keep the cached path short past the rejoin point. A long preview tail
      // makes the cached endpoint sit ~2 m ahead of the robot even after the
      // robot is already back on the nominal corridor, which prevented the
      // FOLLOW_AVOIDANCE release guard from firing.
      const double end_x = rejoin_x + 0.30;
      const double mid_x = start_x + 0.5 * std::max(0.20, entry_x - start_x);
      const double rejoin_y = nominalYAtX(nominal_local, rejoin_x);
      const double end_y = nominalYAtX(nominal_local, end_x);
      const double entry_target_y = 0.15 * target_y;
      const double pass_mid_y = 0.55 * target_y;
      const double rejoin_mid_y = target_y + 0.45 * (rejoin_y - target_y);

      const std::vector<std::pair<double, double>> waypts_local{
          {start_x, 0.0},
          {mid_x, 0.0},
          {entry_x, entry_target_y},
          {0.5 * (entry_x + pass_x), pass_mid_y},
          {pass_x, target_y},
          {0.5 * (pass_x + rejoin_x), rejoin_mid_y},
          {rejoin_x, rejoin_y},
          {end_x, end_y},
      };
      std::vector<WorldXY> ctrl;
      ctrl.reserve(waypts_local.size());
      for (const auto& wp : waypts_local) {
        ctrl.push_back(localToWorld(wp.first, wp.second, robot_w.x, robot_w.y, robot_yaw));
      }
      auto sampled = sampleWorldPoints(ctrl, std::max(0.10, g.resolution_m));
      auto grid_path = worldPointsToGridPath(sampled, g);
      if (grid_path.size() < 2) continue;

      // Collision check against the blocked mask (which already includes the
      // drivable grid + dynamic obstacle overlay).
      bool any_blocked = false;
      for (const auto& cell : grid_path) {
        if (!g.in_bounds(cell.x, cell.y)) { any_blocked = true; break; }
        const std::size_t idx = static_cast<std::size_t>(cell.y) *
                                static_cast<std::size_t>(g.width) +
                                static_cast<std::size_t>(cell.x);
        if (blocked[idx]) { any_blocked = true; break; }
      }
      if (any_blocked) continue;

      // Score: prefer the preferred side, then smaller lateral offset, then
      // smaller curvature. We approximate curvature with the max heading
      // delta between consecutive sampled segments.
      double max_heading_delta = 0.0;
      double prev_yaw = std::numeric_limits<double>::quiet_NaN();
      for (std::size_t i = 1; i < sampled.size(); ++i) {
        const double yaw = std::atan2(sampled[i].y - sampled[i - 1].y,
                                      sampled[i].x - sampled[i - 1].x);
        if (!std::isnan(prev_yaw)) {
          double dy_ang = yaw - prev_yaw;
          while (dy_ang > M_PI) dy_ang -= 2.0 * M_PI;
          while (dy_ang < -M_PI) dy_ang += 2.0 * M_PI;
          max_heading_delta = std::max(max_heading_delta, std::abs(dy_ang));
        }
        prev_yaw = yaw;
      }

      const double penalty_side = (side == side_order[0]) ? 0.0 : 1.0;
      const double mean_nominal_dist = meanPolylineDistance(sampled, nominal_world);
      const double end_nominal_dist = pointPolylineDistance(sampled.back(), nominal_world);
      const double score = penalty_side * 10.0 + offset_m +
                           1.40 * mean_nominal_dist +
                           2.20 * end_nominal_dist +
                           0.22 * max_heading_delta;
      if (score < best_score) {
        best_score = score;
        result.waypoints = std::move(sampled);
        result.grid_path = std::move(grid_path);
        result.found = true;
        result.side = side;
        result.max_curvature = max_heading_delta;
      }
      break;  // first feasible offset for this side wins
    }
  }
  return result;
}

}  // namespace clr
