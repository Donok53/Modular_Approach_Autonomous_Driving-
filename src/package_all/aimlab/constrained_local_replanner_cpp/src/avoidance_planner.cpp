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
  } else if (obs_ly > 0.0) {
    side_order = {{-1, +1}};  // obstacle on left -> sidestep right
  } else {
    side_order = {{+1, -1}};
  }

  const double start_x = std::max(0.0, obs_lx - 1.20);
  const double preview_x =
      std::max(params.sidestep_preview_m,
               obs_lx + params.sidestep_forward_margin_m + params.footprint.half_length_m * 2.0);
  // Lateral clearance the path centerline needs from the obstacle centroid.
  // The overlay already inflates obstacles by (half_width + block_margin),
  // so we only add a small safety nudge here — the previous +0.08 m made the
  // sidestep wider than necessary without buying real safety.
  const double clearance_y =
      params.footprint.half_width_m + params.footprint.padding_m +
      params.obstacle_block_margin_m + 0.02;

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
      const double target_y = static_cast<double>(side) * offset_m;

      const double entry_x = std::max(start_x + 0.15, obs_lx - 0.20);
      const double pass_x = std::max(entry_x + 0.35,
                                     obs_lx + params.sidestep_forward_margin_m);
      // Rejoin curve: after passing the obstacle the path must come back to
      // the nominal corridor (y = 0). Without this the candidate "stays
      // wide" for the entire preview window and the robot looks like it is
      // taking a huge detour.
      const double rejoin_x = pass_x + params.rejoin_min_distance_m;
      const double end_x = std::max(preview_x, rejoin_x + 0.30);
      const double mid_x = start_x + 0.5 * std::max(0.20, entry_x - start_x);

      const std::vector<std::pair<double, double>> waypts_local{
          {start_x, 0.0},
          {mid_x, 0.0},
          {entry_x, 0.35 * target_y},
          {0.5 * (entry_x + pass_x), 0.80 * target_y},
          {pass_x, target_y},
          {0.5 * (pass_x + rejoin_x), 0.60 * target_y},
          {rejoin_x, 0.0},
          {end_x, 0.0},
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
      const double score = penalty_side * 10.0 + offset_m + 0.05 * max_heading_delta;
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
