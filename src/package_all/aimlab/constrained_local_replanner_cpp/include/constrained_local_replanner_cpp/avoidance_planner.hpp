#pragma once

#include <vector>

#include "constrained_local_replanner_cpp/types.hpp"

namespace clr {

struct AvoidanceResult {
  // World-frame waypoints (sampled along the chosen detour) and the
  // corresponding grid path for downstream collision/line-of-sight reuse.
  std::vector<WorldXY> waypoints;
  std::vector<GridCell> grid_path;
  bool found{false};
  // Side: -1 = left, +1 = right, 0 = none.
  int side{0};
  double max_curvature{0.0};
};

// Build a single sidestep candidate around an obstacle blocking the nominal
// path. This is the C++ port of `_build_sidestep_avoidance_path` from the
// Python replanner. The branch (`A*`-style) search is not yet ported.
//
// `nominal_path` is the global path projected into grid cells, in the same
// grid as `g` / `blocked`. `start_cell` is the robot's projected cell.
// `obstacle_world` is the centroid of the blocking obstacle in world frame.
// `preferred_direction` mirrors the Python override (-1=left, +1=right, 0=auto).
AvoidanceResult buildSidestepAvoidance(const std::vector<GridCell>& nominal_path,
                                       const std::vector<uint8_t>& blocked,
                                       const OccupancyView& g,
                                       GridCell start_cell,
                                       WorldXY obstacle_world,
                                       const PlannerParams& params,
                                       double robot_yaw,
                                       int preferred_direction = 0);

}  // namespace clr
