#pragma once

#include <vector>

#include "constrained_local_replanner_cpp/types.hpp"

namespace nav_msgs {
class OccupancyGrid;
}

namespace clr {

// Pack a ROS OccupancyGrid into the planner's lightweight view.
OccupancyView fromOccupancyGrid(const nav_msgs::OccupancyGrid& msg);

// Treat `cost > occ_threshold` as blocked. If `unknown_is_occupied=true`
// then cells with -1 are also blocked.
inline bool cellIsBlocked(const OccupancyView& g, int gx, int gy,
                          bool unknown_is_occupied, int8_t occ_threshold = 50) {
  if (!g.in_bounds(gx, gy)) return true;
  const int8_t v = g.at(gx, gy);
  if (v < 0) return unknown_is_occupied;
  return v >= occ_threshold;
}

inline bool cellIsDrivableFree(const OccupancyView& g, int gx, int gy,
                               bool unknown_is_occupied, int8_t occ_threshold = 50) {
  return !cellIsBlocked(g, gx, gy, unknown_is_occupied, occ_threshold);
}

// Overlay points (cluster centroids or raw obstacle hits) onto a mutable
// blocked-grid mask. `blocked` is sized to match `g`, byte-per-cell.
void overlayPoints(const OccupancyView& g, std::vector<uint8_t>& blocked,
                   const std::vector<WorldXY>& points, double inflate_m);

// Build a binary blocked mask from the occupancy view itself (no overlay).
std::vector<uint8_t> baseBlockedMask(const OccupancyView& g, bool unknown_is_occupied,
                                     int8_t occ_threshold = 50);

}  // namespace clr
