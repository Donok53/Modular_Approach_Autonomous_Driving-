#include "constrained_local_replanner_cpp/grid.hpp"

#include <cmath>

#include <nav_msgs/OccupancyGrid.h>

namespace clr {

OccupancyView fromOccupancyGrid(const nav_msgs::OccupancyGrid& msg) {
  OccupancyView view;
  view.width = static_cast<int>(msg.info.width);
  view.height = static_cast<int>(msg.info.height);
  view.resolution_m = static_cast<double>(msg.info.resolution);
  view.origin.x = msg.info.origin.position.x;
  view.origin.y = msg.info.origin.position.y;
  view.data = msg.data;  // copy: small enough at 5 cm res, simpler ownership
  return view;
}

std::vector<uint8_t> baseBlockedMask(const OccupancyView& g, bool unknown_is_occupied,
                                     int8_t occ_threshold) {
  const std::size_t n = static_cast<std::size_t>(g.width) * static_cast<std::size_t>(g.height);
  std::vector<uint8_t> mask(n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    const int8_t v = g.data[i];
    if (v < 0) {
      mask[i] = unknown_is_occupied ? 1 : 0;
    } else {
      mask[i] = (v >= occ_threshold) ? 1 : 0;
    }
  }
  return mask;
}

void overlayPoints(const OccupancyView& g, std::vector<uint8_t>& blocked,
                   const std::vector<WorldXY>& points, double inflate_m) {
  if (g.width <= 0 || g.height <= 0 || blocked.empty()) return;
  const int radius_cells = std::max(0,
      static_cast<int>(std::ceil(inflate_m / std::max(1e-6, g.resolution_m))));
  const int r2 = radius_cells * radius_cells;
  for (const auto& p : points) {
    const GridCell c = g.world_to_cell(p.x, p.y);
    for (int dy = -radius_cells; dy <= radius_cells; ++dy) {
      for (int dx = -radius_cells; dx <= radius_cells; ++dx) {
        if (dx * dx + dy * dy > r2) continue;
        const int gx = c.x + dx;
        const int gy = c.y + dy;
        if (!g.in_bounds(gx, gy)) continue;
        blocked[static_cast<std::size_t>(gy) * static_cast<std::size_t>(g.width) +
                static_cast<std::size_t>(gx)] = 1;
      }
    }
  }
}

}  // namespace clr
