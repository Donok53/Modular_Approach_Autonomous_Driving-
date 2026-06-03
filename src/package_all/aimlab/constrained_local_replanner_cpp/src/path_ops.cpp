#include "constrained_local_replanner_cpp/path_ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace clr {

namespace {
inline bool blockedAt(const std::vector<uint8_t>& blocked, int width, int height,
                      int gx, int gy) {
  if (gx < 0 || gy < 0 || gx >= width || gy >= height) return true;
  return blocked[static_cast<std::size_t>(gy) * static_cast<std::size_t>(width) +
                 static_cast<std::size_t>(gx)] != 0;
}
}  // namespace

bool hasLineOfSight(const std::vector<uint8_t>& blocked, int width, int height,
                    GridCell a, GridCell b) {
  // Standard integer Bresenham. Treats every visited cell as a hit.
  int x0 = a.x, y0 = a.y, x1 = b.x, y1 = b.y;
  const int dx = std::abs(x1 - x0);
  const int dy = -std::abs(y1 - y0);
  const int sx = (x0 < x1) ? 1 : -1;
  const int sy = (y0 < y1) ? 1 : -1;
  int err = dx + dy;
  while (true) {
    if (blockedAt(blocked, width, height, x0, y0)) return false;
    if (x0 == x1 && y0 == y1) return true;
    const int e2 = 2 * err;
    if (e2 >= dy) { err += dy; x0 += sx; }
    if (e2 <= dx) { err += dx; y0 += sy; }
  }
}

bool pathBlockedAhead(const std::vector<GridCell>& grid_path,
                      const std::vector<uint8_t>& blocked,
                      int width, int height,
                      GridCell start_cell, double resolution_m,
                      double max_check_m) {
  if (grid_path.empty()) return false;
  // Find the path index nearest to start_cell — that's the "robot's" anchor.
  std::size_t nearest = 0;
  long best = std::numeric_limits<long>::max();
  for (std::size_t i = 0; i < grid_path.size(); ++i) {
    const long dx = grid_path[i].x - start_cell.x;
    const long dy = grid_path[i].y - start_cell.y;
    const long d2 = dx * dx + dy * dy;
    if (d2 < best) { best = d2; nearest = i; }
  }
  // Walk forward accumulating arc length; stop when we either hit a blocked
  // cell or exceed max_check_m. Distance is cells_traversed * resolution.
  double traveled_m = 0.0;
  if (blockedAt(blocked, width, height, grid_path[nearest].x, grid_path[nearest].y))
    return true;
  for (std::size_t i = nearest + 1; i < grid_path.size(); ++i) {
    const double seg_m = std::hypot(
        static_cast<double>(grid_path[i].x - grid_path[i - 1].x),
        static_cast<double>(grid_path[i].y - grid_path[i - 1].y)) * resolution_m;
    traveled_m += seg_m;
    if (traveled_m > max_check_m) return false;
    if (blockedAt(blocked, width, height, grid_path[i].x, grid_path[i].y)) return true;
  }
  return false;
}

double gridPathMinDistanceToXY(const std::vector<GridCell>& grid_path,
                               const OccupancyView& g, double rx, double ry) {
  double best = std::numeric_limits<double>::infinity();
  for (const auto& c : grid_path) {
    const WorldXY w = g.cell_to_world(c.x, c.y);
    const double d = std::hypot(w.x - rx, w.y - ry);
    if (d < best) best = d;
  }
  return best;
}

double gridPathEndpointDistanceToXY(const std::vector<GridCell>& grid_path,
                                    const OccupancyView& g, double rx, double ry) {
  if (grid_path.empty()) return std::numeric_limits<double>::infinity();
  const WorldXY w = g.cell_to_world(grid_path.back().x, grid_path.back().y);
  return std::hypot(w.x - rx, w.y - ry);
}

std::vector<GridCell> worldPointsToGridPath(const std::vector<WorldXY>& world_points,
                                            const OccupancyView& g) {
  std::vector<GridCell> out;
  out.reserve(world_points.size());
  for (const auto& p : world_points) {
    const GridCell c = g.world_to_cell(p.x, p.y);
    if (out.empty() || c != out.back()) out.push_back(c);
  }
  return out;
}

double polylineLengthM(const std::vector<WorldXY>& pts) {
  double total = 0.0;
  for (std::size_t i = 1; i < pts.size(); ++i) {
    total += std::hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
  }
  return total;
}

}  // namespace clr
