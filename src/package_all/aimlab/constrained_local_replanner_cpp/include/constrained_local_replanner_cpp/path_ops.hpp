#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "constrained_local_replanner_cpp/types.hpp"

namespace clr {

// Bresenham line traversal between two grid cells. Visits every cell along
// the line; returns true if every cell is unblocked. `blocked` is row-major
// width*height. Used for line-of-sight in branch search.
bool hasLineOfSight(const std::vector<uint8_t>& blocked, int width, int height,
                    GridCell a, GridCell b);

// Returns true if any cell on `grid_path` is blocked within `max_check_m`
// arc-length of `start_cell`. This is the C++ port of the hot inner check
// the Python replanner runs every tick on every candidate path.
bool pathBlockedAhead(const std::vector<GridCell>& grid_path,
                      const std::vector<uint8_t>& blocked,
                      int width, int height,
                      GridCell start_cell, double resolution_m,
                      double max_check_m);

// Minimum perpendicular distance (in meters) from a robot position `(rx, ry)`
// to the polyline traced by `grid_path` on a grid with the given resolution
// and origin. Used by the avoidance-reuse and endpoint-distance guards.
double gridPathMinDistanceToXY(const std::vector<GridCell>& grid_path,
                               const OccupancyView& g, double rx, double ry);

// Distance from the robot to the cached path's last cell. Used by the
// "robot has passed the detour endpoint" release gate.
double gridPathEndpointDistanceToXY(const std::vector<GridCell>& grid_path,
                                    const OccupancyView& g, double rx, double ry);

// Convert a chain of world points to a sequence of distinct grid cells.
std::vector<GridCell> worldPointsToGridPath(const std::vector<WorldXY>& world_points,
                                            const OccupancyView& g);

// Approximate path length in meters.
double polylineLengthM(const std::vector<WorldXY>& pts);

}  // namespace clr
