#pragma once

#include <vector>

#include "constrained_local_replanner_cpp/types.hpp"

namespace clr {

// Simple voxel-grid 2D clustering: bucket points into resolution_m cells, then
// 4-connected merge. Returns one Cluster summary per blob. This replaces the
// O(n^2) merge the Python node does every tick.
std::vector<Cluster> clusterPoints2D(const std::vector<WorldXY>& points,
                                     double resolution_m);

// Voxel downsample to reduce 10k+ raw points to a tractable set before
// clustering. Returns one representative per resolution_m cell.
std::vector<WorldXY> voxelDownsample2D(const std::vector<WorldXY>& points,
                                       double resolution_m);

}  // namespace clr
