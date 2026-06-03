#pragma once

#include <cstdint>
#include <vector>

#include "constrained_local_replanner_cpp/types.hpp"

namespace clr {

// Grid A* from `start` to any of `rejoin_candidates`. Returns the cells along
// the cheapest collision-free path, including start and the chosen rejoin
// cell. Empty result means no path found within `max_expand` expansions.
//
// The cost function is 8-connected unit step (1.0 / sqrt(2)). `blocked` is
// the row-major width*height byte mask used elsewhere in this package; a
// cell is impassable iff blocked[y*width + x] != 0 or out-of-bounds.
std::vector<GridCell> aStarBranch(const std::vector<uint8_t>& blocked,
                                  int width, int height,
                                  GridCell start,
                                  const std::vector<GridCell>& rejoin_candidates,
                                  int max_expand,
                                  double time_budget_s);

}  // namespace clr
