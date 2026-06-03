#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

namespace clr {

// Integer grid cell.
struct GridCell {
  int x{0};
  int y{0};
  bool operator==(const GridCell& o) const noexcept { return x == o.x && y == o.y; }
  bool operator!=(const GridCell& o) const noexcept { return !(*this == o); }
};

// 2D world-frame point (map frame).
struct WorldXY {
  double x{0.0};
  double y{0.0};
};

// 2D point cloud cluster summary.
struct Cluster {
  WorldXY centroid;
  WorldXY min_xy;
  WorldXY max_xy;
  std::size_t point_count{0};
};

// Robot footprint used for collision / path-blocked checks.
struct RobotFootprint {
  double half_length_m{0.30};   // along forward axis
  double half_width_m{0.29};    // along lateral axis
  double padding_m{0.0};        // extra inflation
  double block_margin_m{0.18};  // additional radius for path-blocking decisions
};

// Local-replanner runtime parameters mirrored from the Python defaults that
// matter for the C++ core. Only the fields the C++ code reads are listed —
// the long tail of trigger/debounce/visual params stays in Python until they
// are ported.
struct PlannerParams {
  double lookahead_m{8.5};
  double avoidance_trigger_ahead_m{8.5};
  double obstacle_block_margin_m{0.18};
  double sidestep_min_offset_m{0.52};
  double sidestep_max_offset_m{1.15};
  double sidestep_preview_m{2.0};
  double sidestep_forward_margin_m{0.55};
  double rejoin_min_distance_m{1.4};
  double avoidance_keep_until_endpoint_distance_m{0.30};
  double pointcloud_cluster_resolution_m{0.20};

  int max_expand{4500};
  int branch_max_rejoin_candidates{12};
  double branch_time_budget_s{0.45};
  bool grid_unknown_is_occupied{true};

  RobotFootprint footprint{};
};

// Cached drivable grid view (occupancy + metadata) extracted from nav_msgs::OccupancyGrid.
struct OccupancyView {
  int width{0};
  int height{0};
  double resolution_m{0.05};
  WorldXY origin{};       // origin of cell (0,0) in world frame
  std::vector<int8_t> data; // row-major (gy * width + gx), -1=unknown, 0..100=cost

  inline bool in_bounds(int gx, int gy) const noexcept {
    return gx >= 0 && gy >= 0 && gx < width && gy < height;
  }
  inline int8_t at(int gx, int gy) const noexcept {
    return data[static_cast<std::size_t>(gy) * static_cast<std::size_t>(width) +
                static_cast<std::size_t>(gx)];
  }
  inline GridCell world_to_cell(double wx, double wy) const noexcept {
    return GridCell{
        static_cast<int>(std::floor((wx - origin.x) / resolution_m)),
        static_cast<int>(std::floor((wy - origin.y) / resolution_m)),
    };
  }
  inline WorldXY cell_to_world(int gx, int gy) const noexcept {
    return WorldXY{
        origin.x + (static_cast<double>(gx) + 0.5) * resolution_m,
        origin.y + (static_cast<double>(gy) + 0.5) * resolution_m,
    };
  }
};

}  // namespace clr
