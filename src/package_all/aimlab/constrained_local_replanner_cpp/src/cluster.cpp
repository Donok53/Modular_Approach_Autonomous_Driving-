#include "constrained_local_replanner_cpp/cluster.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace clr {

namespace {

struct CellKey {
  int x;
  int y;
  bool operator==(const CellKey& o) const noexcept { return x == o.x && y == o.y; }
};
struct CellKeyHash {
  std::size_t operator()(const CellKey& c) const noexcept {
    // Cantor pairing-ish; works fine for the typical small range we deal with.
    const std::size_t ux = static_cast<std::size_t>(c.x + (1 << 20));
    const std::size_t uy = static_cast<std::size_t>(c.y + (1 << 20));
    return (ux * 2654435761u) ^ uy;
  }
};

inline CellKey keyOf(double x, double y, double res) {
  return CellKey{
      static_cast<int>(std::floor(x / res)),
      static_cast<int>(std::floor(y / res)),
  };
}

}  // namespace

std::vector<WorldXY> voxelDownsample2D(const std::vector<WorldXY>& points,
                                       double resolution_m) {
  if (resolution_m <= 0.0) return points;
  std::unordered_map<CellKey, std::pair<WorldXY, std::size_t>, CellKeyHash> buckets;
  buckets.reserve(points.size());
  for (const auto& p : points) {
    const CellKey k = keyOf(p.x, p.y, resolution_m);
    auto it = buckets.find(k);
    if (it == buckets.end()) {
      buckets.emplace(k, std::make_pair(p, std::size_t{1}));
    } else {
      it->second.first.x += p.x;
      it->second.first.y += p.y;
      ++it->second.second;
    }
  }
  std::vector<WorldXY> out;
  out.reserve(buckets.size());
  for (const auto& kv : buckets) {
    const auto& sum = kv.second.first;
    const double n = static_cast<double>(kv.second.second);
    out.push_back(WorldXY{sum.x / n, sum.y / n});
  }
  return out;
}

std::vector<Cluster> clusterPoints2D(const std::vector<WorldXY>& points,
                                     double resolution_m) {
  std::vector<Cluster> clusters;
  if (points.empty() || resolution_m <= 0.0) return clusters;

  // Bucket each point into a grid cell at `resolution_m` and remember the
  // points that fell into that bucket.
  std::unordered_map<CellKey, std::vector<std::size_t>, CellKeyHash> buckets;
  buckets.reserve(points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    buckets[keyOf(points[i].x, points[i].y, resolution_m)].push_back(i);
  }

  // 4-connected BFS over occupied buckets to form blobs.
  std::unordered_set<CellKey, CellKeyHash> visited;
  visited.reserve(buckets.size());
  static const int neighbors[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

  for (const auto& seed : buckets) {
    if (visited.count(seed.first)) continue;
    std::vector<CellKey> queue{seed.first};
    visited.insert(seed.first);
    Cluster cl;
    cl.min_xy = WorldXY{std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::infinity()};
    cl.max_xy = WorldXY{-std::numeric_limits<double>::infinity(),
                        -std::numeric_limits<double>::infinity()};
    double sx = 0.0, sy = 0.0;
    std::size_t total = 0;
    while (!queue.empty()) {
      const CellKey k = queue.back();
      queue.pop_back();
      const auto it = buckets.find(k);
      if (it == buckets.end()) continue;
      for (const std::size_t pi : it->second) {
        const WorldXY& p = points[pi];
        sx += p.x;
        sy += p.y;
        cl.min_xy.x = std::min(cl.min_xy.x, p.x);
        cl.min_xy.y = std::min(cl.min_xy.y, p.y);
        cl.max_xy.x = std::max(cl.max_xy.x, p.x);
        cl.max_xy.y = std::max(cl.max_xy.y, p.y);
        ++total;
      }
      for (const auto& n : neighbors) {
        const CellKey nk{k.x + n[0], k.y + n[1]};
        if (visited.count(nk)) continue;
        if (buckets.find(nk) == buckets.end()) continue;
        visited.insert(nk);
        queue.push_back(nk);
      }
    }
    if (total == 0) continue;
    cl.point_count = total;
    cl.centroid.x = sx / static_cast<double>(total);
    cl.centroid.y = sy / static_cast<double>(total);
    clusters.push_back(cl);
  }
  return clusters;
}

}  // namespace clr
