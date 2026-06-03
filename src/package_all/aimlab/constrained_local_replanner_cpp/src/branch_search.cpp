#include "constrained_local_replanner_cpp/branch_search.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace clr {

namespace {

struct Node {
  int idx;
  double f;
};
struct NodeCmp {
  bool operator()(const Node& a, const Node& b) const noexcept { return a.f > b.f; }
};

inline int packIdx(int gx, int gy, int width) {
  return gy * width + gx;
}

inline bool isBlocked(const std::vector<uint8_t>& blocked, int width, int height,
                      int gx, int gy) {
  if (gx < 0 || gy < 0 || gx >= width || gy >= height) return true;
  return blocked[static_cast<std::size_t>(gy) * static_cast<std::size_t>(width) +
                 static_cast<std::size_t>(gx)] != 0;
}

}  // namespace

std::vector<GridCell> aStarBranch(const std::vector<uint8_t>& blocked,
                                  int width, int height,
                                  GridCell start,
                                  const std::vector<GridCell>& rejoin_candidates,
                                  int max_expand,
                                  double time_budget_s) {
  std::vector<GridCell> result;
  if (rejoin_candidates.empty()) return result;
  if (isBlocked(blocked, width, height, start.x, start.y)) return result;

  // Set of acceptable goal cells. Heuristic uses distance to the nearest one.
  std::unordered_set<int> goals;
  goals.reserve(rejoin_candidates.size());
  for (const auto& g : rejoin_candidates) {
    if (!isBlocked(blocked, width, height, g.x, g.y)) {
      goals.insert(packIdx(g.x, g.y, width));
    }
  }
  if (goals.empty()) return result;

  auto heuristic = [&](int gx, int gy) -> double {
    double best = std::numeric_limits<double>::infinity();
    for (const auto& g : rejoin_candidates) {
      const double d = std::hypot(static_cast<double>(gx - g.x),
                                  static_cast<double>(gy - g.y));
      if (d < best) best = d;
    }
    return best;
  };

  std::priority_queue<Node, std::vector<Node>, NodeCmp> open;
  std::unordered_map<int, double> g_score;
  std::unordered_map<int, int> came_from;

  const int start_idx = packIdx(start.x, start.y, width);
  g_score[start_idx] = 0.0;
  open.push(Node{start_idx, heuristic(start.x, start.y)});

  const auto t0 = std::chrono::steady_clock::now();
  int expansions = 0;
  static const int dxs[8] = {1, -1, 0, 0, 1, 1, -1, -1};
  static const int dys[8] = {0, 0, 1, -1, 1, -1, 1, -1};
  static const double costs[8] = {1.0, 1.0, 1.0, 1.0,
                                  1.4142135624, 1.4142135624,
                                  1.4142135624, 1.4142135624};

  int goal_idx = -1;
  while (!open.empty()) {
    const Node cur = open.top();
    open.pop();
    if (goals.count(cur.idx)) {
      goal_idx = cur.idx;
      break;
    }
    if (++expansions > max_expand) break;
    if ((expansions & 0x7F) == 0) {
      const auto dt =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
              .count();
      if (dt > time_budget_s) break;
    }
    const int cx = cur.idx % width;
    const int cy = cur.idx / width;
    const double cur_g = g_score[cur.idx];
    for (int n = 0; n < 8; ++n) {
      const int nx = cx + dxs[n];
      const int ny = cy + dys[n];
      if (isBlocked(blocked, width, height, nx, ny)) continue;
      const int nidx = packIdx(nx, ny, width);
      const double tentative = cur_g + costs[n];
      auto it = g_score.find(nidx);
      if (it == g_score.end() || tentative < it->second) {
        g_score[nidx] = tentative;
        came_from[nidx] = cur.idx;
        open.push(Node{nidx, tentative + heuristic(nx, ny)});
      }
    }
  }

  if (goal_idx < 0) return result;
  // Reconstruct path.
  std::vector<int> stack;
  int idx = goal_idx;
  while (idx != start_idx) {
    stack.push_back(idx);
    auto it = came_from.find(idx);
    if (it == came_from.end()) return {};  // broken — should not happen
    idx = it->second;
  }
  stack.push_back(start_idx);
  result.reserve(stack.size());
  for (auto rit = stack.rbegin(); rit != stack.rend(); ++rit) {
    result.push_back(GridCell{*rit % width, *rit / width});
  }
  return result;
}

}  // namespace clr
