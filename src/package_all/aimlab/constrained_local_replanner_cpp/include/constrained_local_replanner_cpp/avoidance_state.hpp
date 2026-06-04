#pragma once

#include <chrono>
#include <cstdint>
#include <vector>

#include "constrained_local_replanner_cpp/types.hpp"

namespace clr {

enum class PathMode : uint8_t {
  FOLLOW_LOCAL = 0,
  FOLLOW_AVOIDANCE = 1,
  HOLD = 2,
};

const char* pathModeToString(PathMode m);

struct LockedStatic {
  WorldXY centroid;
  int hits{0};
  bool locked{false};
  double last_seen_sec{0.0};
};

// Lightweight state machine that owns the avoidance lifecycle:
//   FOLLOW_LOCAL -> (blocker confirmed for N ticks) -> FOLLOW_AVOIDANCE
//   FOLLOW_AVOIDANCE -> (endpoint reached + clear hold) -> FOLLOW_LOCAL
//   optionally HOLD when nominal is blocked and no detour exists
class AvoidanceStateMachine {
 public:
  struct Config {
    int trigger_confirm_cycles{2};
    double clear_detour_hold_s{3.5};
    double keep_until_endpoint_distance_m{0.30};
    double locked_static_hit_radius_m{0.40};
    int locked_static_persistence_hits{3};
    double locked_static_hold_radius_m{3.0};
    // Keep locked static memory only long enough to bridge short perception
    // dropouts. Set <= 0 to disable locked static memory entirely.
    double locked_static_ttl_s{0.8};
    // If HOLD persists this long with no candidate, fall through to
    // FOLLOW_LOCAL so the robot stops being stuck. DWA's near-field
    // safety stop still catches genuine collision threats.
    double hold_escape_timeout_s{5.0};
    // The C++ blocker check is intentionally fast and can be conservative in
    // cluttered indoor scenes. By default, a blocked nominal path with no valid
    // detour keeps publishing the nominal path; DWA's raw near-field stop owns
    // the actual collision stop.
    bool hold_without_candidate{false};
  };

  AvoidanceStateMachine() : cfg_(Config()) {}
  explicit AvoidanceStateMachine(Config cfg) : cfg_(cfg) {}

  // Update internal state. Returns the path mode to publish this tick.
  // - `nominal_blocked` : nominal path is currently blocked
  // - `avoidance_candidate_available` : sidestep/branch produced a valid path
  // - `endpoint_distance_m` : how far the robot is from the cached avoidance
  //                          path's endpoint (only meaningful when avoidance
  //                          is active)
  // - `cached_path_still_drivable` : false if cached avoidance path now
  //                                  intersects a non-drivable cell
  // - `locked_static_nearby` : a confirmed static blocker is within
  //                            cfg.locked_static_hold_radius_m of the robot
  PathMode update(bool nominal_blocked, bool avoidance_candidate_available,
                  double endpoint_distance_m, bool cached_path_still_drivable,
                  bool locked_static_nearby, double now_sec);

  PathMode currentMode() const noexcept { return mode_; }

  // Push a candidate blocker centroid. After
  // cfg.locked_static_persistence_hits sightings within
  // cfg.locked_static_hit_radius_m it becomes "locked" and is preserved
  // briefly across perception dropouts.
  void recordStaticHit(WorldXY centroid, double now_sec);

  // Drop stale locked-static entries. Called once per planner tick.
  void pruneStaleStatic(double now_sec);

  // True if any confirmed (locked) static is within `radius_m` of the robot.
  bool lockedStaticNearby(WorldXY robot, double radius_m) const;

  const std::vector<LockedStatic>& lockedList() const noexcept { return locked_; }

 private:
  Config cfg_{};
  PathMode mode_{PathMode::FOLLOW_LOCAL};
  int confirm_count_{0};
  double last_clear_seen_sec_{0.0};
  double avoidance_entry_sec_{0.0};
  double hold_entry_sec_{0.0};
  std::vector<LockedStatic> locked_;
};

}  // namespace clr
