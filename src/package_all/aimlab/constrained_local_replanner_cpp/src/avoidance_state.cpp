#include "constrained_local_replanner_cpp/avoidance_state.hpp"

#include <cmath>

namespace clr {

const char* pathModeToString(PathMode m) {
  switch (m) {
    case PathMode::FOLLOW_LOCAL:     return "follow_local";
    case PathMode::FOLLOW_AVOIDANCE: return "follow_avoidance";
    case PathMode::HOLD:             return "hold";
  }
  return "follow_local";
}

PathMode AvoidanceStateMachine::update(bool nominal_blocked,
                                       bool avoidance_candidate_available,
                                       double endpoint_distance_m,
                                       bool cached_path_still_drivable,
                                       bool locked_static_nearby,
                                       double now_sec) {
  switch (mode_) {
    case PathMode::FOLLOW_LOCAL: {
      if (nominal_blocked) {
        ++confirm_count_;
        if (confirm_count_ >= cfg_.trigger_confirm_cycles &&
            avoidance_candidate_available) {
          mode_ = PathMode::FOLLOW_AVOIDANCE;
          avoidance_entry_sec_ = now_sec;
          last_clear_seen_sec_ = 0.0;
          confirm_count_ = 0;
        } else if (confirm_count_ >= cfg_.trigger_confirm_cycles &&
                   !avoidance_candidate_available) {
          // Blocker confirmed but no detour found — escalate to HOLD until
          // either obstacle clears or a candidate appears.
          mode_ = PathMode::HOLD;
          confirm_count_ = 0;
        }
      } else {
        confirm_count_ = 0;
      }
      break;
    }
    case PathMode::FOLLOW_AVOIDANCE: {
      // Track time since last "clear" perception so the release guard can use
      // it the same way the Python replanner's avoidance_clear_detour_hold_s
      // works.
      if (!nominal_blocked) {
        if (last_clear_seen_sec_ <= 0.0) last_clear_seen_sec_ = now_sec;
      } else {
        last_clear_seen_sec_ = 0.0;
      }
      const bool stale = !cached_path_still_drivable;
      if (stale && !avoidance_candidate_available) {
        // Cached path lost validity and no fresh candidate — bail to HOLD.
        mode_ = PathMode::HOLD;
        break;
      }
      const bool clear_long_enough =
          last_clear_seen_sec_ > 0.0 &&
          (now_sec - last_clear_seen_sec_) >= cfg_.clear_detour_hold_s;
      const bool past_endpoint =
          endpoint_distance_m <= cfg_.keep_until_endpoint_distance_m;
      if (clear_long_enough && past_endpoint && !locked_static_nearby) {
        mode_ = PathMode::FOLLOW_LOCAL;
        confirm_count_ = 0;
      }
      break;
    }
    case PathMode::HOLD: {
      if (avoidance_candidate_available) {
        mode_ = PathMode::FOLLOW_AVOIDANCE;
        avoidance_entry_sec_ = now_sec;
        last_clear_seen_sec_ = 0.0;
      } else if (!nominal_blocked) {
        mode_ = PathMode::FOLLOW_LOCAL;
        confirm_count_ = 0;
      }
      break;
    }
  }
  return mode_;
}

void AvoidanceStateMachine::recordStaticHit(WorldXY centroid) {
  const double r2 = cfg_.locked_static_hit_radius_m *
                    cfg_.locked_static_hit_radius_m;
  for (auto& entry : locked_) {
    const double dx = entry.centroid.x - centroid.x;
    const double dy = entry.centroid.y - centroid.y;
    if (dx * dx + dy * dy <= r2) {
      entry.hits += 1;
      // Running mean to stabilise the centroid as more hits accumulate.
      const double n = static_cast<double>(entry.hits);
      entry.centroid.x = entry.centroid.x + (centroid.x - entry.centroid.x) / n;
      entry.centroid.y = entry.centroid.y + (centroid.y - entry.centroid.y) / n;
      if (!entry.locked && entry.hits >= cfg_.locked_static_persistence_hits) {
        entry.locked = true;
      }
      return;
    }
  }
  locked_.push_back(LockedStatic{centroid, 1, false});
}

bool AvoidanceStateMachine::lockedStaticNearby(WorldXY robot,
                                               double radius_m) const {
  if (radius_m <= 0.0) return false;
  const double r2 = radius_m * radius_m;
  for (const auto& entry : locked_) {
    if (!entry.locked) continue;
    const double dx = entry.centroid.x - robot.x;
    const double dy = entry.centroid.y - robot.y;
    if (dx * dx + dy * dy <= r2) return true;
  }
  return false;
}

}  // namespace clr
