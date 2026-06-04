#include "constrained_local_replanner_cpp/avoidance_state.hpp"

#include <algorithm>
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
          if (cfg_.hold_without_candidate) {
            // Legacy behavior: an unrouteable blocker becomes a local HOLD.
            mode_ = PathMode::HOLD;
            hold_entry_sec_ = now_sec;
          }
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
        if (cfg_.hold_without_candidate) {
          mode_ = PathMode::HOLD;
          hold_entry_sec_ = now_sec;
        } else {
          // Conservative path blocking can invalidate a cached detour even
          // when raw near-field is clear. Return to nominal and let the
          // safety stop own close obstacles.
          mode_ = PathMode::FOLLOW_LOCAL;
          confirm_count_ = 0;
          last_clear_seen_sec_ = 0.0;
        }
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
      if (!nominal_blocked) {
        mode_ = PathMode::FOLLOW_LOCAL;
        confirm_count_ = 0;
        hold_entry_sec_ = 0.0;
      } else if (avoidance_candidate_available) {
        mode_ = PathMode::FOLLOW_AVOIDANCE;
        avoidance_entry_sec_ = now_sec;
        last_clear_seen_sec_ = 0.0;
        hold_entry_sec_ = 0.0;
      } else if (!cfg_.hold_without_candidate) {
        mode_ = PathMode::FOLLOW_LOCAL;
        confirm_count_ = 0;
        hold_entry_sec_ = 0.0;
      } else if (hold_entry_sec_ > 0.0 &&
                 cfg_.hold_escape_timeout_s > 0.0 &&
                 (now_sec - hold_entry_sec_) > cfg_.hold_escape_timeout_s) {
        // Optional manual escape hatch for conservative maps. The default is
        // disabled so a currently blocked nominal path never becomes a forward
        // drive command just because no candidate was found.
        mode_ = PathMode::FOLLOW_LOCAL;
        confirm_count_ = 0;
        hold_entry_sec_ = 0.0;
      }
      break;
    }
  }
  return mode_;
}

void AvoidanceStateMachine::forceFollowLocal() {
  mode_ = PathMode::FOLLOW_LOCAL;
  confirm_count_ = 0;
  last_clear_seen_sec_ = 0.0;
  avoidance_entry_sec_ = 0.0;
  hold_entry_sec_ = 0.0;
}

void AvoidanceStateMachine::recordStaticHit(WorldXY centroid, double now_sec) {
  if (cfg_.locked_static_ttl_s <= 0.0) {
    locked_.clear();
    return;
  }
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
      entry.last_seen_sec = now_sec;
      if (!entry.locked && entry.hits >= cfg_.locked_static_persistence_hits) {
        entry.locked = true;
      }
      return;
    }
  }
  locked_.push_back(LockedStatic{centroid, 1, false, now_sec});
}

void AvoidanceStateMachine::pruneStaleStatic(double now_sec) {
  if (locked_.empty()) return;
  if (cfg_.locked_static_ttl_s <= 0.0) {
    locked_.clear();
    return;
  }
  const double ttl = cfg_.locked_static_ttl_s;
  locked_.erase(
      std::remove_if(locked_.begin(), locked_.end(),
                     [now_sec, ttl](const LockedStatic& entry) {
                       return entry.last_seen_sec <= 0.0 ||
                              (now_sec - entry.last_seen_sec) > ttl;
                     }),
      locked_.end());
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
