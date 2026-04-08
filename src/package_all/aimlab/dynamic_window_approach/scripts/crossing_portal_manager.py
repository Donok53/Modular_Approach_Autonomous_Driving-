#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os

import rospy
import yaml
from dynamic_window_approach.msg import TrackedObjectArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from std_msgs.msg import String


class CrossingPortalManager(object):
    """Selects a bidirectional portal and publishes staged direct goals.

    The portal layout lives in a YAML file so operators can edit coordinates
    without touching code. The file is hot-reloaded when it changes.
    """

    def __init__(self):
        self.config_path = str(rospy.get_param("~config_path", "")).strip()
        self.odom_topic = rospy.get_param("~odom_topic", "/lio_localizer/odometry/optimization")
        self.global_path_topic = rospy.get_param("~global_path_topic", "/astar/path")
        self.drivable_grid_topic = rospy.get_param("~drivable_grid_topic", "/lio_sam/drivable_area/grid")
        self.tracked_objects_topic = str(
            rospy.get_param("~tracked_objects_topic", "/perception/tracked_objects")
        ).strip()
        self.direct_goal_topic = rospy.get_param("~direct_goal_topic", "/planning/crossing_direct_goal")
        self.status_topic = rospy.get_param("~status_topic", "/planning/crossing_portal_status")
        self.publish_hz = max(1.0, float(rospy.get_param("~publish_hz", 5.0)))
        self.config_reload_period_s = max(
            0.2, float(rospy.get_param("~config_reload_period_s", 2.0))
        )
        self.robot_width_m = max(0.1, float(rospy.get_param("~robot_width_m", 0.55)))
        self.footprint_padding_m = max(
            0.0, float(rospy.get_param("~footprint_padding_m", 0.05))
        )
        self.default_corridor_half_width_m = max(
            0.3,
            float(
                rospy.get_param(
                    "~default_corridor_half_width_m",
                    0.5 * self.robot_width_m + self.footprint_padding_m + 0.10,
                )
            ),
        )
        self.grid_blocked_threshold = int(rospy.get_param("~grid_blocked_threshold", 60))
        self.grid_unknown_penalty = max(
            0.0, float(rospy.get_param("~grid_unknown_penalty", 25.0))
        )
        self.grid_blocked_penalty = max(
            1.0, float(rospy.get_param("~grid_blocked_penalty", 120.0))
        )
        self.object_penalty_scale = max(
            1.0, float(rospy.get_param("~object_penalty_scale", 80.0))
        )
        self.object_dynamic_bonus = max(
            0.0, float(rospy.get_param("~object_dynamic_bonus", 40.0))
        )

        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.have_odom = False
        self.global_path = None
        self.drivable_grid = None
        self.tracked_objects = []

        self.crossings = []
        self._config_mtime = None
        self._last_reload_sec = 0.0
        self._last_status_text = ""
        self._active_crossing_id = ""
        self._active_target_side = ""
        self._active_candidate_id = ""
        self._active_stage = "idle"
        self._active_candidate_score = float("inf")

        self.pub_goal = rospy.Publisher(self.direct_goal_topic, PoseStamped, queue_size=1)
        self.pub_status = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)

        self.sub_odom = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=10)
        self.sub_global_path = rospy.Subscriber(
            self.global_path_topic, Path, self.global_path_callback, queue_size=5
        )
        self.sub_grid = rospy.Subscriber(
            self.drivable_grid_topic, OccupancyGrid, self.grid_callback, queue_size=3
        )
        self.sub_objects = None
        if self.tracked_objects_topic:
            self.sub_objects = rospy.Subscriber(
                self.tracked_objects_topic,
                TrackedObjectArray,
                self.tracked_objects_callback,
                queue_size=3,
            )

        self._load_config(force=True)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_hz), self.on_timer)
        rospy.loginfo(
            "crossing_portal_manager started | config=%s odom=%s global=%s grid=%s tracked=%s goal=%s status=%s",
            self.config_path if self.config_path else "-",
            self.odom_topic,
            self.global_path_topic,
            self.drivable_grid_topic,
            self.tracked_objects_topic if self.tracked_objects_topic else "-",
            self.direct_goal_topic,
            self.status_topic,
        )

    @staticmethod
    def _clamp(value, lo, hi):
        return max(lo, min(hi, value))

    @staticmethod
    def _angle_wrap(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _point(data):
        return (float(data.get("x", 0.0)), float(data.get("y", 0.0)))

    @staticmethod
    def _pose_dict(data, default_yaw_deg=0.0):
        if data is None:
            return {
                "x": 0.0,
                "y": 0.0,
                "yaw": math.radians(default_yaw_deg),
            }
        if "pose" in data and isinstance(data["pose"], dict):
            data = data["pose"]
        return {
            "x": float(data.get("x", 0.0)),
            "y": float(data.get("y", 0.0)),
            "yaw": math.radians(float(data.get("yaw_deg", default_yaw_deg))),
        }

    @staticmethod
    def _dist_xy(a, b):
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

    @staticmethod
    def _point_segment_distance(px, py, ax, ay, bx, by):
        dx = bx - ax
        dy = by - ay
        denom = dx * dx + dy * dy
        if denom <= 1e-9:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / denom
        t = max(0.0, min(1.0, t))
        cx = ax + t * dx
        cy = ay + t * dy
        return math.hypot(px - cx, py - cy)

    def _status(self, text):
        if text == self._last_status_text:
            return
        self._last_status_text = text
        self.pub_status.publish(String(data=text))
        rospy.loginfo("crossing_portal_manager: %s", text)

    def _clear_active_state(self, reason):
        if self._active_crossing_id:
            self._status(
                "inactive reason={} crossing={} stage={}".format(
                    reason, self._active_crossing_id, self._active_stage
                )
            )
        self._active_crossing_id = ""
        self._active_target_side = ""
        self._active_candidate_id = ""
        self._active_stage = "idle"
        self._active_candidate_score = float("inf")

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.odom_x = float(p.x)
        self.odom_y = float(p.y)
        self.odom_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.have_odom = True

    def global_path_callback(self, msg):
        self.global_path = msg

    def grid_callback(self, msg):
        self.drivable_grid = msg

    def tracked_objects_callback(self, msg):
        self.tracked_objects = list(msg.objects)

    def _load_config(self, force=False):
        if not self.config_path:
            if force:
                rospy.logwarn("crossing_portal_manager: no config_path set; staying idle")
            self.crossings = []
            return
        if not os.path.isfile(self.config_path):
            if force:
                rospy.logwarn(
                    "crossing_portal_manager: config file not found: %s", self.config_path
                )
            self.crossings = []
            return
        try:
            mtime = os.path.getmtime(self.config_path)
        except OSError:
            return
        if (not force) and self._config_mtime is not None and mtime <= self._config_mtime:
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
        except Exception as exc:
            rospy.logwarn("crossing_portal_manager: failed to load config: %s", str(exc))
            return
        defaults = raw.get("defaults", {}) if isinstance(raw, dict) else {}
        crossings_raw = raw.get("crossings", []) if isinstance(raw, dict) else []
        crossings = []
        for item in crossings_raw:
            crossing = self._normalize_crossing(item, defaults)
            if crossing is not None:
                crossings.append(crossing)
        self.crossings = crossings
        self._config_mtime = mtime
        rospy.loginfo(
            "crossing_portal_manager: loaded %d portal crossings from %s",
            len(self.crossings),
            self.config_path,
        )

    def _normalize_crossing(self, item, defaults):
        if not isinstance(item, dict):
            return None
        portal = item.get("portal", {})
        if not isinstance(portal, dict) or "center" not in portal:
            return None
        center = self._point(portal["center"])
        normal_yaw = math.radians(float(portal.get("normal_deg", 0.0)))
        corridor_half_width_m = max(
            self.default_corridor_half_width_m,
            float(
                portal.get(
                    "corridor_half_width_m",
                    defaults.get("corridor_half_width_m", self.default_corridor_half_width_m),
                )
            ),
        )
        candidate_score_threshold = float(
            item.get(
                "candidate_score_threshold",
                defaults.get("candidate_score_threshold", 220.0),
            )
        )
        switch_margin = float(
            item.get("candidate_switch_margin", defaults.get("candidate_switch_margin", 35.0))
        )
        crossing = {
            "id": str(item.get("id", "crossing_{}".format(len(self.crossings) + 1))),
            "enabled": bool(item.get("enabled", True)),
            "activation_radius_m": float(
                item.get("activation_radius_m", defaults.get("activation_radius_m", 4.0))
            ),
            "commit_distance_m": float(
                item.get("commit_distance_m", defaults.get("commit_distance_m", 0.80))
            ),
            "completion_radius_m": float(
                item.get("completion_radius_m", defaults.get("completion_radius_m", 0.75))
            ),
            "side_deadband_m": float(
                item.get("side_deadband_m", defaults.get("side_deadband_m", 0.15))
            ),
            "candidate_score_threshold": candidate_score_threshold,
            "candidate_switch_margin": switch_margin,
            "portal": {
                "center": center,
                "normal_yaw": normal_yaw,
                "width_m": float(portal.get("width_m", defaults.get("width_m", 1.40))),
                "approach_offset_m": float(
                    portal.get("approach_offset_m", defaults.get("approach_offset_m", 1.30))
                ),
                "post_offset_m": float(
                    portal.get("post_offset_m", defaults.get("post_offset_m", 1.10))
                ),
                "portal_check_length_m": float(
                    portal.get(
                        "portal_check_length_m", defaults.get("portal_check_length_m", 0.80)
                    )
                ),
                "corridor_half_width_m": corridor_half_width_m,
            },
            "side_a": {
                "name": str(item.get("side_a", {}).get("name", "side_a")),
                "candidates": self._normalize_candidates(
                    item.get("side_a", {}).get("exit_candidates", []),
                    defaults,
                    normal_yaw + math.pi,
                    corridor_half_width_m,
                ),
            },
            "side_b": {
                "name": str(item.get("side_b", {}).get("name", "side_b")),
                "candidates": self._normalize_candidates(
                    item.get("side_b", {}).get("exit_candidates", []),
                    defaults,
                    normal_yaw,
                    corridor_half_width_m,
                ),
            },
        }
        return crossing

    def _normalize_candidates(self, raw_candidates, defaults, default_yaw, corridor_half_width_m):
        out = []
        for idx, item in enumerate(raw_candidates):
            if not isinstance(item, dict):
                continue
            pose = self._pose_dict(item, math.degrees(default_yaw))
            out.append(
                {
                    "id": str(item.get("id", "candidate_{}".format(idx + 1))),
                    "pose": pose,
                    "clearance_radius_m": float(
                        item.get(
                            "clearance_radius_m",
                            defaults.get("candidate_clearance_radius_m", 0.65),
                        )
                    ),
                    "corridor_half_width_m": max(
                        corridor_half_width_m,
                        float(
                            item.get(
                                "corridor_half_width_m",
                                defaults.get(
                                    "candidate_corridor_half_width_m", corridor_half_width_m
                                ),
                            )
                        ),
                    ),
                }
            )
        return out

    def _maybe_reload_config(self):
        now_sec = rospy.get_time()
        if (now_sec - self._last_reload_sec) < self.config_reload_period_s:
            return
        self._last_reload_sec = now_sec
        self._load_config(force=False)

    def _classify_side(self, crossing, x, y, fallback=""):
        center_x, center_y = crossing["portal"]["center"]
        ny = crossing["portal"]["normal_yaw"]
        signed = (x - center_x) * math.cos(ny) + (y - center_y) * math.sin(ny)
        deadband = crossing["side_deadband_m"]
        if signed > deadband:
            return "side_b", signed
        if signed < -deadband:
            return "side_a", signed
        if fallback in ("side_a", "side_b"):
            return fallback, signed
        return ("side_b" if signed >= 0.0 else "side_a"), signed

    def _global_target_side(self, crossing):
        if self.global_path is None or not self.global_path.poses:
            return ""
        goal_pose = self.global_path.poses[-1].pose.position
        side, _ = self._classify_side(
            crossing, float(goal_pose.x), float(goal_pose.y), fallback=""
        )
        return side

    def _current_crossing(self):
        if not self.have_odom or not self.crossings:
            return None, "", ""

        if self._active_crossing_id:
            crossing = next(
                (c for c in self.crossings if c["id"] == self._active_crossing_id and c["enabled"]),
                None,
            )
            if crossing is not None:
                dist_to_portal = self._dist_xy(
                    (self.odom_x, self.odom_y), crossing["portal"]["center"]
                )
                if dist_to_portal <= (crossing["activation_radius_m"] + 4.0):
                    current_side, _ = self._classify_side(
                        crossing, self.odom_x, self.odom_y, fallback=self._active_target_side
                    )
                    return crossing, current_side, self._active_target_side

        best = None
        best_dist = float("inf")
        for crossing in self.crossings:
            if not crossing["enabled"]:
                continue
            dist_to_portal = self._dist_xy(
                (self.odom_x, self.odom_y), crossing["portal"]["center"]
            )
            if dist_to_portal > crossing["activation_radius_m"]:
                continue
            current_side, _ = self._classify_side(crossing, self.odom_x, self.odom_y)
            target_side = self._global_target_side(crossing)
            if target_side not in ("side_a", "side_b") or target_side == current_side:
                continue
            if dist_to_portal < best_dist:
                best = (crossing, current_side, target_side)
                best_dist = dist_to_portal
        if best is None:
            return None, "", ""
        return best

    def _pose_stamped(self, x, y, yaw, frame_id):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = math.sin(0.5 * yaw)
        msg.pose.orientation.w = math.cos(0.5 * yaw)
        return msg

    def _approach_pose(self, crossing, side_name):
        cx, cy = crossing["portal"]["center"]
        ny = crossing["portal"]["normal_yaw"]
        offset = crossing["portal"]["approach_offset_m"]
        direction = -1.0 if side_name == "side_a" else 1.0
        x = cx + direction * math.cos(ny) * offset
        y = cy + direction * math.sin(ny) * offset
        yaw = ny if side_name == "side_a" else self._angle_wrap(ny + math.pi)
        return self._pose_stamped(x, y, yaw, self.drivable_grid.header.frame_id if self.drivable_grid else "map")

    def _portal_center_pose(self, crossing, target_side):
        cx, cy = crossing["portal"]["center"]
        ny = crossing["portal"]["normal_yaw"]
        yaw = ny if target_side == "side_b" else self._angle_wrap(ny + math.pi)
        return self._pose_stamped(
            cx,
            cy,
            yaw,
            self.drivable_grid.header.frame_id if self.drivable_grid else "map",
        )

    def _candidate_pose(self, candidate):
        pose = candidate["pose"]
        return self._pose_stamped(
            pose["x"],
            pose["y"],
            pose["yaw"],
            self.drivable_grid.header.frame_id if self.drivable_grid else "map",
        )

    def _grid_penalty_at(self, x, y):
        if self.drivable_grid is None:
            return 0.0
        g = self.drivable_grid
        res = max(1e-3, float(g.info.resolution))
        ox = float(g.info.origin.position.x)
        oy = float(g.info.origin.position.y)
        gx = int(math.floor((x - ox) / res))
        gy = int(math.floor((y - oy) / res))
        if gx < 0 or gy < 0 or gx >= g.info.width or gy >= g.info.height:
            return self.grid_blocked_penalty
        value = int(g.data[gy * g.info.width + gx])
        if value < 0:
            return self.grid_unknown_penalty
        if value >= self.grid_blocked_threshold:
            return self.grid_blocked_penalty
        return float(value) * 0.35

    def _segment_grid_score(self, start_xy, end_xy, half_width_m):
        ax, ay = start_xy
        bx, by = end_xy
        seg_len = self._dist_xy(start_xy, end_xy)
        if seg_len <= 1e-6:
            return self._grid_penalty_at(ax, ay)
        step_m = 0.20
        steps = max(2, int(math.ceil(seg_len / step_m)))
        tx = -(by - ay) / seg_len
        ty = (bx - ax) / seg_len
        lateral_offsets = [0.0]
        if half_width_m > 0.05:
            lateral_offsets = [-half_width_m, 0.0, half_width_m]
        total = 0.0
        for idx in range(steps + 1):
            ratio = float(idx) / float(steps)
            px = ax + (bx - ax) * ratio
            py = ay + (by - ay) * ratio
            for offset in lateral_offsets:
                total += self._grid_penalty_at(px + tx * offset, py + ty * offset)
        return total

    def _circle_grid_score(self, center_xy, radius_m):
        cx, cy = center_xy
        samples = max(6, int(math.ceil((2.0 * math.pi * max(radius_m, 0.05)) / 0.25)))
        total = self._grid_penalty_at(cx, cy)
        for idx in range(samples):
            ang = (2.0 * math.pi * float(idx)) / float(samples)
            total += self._grid_penalty_at(
                cx + math.cos(ang) * radius_m,
                cy + math.sin(ang) * radius_m,
            )
        return total

    def _object_segment_penalty(self, start_xy, end_xy, half_width_m, candidate_xy, clearance_radius_m):
        total = 0.0
        ax, ay = start_xy
        bx, by = end_xy
        for obj in self.tracked_objects:
            ox = float(obj.pose.position.x)
            oy = float(obj.pose.position.y)
            obj_radius = max(0.30, 0.5 * max(float(obj.size.x), float(obj.size.y)))
            speed = math.hypot(float(obj.twist.linear.x), float(obj.twist.linear.y))
            candidate_dist = math.hypot(ox - candidate_xy[0], oy - candidate_xy[1])
            corridor_dist = self._point_segment_distance(ox, oy, ax, ay, bx, by)
            if candidate_dist <= (clearance_radius_m + obj_radius):
                total += self.object_penalty_scale + speed * self.object_dynamic_bonus
            if corridor_dist <= (half_width_m + obj_radius):
                total += 0.8 * self.object_penalty_scale + speed * self.object_dynamic_bonus
        return total

    def _portal_opening_score(self, crossing):
        cx, cy = crossing["portal"]["center"]
        yaw = crossing["portal"]["normal_yaw"]
        half_width = 0.5 * crossing["portal"]["width_m"]
        half_len = 0.5 * crossing["portal"]["portal_check_length_m"]
        nx = math.cos(yaw)
        ny = math.sin(yaw)
        tx = -math.sin(yaw)
        ty = math.cos(yaw)
        samples = [(-half_len, -half_width), (-half_len, 0.0), (-half_len, half_width),
                   (0.0, -half_width), (0.0, 0.0), (0.0, half_width),
                   (half_len, -half_width), (half_len, 0.0), (half_len, half_width)]
        total = 0.0
        for along, lateral in samples:
            px = cx + nx * along + tx * lateral
            py = cy + ny * along + ty * lateral
            total += self._grid_penalty_at(px, py)
        return total

    def _candidate_score(self, crossing, candidate):
        portal_center = crossing["portal"]["center"]
        candidate_xy = (candidate["pose"]["x"], candidate["pose"]["y"])
        corridor_half_width = candidate["corridor_half_width_m"]
        score = 0.0
        score += self._portal_opening_score(crossing)
        score += self._segment_grid_score(portal_center, candidate_xy, corridor_half_width)
        score += self._circle_grid_score(candidate_xy, candidate["clearance_radius_m"])
        score += self._object_segment_penalty(
            portal_center,
            candidate_xy,
            corridor_half_width,
            candidate_xy,
            candidate["clearance_radius_m"],
        )
        score += 0.5 * self._dist_xy(portal_center, candidate_xy)
        return score

    def _select_candidate(self, crossing, target_side):
        side_cfg = crossing[target_side]
        candidates = side_cfg["candidates"]
        if not candidates:
            return None, float("inf")
        scored = []
        for candidate in candidates:
            scored.append((self._candidate_score(crossing, candidate), candidate))
        scored.sort(key=lambda item: item[0])
        best_score, best_candidate = scored[0]

        current_candidate = next(
            (cand for _, cand in scored if cand["id"] == self._active_candidate_id),
            None,
        )
        if current_candidate is not None:
            current_score = next(score for score, cand in scored if cand["id"] == current_candidate["id"])
            if current_score <= (best_score + crossing["candidate_switch_margin"]):
                return current_candidate, current_score
        if best_score > crossing["candidate_score_threshold"]:
            return None, best_score
        return best_candidate, best_score

    def _current_pose_goal(self):
        return self._pose_stamped(
            self.odom_x,
            self.odom_y,
            self.odom_yaw,
            self.drivable_grid.header.frame_id if self.drivable_grid else "map",
        )

    def _stage_goal(self, crossing, current_side, target_side, candidate):
        portal_center = crossing["portal"]["center"]
        dist_to_portal = self._dist_xy((self.odom_x, self.odom_y), portal_center)
        _, signed = self._classify_side(crossing, self.odom_x, self.odom_y, fallback=current_side)
        if candidate is None:
            if dist_to_portal <= crossing["commit_distance_m"]:
                return "hold", self._current_pose_goal()
            return "hold", self._approach_pose(crossing, current_side)

        if current_side != target_side:
            if (
                dist_to_portal <= crossing["commit_distance_m"]
                or abs(signed) <= (0.5 * crossing["portal"]["post_offset_m"])
            ):
                return "commit", self._portal_center_pose(crossing, target_side)
            return "approach", self._approach_pose(crossing, current_side)

        candidate_xy = (candidate["pose"]["x"], candidate["pose"]["y"])
        if self._dist_xy((self.odom_x, self.odom_y), candidate_xy) <= crossing["completion_radius_m"]:
            return "complete", None
        return "exit", self._candidate_pose(candidate)

    def _publish_goal(self, crossing, target_side, stage, candidate, goal_msg, score):
        if goal_msg is None:
            self._clear_active_state("complete")
            return
        self.pub_goal.publish(goal_msg)
        candidate_id = candidate["id"] if candidate is not None else "none"
        text = "active crossing={} direction={}->{} stage={} candidate={} score={:.1f}".format(
            crossing["id"],
            "A" if target_side == "side_b" else "B",
            "B" if target_side == "side_b" else "A",
            stage,
            candidate_id,
            score if math.isfinite(score) else -1.0,
        )
        if (
            self._active_crossing_id != crossing["id"]
            or self._active_target_side != target_side
            or self._active_candidate_id != candidate_id
            or self._active_stage != stage
            or abs(score - self._active_candidate_score) > 5.0
        ):
            self._status(text)
        self._active_crossing_id = crossing["id"]
        self._active_target_side = target_side
        self._active_candidate_id = candidate_id if candidate is not None else ""
        self._active_stage = stage
        self._active_candidate_score = score

    def on_timer(self, _evt):
        self._maybe_reload_config()
        if not self.have_odom or self.drivable_grid is None:
            return

        crossing, current_side, target_side = self._current_crossing()
        if crossing is None:
            self._clear_active_state("no_crossing")
            return

        candidate, score = self._select_candidate(crossing, target_side)
        stage, goal_msg = self._stage_goal(crossing, current_side, target_side, candidate)
        if stage == "complete":
            self._clear_active_state("reached_target_side")
            return
        self._publish_goal(crossing, target_side, stage, candidate, goal_msg, score)


def main():
    rospy.init_node("crossing_portal_manager", anonymous=False)
    CrossingPortalManager()
    rospy.spin()


if __name__ == "__main__":
    main()
