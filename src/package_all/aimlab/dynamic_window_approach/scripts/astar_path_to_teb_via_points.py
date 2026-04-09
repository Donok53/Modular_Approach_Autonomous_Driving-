#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import math

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry


class AStarPathToTebViaPoints(object):
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/astar/path")
        self.local_input_topic = str(rospy.get_param("~local_input_topic", "")).strip()
        self.avoidance_input_topic = str(rospy.get_param("~avoidance_input_topic", "")).strip()
        self.local_path_timeout_s = max(
            0.1, float(rospy.get_param("~local_path_timeout_s", 4.0))
        )
        self.avoidance_path_timeout_s = max(
            0.1, float(rospy.get_param("~avoidance_path_timeout_s", 1.5))
        )
        self.output_topic = rospy.get_param(
            "~output_topic", "/move_base/TebLocalPlannerROS/via_points"
        )
        self.odom_topic = str(rospy.get_param("~odom_topic", "")).strip()
        self.goal_output_topic = str(rospy.get_param("~goal_output_topic", "")).strip()
        self.goal_lookahead_m = max(
            0.0, float(rospy.get_param("~goal_lookahead_m", 4.0))
        )
        self.final_goal_switch_distance_m = max(
            0.0, float(rospy.get_param("~final_goal_switch_distance_m", 0.5))
        )
        self.goal_update_min_dist_m = max(
            0.0, float(rospy.get_param("~goal_update_min_dist_m", 0.50))
        )
        self.local_goal_tail_backoff_m = max(
            0.0, float(rospy.get_param("~local_goal_tail_backoff_m", 0.0))
        )
        self.local_goal_tail_backoff_trigger_m = max(
            0.0,
            float(
                rospy.get_param(
                    "~local_goal_tail_backoff_trigger_m",
                    max(self.local_goal_tail_backoff_m, 0.0),
                )
            ),
        )
        self.respect_local_hold = bool(rospy.get_param("~respect_local_hold", True))
        self.hold_requires_avoidance_path = bool(
            rospy.get_param("~hold_requires_avoidance_path", True)
        )
        self.min_spacing_m = max(
            0.0, float(rospy.get_param("~min_spacing_m", 0.50))
        )
        self.max_points = max(2, int(rospy.get_param("~max_points", 200)))
        self.path_progress_backtrack_m = max(
            0.0, float(rospy.get_param("~path_progress_backtrack_m", 0.8))
        )
        self.path_progress_forward_search_m = max(
            0.5, float(rospy.get_param("~path_progress_forward_search_m", 3.0))
        )
        self.path_progress_recovery_distance_m = max(
            0.2, float(rospy.get_param("~path_progress_recovery_distance_m", 1.5))
        )

        self._last_sig = None
        self._last_key = None
        self._last_goal_sig = None
        self._fallback_msg = None
        self._fallback_rx_time = 0.0
        self._local_msg = None
        self._local_rx_time = 0.0
        self._avoidance_msg = None
        self._avoidance_rx_time = 0.0
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._have_odom = False
        self._progress_sig = {"fallback": None, "local": None, "avoidance": None}
        self._progress_idx = {"fallback": 0, "local": 0, "avoidance": 0}

        self.pub = rospy.Publisher(self.output_topic, Path, queue_size=1, latch=True)
        self.goal_pub = None
        if self.goal_output_topic:
            self.goal_pub = rospy.Publisher(
                self.goal_output_topic, PoseStamped, queue_size=1, latch=True
            )
        self.sub = rospy.Subscriber(
            self.input_topic, Path, self._make_callback("fallback"), queue_size=2
        )
        self.sub_local = None
        if self.local_input_topic:
            self.sub_local = rospy.Subscriber(
                self.local_input_topic,
                Path,
                self._make_callback("local"),
                queue_size=2,
            )
        self.sub_avoidance = None
        if self.avoidance_input_topic:
            self.sub_avoidance = rospy.Subscriber(
                self.avoidance_input_topic,
                Path,
                self._make_callback("avoidance"),
                queue_size=2,
            )
        self.sub_odom = None
        if self.odom_topic:
            self.sub_odom = rospy.Subscriber(
                self.odom_topic,
                Odometry,
                self._odom_callback,
                queue_size=5,
            )
        self.watchdog = rospy.Timer(rospy.Duration(2.0), self._watchdog_callback)

        rospy.loginfo(
            "astar_path_to_teb_via_points started | fallback=%s local=%s avoidance=%s odom=%s out=%s goal_out=%s goal_lookahead=%.2fm final_switch=%.2fm tail_backoff=%.2fm@%.2fm local_hold=%s hold_requires_avoid=%s spacing=%.2fm max_points=%d",
            self.input_topic,
            self.local_input_topic if self.local_input_topic else "-",
            self.avoidance_input_topic if self.avoidance_input_topic else "-",
            self.odom_topic if self.odom_topic else "-",
            self.output_topic,
            self.goal_output_topic if self.goal_output_topic else "-",
            self.goal_lookahead_m,
            self.final_goal_switch_distance_m,
            self.local_goal_tail_backoff_m,
            self.local_goal_tail_backoff_trigger_m,
            "on" if self.respect_local_hold else "off",
            "on" if self.hold_requires_avoidance_path else "off",
            self.min_spacing_m,
            self.max_points,
        )

    def _watchdog_callback(self, _event):
        if self._fallback_rx_time > 0.0 or self._local_rx_time > 0.0 or self._avoidance_rx_time > 0.0:
            return
        rospy.logwarn_throttle(
            5.0,
            "astar_path_to_teb_via_points: waiting for path input | fallback=%s local=%s avoidance=%s",
            self.input_topic,
            self.local_input_topic if self.local_input_topic else "-",
            self.avoidance_input_topic if self.avoidance_input_topic else "-",
        )

    @staticmethod
    def _dist(a, b):
        dx = float(a.pose.position.x) - float(b.pose.position.x)
        dy = float(a.pose.position.y) - float(b.pose.position.y)
        return math.hypot(dx, dy)

    @staticmethod
    def _set_pose_yaw(pose_stamped, yaw):
        pose_stamped.pose.orientation.x = 0.0
        pose_stamped.pose.orientation.y = 0.0
        pose_stamped.pose.orientation.z = math.sin(0.5 * yaw)
        pose_stamped.pose.orientation.w = math.cos(0.5 * yaw)

    @staticmethod
    def _angle_diff(a, b):
        return math.atan2(math.sin(a - b), math.cos(a - b))

    @staticmethod
    def _segment_yaw(a, b):
        dx = float(b.pose.position.x) - float(a.pose.position.x)
        dy = float(b.pose.position.y) - float(a.pose.position.y)
        if abs(dx) <= 1e-6 and abs(dy) <= 1e-6:
            return None
        return math.atan2(dy, dx)

    def _path_pose_yaw(self, msg, idx):
        if msg is None or not msg.poses or len(msg.poses) == 1:
            return 0.0

        pose_count = len(msg.poses)
        candidate_pairs = []
        if 0 < idx < pose_count - 1:
            candidate_pairs.append((msg.poses[idx - 1], msg.poses[idx + 1]))
        if idx < pose_count - 1:
            candidate_pairs.append((msg.poses[idx], msg.poses[idx + 1]))
        if idx > 0:
            candidate_pairs.append((msg.poses[idx - 1], msg.poses[idx]))

        for pose_a, pose_b in candidate_pairs:
            yaw = self._segment_yaw(pose_a, pose_b)
            if yaw is not None:
                return yaw
        return 0.0

    def _orient_path(self, path):
        if path is None or not path.poses:
            return path
        for idx, pose in enumerate(path.poses):
            self._set_pose_yaw(pose, self._path_pose_yaw(path, idx))
        return path

    def _odom_callback(self, msg):
        self._odom_x = float(msg.pose.pose.position.x)
        self._odom_y = float(msg.pose.pose.position.y)
        self._have_odom = True

    @staticmethod
    def _path_signature(msg):
        if not msg.poses:
            return ()
        sig = [len(msg.poses)]
        stride = max(1, len(msg.poses) // 16)
        for pose in msg.poses[::stride]:
            sig.append(round(float(pose.pose.position.x), 2))
            sig.append(round(float(pose.pose.position.y), 2))
        last = msg.poses[-1]
        sig.append(round(float(last.pose.position.x), 2))
        sig.append(round(float(last.pose.position.y), 2))
        return tuple(sig)

    @staticmethod
    def _tracking_signature(msg):
        if msg is None or not msg.poses:
            return ()
        poses = msg.poses[2:] if len(msg.poses) > 4 else (msg.poses[1:] if len(msg.poses) > 2 else msg.poses)
        sig = [len(msg.poses)]
        stride = max(1, len(poses) // 16)
        for pose in poses[::stride]:
            sig.append(round(float(pose.pose.position.x), 2))
            sig.append(round(float(pose.pose.position.y), 2))
        last = poses[-1]
        sig.append(round(float(last.pose.position.x), 2))
        sig.append(round(float(last.pose.position.y), 2))
        return tuple(sig)

    def _downsample(self, msg):
        out = Path()
        out.header = msg.header
        if len(msg.poses) <= 2 or self.min_spacing_m <= 1e-6:
            out.poses = [copy.deepcopy(pose) for pose in msg.poses[: self.max_points]]
            return self._orient_path(out)

        selected_poses = [msg.poses[0]]
        last = msg.poses[0]

        for pose in msg.poses[1:-1]:
            if self._dist(last, pose) < self.min_spacing_m:
                continue
            selected_poses.append(pose)
            last = pose
            if len(selected_poses) >= (self.max_points - 1):
                break

        if selected_poses[-1] is not msg.poses[-1]:
            selected_poses.append(msg.poses[-1])

        if len(selected_poses) > self.max_points:
            step = max(1, len(selected_poses) // max(1, self.max_points - 1))
            reduced = selected_poses[::step]
            if reduced[-1] is not selected_poses[-1]:
                reduced.append(selected_poses[-1])
            selected_poses = reduced[: self.max_points]
            if selected_poses[-1] is not msg.poses[-1]:
                selected_poses[-1] = msg.poses[-1]

        out.poses = [copy.deepcopy(pose) for pose in selected_poses]
        return self._orient_path(out)

    def _make_callback(self, source):
        def _callback(msg):
            now_sec = rospy.get_time()
            if source == "fallback":
                self._fallback_msg = msg
                self._fallback_rx_time = now_sec
            elif source == "local":
                self._local_msg = msg
                self._local_rx_time = now_sec
            elif source == "avoidance":
                self._avoidance_msg = msg
                self._avoidance_rx_time = now_sec
            self._publish_selected()

        return _callback

    @staticmethod
    def _nearest_pose_index_in_window(msg, start_idx, end_idx, odom_x, odom_y):
        best_idx = max(0, min(int(start_idx), len(msg.poses) - 1))
        best_dist_sq = float("inf")
        for idx in range(max(0, int(start_idx)), min(len(msg.poses) - 1, int(end_idx)) + 1):
            pose = msg.poses[idx]
            dx = float(pose.pose.position.x) - odom_x
            dy = float(pose.pose.position.y) - odom_y
            dist_sq = dx * dx + dy * dy
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_idx = idx
        return best_idx, math.sqrt(best_dist_sq) if best_dist_sq < float("inf") else float("inf")

    def _nearest_pose_index(self, source, msg):
        if msg is None or not msg.poses:
            return 0
        if not self._have_odom:
            return 0

        if source not in self._progress_sig:
            source = "fallback"

        sig = self._tracking_signature(msg)
        if sig != self._progress_sig[source]:
            self._progress_sig[source] = sig
            self._progress_idx[source] = 0

        last_idx = max(0, min(int(self._progress_idx[source]), len(msg.poses) - 1))
        start_idx = last_idx
        backtrack_m = 0.0
        while start_idx > 0 and backtrack_m < self.path_progress_backtrack_m:
            backtrack_m += self._dist(msg.poses[start_idx - 1], msg.poses[start_idx])
            start_idx -= 1

        end_idx = last_idx
        forward_m = 0.0
        while end_idx + 1 < len(msg.poses) and forward_m < self.path_progress_forward_search_m:
            forward_m += self._dist(msg.poses[end_idx], msg.poses[end_idx + 1])
            end_idx += 1

        best_idx, dist_m = self._nearest_pose_index_in_window(
            msg,
            start_idx,
            end_idx,
            self._odom_x,
            self._odom_y,
        )
        if dist_m > self.path_progress_recovery_distance_m:
            best_idx, _dist_m = self._nearest_pose_index_in_window(
                msg,
                0,
                len(msg.poses) - 1,
                self._odom_x,
                self._odom_y,
            )

        self._progress_sig[source] = sig
        self._progress_idx[source] = best_idx
        return best_idx

    def _reset_progress_if_needed(self, source):
        if source in self._progress_sig:
            self._progress_sig[source] = None
            self._progress_idx[source] = 0

    def _goal_idx_backed_off_from_tail(self, msg, start_idx):
        if (
            msg is None
            or not msg.poses
            or self.local_goal_tail_backoff_m <= 1e-6
            or len(msg.poses) < 2
        ):
            return None

        min_idx = min(len(msg.poses) - 1, max(0, start_idx + 1))
        tail_idx = len(msg.poses) - 1
        if min_idx >= tail_idx:
            return tail_idx

        accum_m = 0.0
        chosen_idx = tail_idx
        for idx in range(tail_idx, min_idx, -1):
            accum_m += self._dist(msg.poses[idx - 1], msg.poses[idx])
            chosen_idx = idx - 1
            if accum_m >= self.local_goal_tail_backoff_m:
                break
        return max(min_idx, chosen_idx)

    def _goal_from_selected_path(self, source, msg):
        if msg is None or not msg.poses:
            return None, None, "empty"

        if self.goal_lookahead_m <= 1e-6 or len(msg.poses) == 1:
            return msg.poses[-1], len(msg.poses) - 1, "tail"

        start_idx = self._nearest_pose_index(source, msg)
        if start_idx >= len(msg.poses) - 1:
            return msg.poses[-1], len(msg.poses) - 1, "tail"

        remain_m = 0.0
        for idx in range(start_idx, len(msg.poses) - 1):
            remain_m += self._dist(msg.poses[idx], msg.poses[idx + 1])
        if (
            source in ("local", "avoidance")
            and self.local_goal_tail_backoff_m > 1e-6
            and remain_m <= self.local_goal_tail_backoff_trigger_m
        ):
            backoff_idx = self._goal_idx_backed_off_from_tail(msg, start_idx)
            if backoff_idx is not None and backoff_idx < len(msg.poses):
                return msg.poses[backoff_idx], backoff_idx, "tail_backoff"
        if remain_m <= self.final_goal_switch_distance_m:
            return msg.poses[-1], len(msg.poses) - 1, "tail"

        accum_m = 0.0
        for idx in range(start_idx + 1, len(msg.poses)):
            accum_m += self._dist(msg.poses[idx - 1], msg.poses[idx])
            if accum_m >= self.goal_lookahead_m:
                return msg.poses[idx], idx, "lookahead"
        return msg.poses[-1], len(msg.poses) - 1, "tail"

    def _publish_goal_for_selected_path(self, source, msg):
        if self.goal_pub is None or msg is None or not msg.poses:
            return

        goal_pose, goal_idx, goal_reason = self._goal_from_selected_path(source, msg)
        if goal_pose is None:
            return

        goal_x = float(goal_pose.pose.position.x)
        goal_y = float(goal_pose.pose.position.y)
        goal_yaw = self._path_pose_yaw(msg, goal_idx)
        if self._last_goal_sig is not None:
            prev_source, prev_frame, prev_x, prev_y, prev_yaw = self._last_goal_sig
            if (
                prev_source == source
                and prev_frame == msg.header.frame_id
                and math.hypot(goal_x - prev_x, goal_y - prev_y) < self.goal_update_min_dist_m
                and abs(self._angle_diff(goal_yaw, prev_yaw)) < math.radians(10.0)
            ):
                return
        self._last_goal_sig = (source, msg.header.frame_id, goal_x, goal_y, goal_yaw)

        goal = PoseStamped()
        goal.header = msg.header
        goal.pose.position.x = goal_x
        goal.pose.position.y = goal_y
        goal.pose.position.z = float(goal_pose.pose.position.z)
        self._set_pose_yaw(goal, goal_yaw)
        self.goal_pub.publish(goal)
        rospy.loginfo_throttle(
            1.0,
            "astar_path_to_teb_via_points: synced move_base goal | topic=%s source=%s reason=%s idx=%d x=%.2f y=%.2f",
            self.goal_output_topic,
            source,
            goal_reason,
            goal_idx,
            goal_x,
            goal_y,
        )

    def _is_fresh(self, stamp_sec, timeout_s):
        return stamp_sec > 0.0 and (rospy.get_time() - stamp_sec) <= timeout_s

    def _has_fresh_empty_local_path(self):
        if not (
            self.respect_local_hold
            and bool(self.local_input_topic)
            and self._local_msg is not None
            and len(self._local_msg.poses) < 2
            and self._is_fresh(self._local_rx_time, self.local_path_timeout_s)
        ):
            return False
        return True

    def _has_fresh_local_hold(self):
        if not self._has_fresh_empty_local_path():
            return False

        if not self.hold_requires_avoidance_path:
            return True

        return (
            bool(self.avoidance_input_topic)
            and self._avoidance_msg is not None
            and len(self._avoidance_msg.poses) >= 2
            and self._is_fresh(self._avoidance_rx_time, self.avoidance_path_timeout_s)
        )

    @staticmethod
    def _empty_path_like(msg):
        out = Path()
        if msg is not None:
            out.header = msg.header
        else:
            out.header.stamp = rospy.Time.now()
            out.header.frame_id = "map"
        return out

    def _pick_path(self):
        if (
            self.avoidance_input_topic
            and self._avoidance_msg is not None
            and len(self._avoidance_msg.poses) >= 2
            and self._is_fresh(self._avoidance_rx_time, self.avoidance_path_timeout_s)
        ):
            return "avoidance", self._avoidance_msg

        if (
            self.local_input_topic
            and self._local_msg is not None
            and len(self._local_msg.poses) >= 2
            and self._is_fresh(self._local_rx_time, self.local_path_timeout_s)
        ):
            return "local", self._local_msg

        if self._has_fresh_local_hold():
            return "local_hold", self._local_msg

        if self._has_fresh_empty_local_path():
            return "local_hold", self._local_msg

        if self._fallback_msg is not None and len(self._fallback_msg.poses) >= 2:
            return "fallback", self._fallback_msg

        return None, None

    def _publish_selected(self):
        source, msg = self._pick_path()
        if source is None:
            return

        sig = self._path_signature(msg) if msg is not None else ()
        key = (source, sig)
        if key == self._last_key and sig == self._last_sig:
            if source != "local_hold" and msg is not None:
                self._publish_goal_for_selected_path(source, msg)
            return
        self._last_key = key
        self._last_sig = sig

        if source == "local_hold":
            self.pub.publish(self._empty_path_like(msg))
            # Force a fresh goal sync when a valid path resumes after hold.
            self._last_goal_sig = None
            self._reset_progress_if_needed("local")
            self._reset_progress_if_needed("avoidance")
            rospy.loginfo_throttle(
                1.0,
                "astar_path_to_teb_via_points: source=%s published empty via path and paused goal updates",
                source,
            )
            return

        if msg is None:
            return

        via = self._downsample(msg)
        if not via.poses:
            rospy.logwarn_throttle(
                2.0,
                "astar_path_to_teb_via_points: selected source '%s' produced empty path",
                source,
            )
            return
        self.pub.publish(via)
        self._publish_goal_for_selected_path(source, msg)
        rospy.loginfo_throttle(
            1.0,
            "astar_path_to_teb_via_points: source=%s published %d via points from %d path poses",
            source,
            len(via.poses),
            len(msg.poses),
        )


def main():
    rospy.init_node("astar_path_to_teb_via_points", anonymous=False)
    AStarPathToTebViaPoints()
    rospy.spin()


if __name__ == "__main__":
    main()
