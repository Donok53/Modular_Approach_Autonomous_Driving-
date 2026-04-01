#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
        self.goal_update_min_dist_m = max(
            0.0, float(rospy.get_param("~goal_update_min_dist_m", 0.50))
        )
        self.min_spacing_m = max(
            0.0, float(rospy.get_param("~min_spacing_m", 0.50))
        )
        self.max_points = max(2, int(rospy.get_param("~max_points", 200)))

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
            "astar_path_to_teb_via_points started | fallback=%s local=%s avoidance=%s odom=%s out=%s goal_out=%s goal_lookahead=%.2fm spacing=%.2fm max_points=%d",
            self.input_topic,
            self.local_input_topic if self.local_input_topic else "-",
            self.avoidance_input_topic if self.avoidance_input_topic else "-",
            self.odom_topic if self.odom_topic else "-",
            self.output_topic,
            self.goal_output_topic if self.goal_output_topic else "-",
            self.goal_lookahead_m,
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

    def _downsample(self, msg):
        if len(msg.poses) <= 2 or self.min_spacing_m <= 1e-6:
            out = Path()
            out.header = msg.header
            out.poses = list(msg.poses[: self.max_points])
            return out

        out = Path()
        out.header = msg.header
        out.poses.append(msg.poses[0])
        last = msg.poses[0]

        for pose in msg.poses[1:-1]:
            if self._dist(last, pose) < self.min_spacing_m:
                continue
            out.poses.append(pose)
            last = pose
            if len(out.poses) >= (self.max_points - 1):
                break

        if msg.poses[-1] is not out.poses[-1]:
            out.poses.append(msg.poses[-1])

        if len(out.poses) > self.max_points:
            step = max(1, len(out.poses) // max(1, self.max_points - 1))
            reduced = out.poses[::step]
            if reduced[-1] is not out.poses[-1]:
                reduced.append(out.poses[-1])
            out.poses = reduced[: self.max_points]
            if out.poses[-1] is not msg.poses[-1]:
                out.poses[-1] = msg.poses[-1]
        return out

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

    def _nearest_pose_index(self, msg):
        if msg is None or not msg.poses:
            return 0
        if not self._have_odom:
            return 0

        best_idx = 0
        best_dist_sq = float("inf")
        for idx, pose in enumerate(msg.poses):
            dx = float(pose.pose.position.x) - self._odom_x
            dy = float(pose.pose.position.y) - self._odom_y
            dist_sq = dx * dx + dy * dy
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_idx = idx
        return best_idx

    def _goal_from_selected_path(self, msg):
        if msg is None or not msg.poses:
            return None, None

        if self.goal_lookahead_m <= 1e-6 or len(msg.poses) == 1:
            return msg.poses[-1], len(msg.poses) - 1

        start_idx = self._nearest_pose_index(msg)
        if start_idx >= len(msg.poses) - 1:
            return msg.poses[-1], len(msg.poses) - 1

        accum_m = 0.0
        for idx in range(start_idx + 1, len(msg.poses)):
            accum_m += self._dist(msg.poses[idx - 1], msg.poses[idx])
            if accum_m >= self.goal_lookahead_m:
                return msg.poses[idx], idx
        return msg.poses[-1], len(msg.poses) - 1

    def _publish_goal_for_selected_path(self, source, msg):
        if self.goal_pub is None or msg is None or not msg.poses:
            return

        goal_pose, goal_idx = self._goal_from_selected_path(msg)
        if goal_pose is None:
            return

        goal_x = float(goal_pose.pose.position.x)
        goal_y = float(goal_pose.pose.position.y)
        if self._last_goal_sig is not None:
            prev_source, prev_frame, prev_x, prev_y = self._last_goal_sig
            if (
                prev_source == source
                and prev_frame == msg.header.frame_id
                and math.hypot(goal_x - prev_x, goal_y - prev_y) < self.goal_update_min_dist_m
            ):
                return
        self._last_goal_sig = (source, msg.header.frame_id, goal_x, goal_y)

        goal = PoseStamped()
        goal.header = msg.header
        goal.pose = goal_pose.pose
        self.goal_pub.publish(goal)
        rospy.loginfo_throttle(
            1.0,
            "astar_path_to_teb_via_points: synced move_base goal | topic=%s source=%s idx=%d x=%.2f y=%.2f",
            self.goal_output_topic,
            source,
            goal_idx,
            goal_x,
            goal_y,
        )

    def _is_fresh(self, stamp_sec, timeout_s):
        return stamp_sec > 0.0 and (rospy.get_time() - stamp_sec) <= timeout_s

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

        if self._fallback_msg is not None and len(self._fallback_msg.poses) >= 2:
            return "fallback", self._fallback_msg

        return None, None

    def _publish_selected(self):
        source, msg = self._pick_path()
        if msg is None:
            return

        sig = self._path_signature(msg)
        key = (source, sig)
        if key == self._last_key and sig == self._last_sig:
            return
        self._last_key = key
        self._last_sig = sig

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
