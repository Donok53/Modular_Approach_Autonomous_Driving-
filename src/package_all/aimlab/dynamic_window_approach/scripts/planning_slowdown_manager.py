#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rospy
from dynamic_window_approach.msg import BehaviorCommand
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker


class PlanningSlowdownManager:
    def __init__(self):
        self.behavior_cmd_topic = rospy.get_param("~behavior_cmd_topic", "/planning/behavior_cmd")
        self.global_path_topic = rospy.get_param("~global_path_topic", "/astar/path")
        self.path_is_fallback_topic = rospy.get_param("~path_is_fallback_topic", "/astar/path_is_fallback")
        self.path_blocked_topic = rospy.get_param("~path_blocked_topic", "/astar/path_blocked")
        self.caution_topic = str(
            rospy.get_param("~caution_topic", "/planning/global_obstacle_caution")
        ).strip()
        self.goal_topic = rospy.get_param("~goal_topic", "/move_base_simple/goal")
        self.marker_topic = rospy.get_param("~marker_topic", "/planning/planning_speed_limit_marker")
        self.marker_frame = rospy.get_param("~marker_frame", "map")

        self.slow_speed_limit_mps = max(
            0.0, float(rospy.get_param("~slow_speed_limit_mps", 0.12))
        )
        self.clear_speed_limit_mps = max(
            self.slow_speed_limit_mps,
            float(rospy.get_param("~clear_speed_limit_mps", 99.0)),
        )
        self.hold_s = max(0.0, float(rospy.get_param("~hold_s", 2.0)))
        self.no_path_grace_s = max(0.0, float(rospy.get_param("~no_path_grace_s", 0.8)))
        self.caution_timeout_s = max(
            0.1, float(rospy.get_param("~caution_timeout_s", 1.2))
        )
        self.publish_hz = max(1.0, float(rospy.get_param("~publish_hz", 5.0)))

        self.fallback_active = False
        self.path_blocked_active = False
        self.caution_active = False
        self.path_has_poses = False
        self.active_goal = False
        self.last_goal_sec = 0.0
        self.last_caution_sec = 0.0
        self.last_path_sec = 0.0
        self.last_nonempty_path_sec = 0.0
        self.slow_until_sec = 0.0
        self.marker_x = 0.0
        self.marker_y = 0.0

        self.pub_behavior = rospy.Publisher(self.behavior_cmd_topic, BehaviorCommand, queue_size=5)
        self.pub_marker = rospy.Publisher(self.marker_topic, Marker, queue_size=2, latch=True)

        self.sub_fallback = rospy.Subscriber(
            self.path_is_fallback_topic, Bool, self.fallback_callback, queue_size=5
        )
        self.sub_path_blocked = rospy.Subscriber(
            self.path_blocked_topic, Bool, self.path_blocked_callback, queue_size=5
        )
        self.sub_caution = None
        if self.caution_topic:
            self.sub_caution = rospy.Subscriber(
                self.caution_topic, Bool, self.caution_callback, queue_size=5
            )
        self.sub_path = rospy.Subscriber(
            self.global_path_topic, Path, self.path_callback, queue_size=5
        )
        self.sub_goal = rospy.Subscriber(
            self.goal_topic, PoseStamped, self.goal_callback, queue_size=2
        )
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_hz), self.timer_callback)

        rospy.loginfo(
            "planning_slowdown_manager started | path=%s fallback=%s blocked=%s caution=%s goal=%s behavior=%s marker=%s slow=%.2fm/s hold=%.1fs no_path_grace=%.1fs caution_timeout=%.1fs",
            self.global_path_topic,
            self.path_is_fallback_topic,
            self.path_blocked_topic,
            self.caution_topic if self.caution_topic else "-",
            self.goal_topic,
            self.behavior_cmd_topic,
            self.marker_topic,
            self.slow_speed_limit_mps,
            self.hold_s,
            self.no_path_grace_s,
            self.caution_timeout_s,
        )

    def fallback_callback(self, msg):
        self.fallback_active = bool(msg.data)

    def path_blocked_callback(self, msg):
        self.path_blocked_active = bool(msg.data)

    def caution_callback(self, msg):
        self.caution_active = bool(msg.data)
        self.last_caution_sec = rospy.Time.now().to_sec()

    def goal_callback(self, msg):
        self.active_goal = True
        self.last_goal_sec = rospy.Time.now().to_sec()
        self.marker_x = float(msg.pose.position.x)
        self.marker_y = float(msg.pose.position.y)

    def path_callback(self, msg):
        now_sec = rospy.Time.now().to_sec()
        self.last_path_sec = now_sec
        self.path_has_poses = bool(msg.poses)
        if self.path_has_poses:
            self.active_goal = True
            self.last_nonempty_path_sec = now_sec
            self.marker_x = float(msg.poses[0].pose.position.x)
            self.marker_y = float(msg.poses[0].pose.position.y)

    def _caution_is_fresh(self, now_sec):
        return (
            self.caution_active
            and self.last_caution_sec > 0.0
            and (now_sec - self.last_caution_sec) <= self.caution_timeout_s
        )

    def _slowdown_reason(self, now_sec):
        if self.fallback_active:
            return "observe: replanning fallback"
        if self.path_blocked_active:
            return "observe: global path blocked"
        if self._caution_is_fresh(now_sec):
            return "observe: distant obstacle evidence"

        new_goal_without_path = (
            self.active_goal
            and not self.path_has_poses
            and self.last_goal_sec > 0.0
            and self.last_goal_sec >= self.last_nonempty_path_sec
            and (now_sec - self.last_goal_sec) >= self.no_path_grace_s
        )
        if new_goal_without_path:
            return "observe: no global path"

        if now_sec < self.slow_until_sec:
            return "observe: hold"
        return ""

    def timer_callback(self, _event):
        now = rospy.Time.now()
        now_sec = now.to_sec()
        reason = self._slowdown_reason(now_sec)
        if reason and not reason.endswith("hold"):
            self.slow_until_sec = max(self.slow_until_sec, now_sec + self.hold_s)
        elif not reason and now_sec < self.slow_until_sec:
            reason = "observe: hold"

        cmd = BehaviorCommand()
        cmd.header.stamp = now
        cmd.stop = False
        if reason:
            cmd.speed_limit = float(self.slow_speed_limit_mps)
            cmd.reason = reason
        else:
            cmd.speed_limit = float(self.clear_speed_limit_mps)
            cmd.reason = "clear"
        self.pub_behavior.publish(cmd)
        self._publish_marker(now, cmd.speed_limit, cmd.reason)

    def _publish_marker(self, stamp, speed_limit, reason):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.marker_frame
        marker.ns = "planning_speed_limit"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = self.marker_x
        marker.pose.position.y = self.marker_y
        marker.pose.position.z = 1.2
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.35

        if reason == "clear":
            marker.color.r = 0.2
            marker.color.g = 0.95
            marker.color.b = 0.2
            marker.text = "Planning speed: normal"
        else:
            marker.color.r = 1.0
            marker.color.g = 0.55
            marker.color.b = 0.0
            marker.text = "Planning speed: %.2f m/s\n%s" % (
                speed_limit if math.isfinite(speed_limit) else 0.0,
                reason,
            )
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration(1.0 / self.publish_hz * 3.0)
        self.pub_marker.publish(marker)


if __name__ == "__main__":
    rospy.init_node("planning_slowdown_manager")
    PlanningSlowdownManager()
    rospy.spin()
