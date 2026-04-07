#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rospy
import tf.transformations as transformations
from nav_msgs.msg import OccupancyGrid, Odometry

from dynamic_window_approach.msg import (
    BehaviorCommand,
    ExplainabilityEvent,
    TrackedObjectArray,
)


class DynamicRiskManager:
    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic", "/lio_localizer/odometry/optimization")
        self.objects_topic = rospy.get_param("~tracked_objects_topic", "/perception/tracked_objects")
        self.drivable_grid_topic = rospy.get_param("~drivable_grid_topic", "/lio_sam/drivable_area/grid")
        self.behavior_cmd_topic = rospy.get_param("~behavior_cmd_topic", "/planning/behavior_cmd")
        self.risk_grid_topic = rospy.get_param("~risk_grid_topic", "/planning/dynamic_risk_grid")

        self.horizon_s = max(0.2, float(rospy.get_param("~prediction_horizon_s", 2.5)))
        self.step_s = max(0.05, float(rospy.get_param("~prediction_step_s", 0.2)))
        self.inflate_m = max(0.1, float(rospy.get_param("~prediction_inflate_m", 0.8)))
        self.pedestrian_inflate_extra_m = max(
            0.0, float(rospy.get_param("~pedestrian_inflate_extra_m", 0.0))
        )

        self.vehicle_stop_ttc_s = max(0.1, float(rospy.get_param("~vehicle_stop_ttc_s", 2.8)))
        self.ped_stop_ttc_s = max(0.1, float(rospy.get_param("~pedestrian_stop_ttc_s", 3.5)))
        self.caution_ttc_s = max(0.1, float(rospy.get_param("~caution_ttc_s", 5.0)))
        self.caution_speed_mps = max(0.05, float(rospy.get_param("~caution_speed_mps", 0.25)))
        self.pedestrian_caution_speed_mps = max(
            0.05,
            min(
                self.caution_speed_mps,
                float(rospy.get_param("~pedestrian_caution_speed_mps", 0.15)),
            ),
        )
        self.default_speed_mps = max(0.05, float(rospy.get_param("~default_speed_mps", 0.55)))
        self.front_lateral_stop_m = max(0.1, float(rospy.get_param("~front_lateral_stop_m", 1.8)))
        self.front_lateral_caution_m = max(0.1, float(rospy.get_param("~front_lateral_caution_m", 2.5)))
        self.pedestrian_stop_distance_m = max(
            0.1, float(rospy.get_param("~pedestrian_stop_distance_m", 1.05))
        )
        self.pedestrian_caution_distance_m = max(
            self.pedestrian_stop_distance_m,
            float(rospy.get_param("~pedestrian_caution_distance_m", 2.00)),
        )
        self.dynamic_speed_thresh_mps = max(
            0.01, float(rospy.get_param("~dynamic_speed_thresh_mps", 0.15))
        )
        self.behavior_stop_on_count = max(
            1, int(rospy.get_param("~behavior_stop_on_count", 2))
        )
        self.behavior_stop_off_count = max(
            1, int(rospy.get_param("~behavior_stop_off_count", 6))
        )
        self.behavior_stop_hold_s = max(
            0.0, float(rospy.get_param("~behavior_stop_hold_s", 0.8))
        )
        self.behavior_caution_on_count = max(
            1, int(rospy.get_param("~behavior_caution_on_count", 2))
        )
        self.behavior_caution_off_count = max(
            1, int(rospy.get_param("~behavior_caution_off_count", 4))
        )
        self.behavior_caution_hold_s = max(
            0.0, float(rospy.get_param("~behavior_caution_hold_s", 0.6))
        )
        self.include_static_in_behavior = bool(
            rospy.get_param("~include_static_objects_in_behavior", False)
        )
        self.include_static_in_risk_grid = bool(
            rospy.get_param("~include_static_objects_in_risk_grid", False)
        )
        self.debug_risk_logging = bool(rospy.get_param("~debug_risk_logging", True))
        self.debug_risk_log_period_s = max(
            0.1, float(rospy.get_param("~debug_risk_log_period_s", 1.0))
        )

        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.odom_vx = 0.0
        self.odom_vy = 0.0
        self.have_odom = False

        self.objects = []
        self.grid_msg = None
        self.behavior_state = "clear"
        self.behavior_state_since = rospy.Time(0)
        self.behavior_state_reason = "clear"
        self.behavior_raw_state = "clear"
        self.behavior_raw_count = 0
        self.explainability_topic = rospy.get_param(
            "~explainability_topic", "/planning/explainability"
        )
        self._last_explain_key = None
        self._last_explain_time = 0.0

        self.pub_behavior = rospy.Publisher(self.behavior_cmd_topic, BehaviorCommand, queue_size=5)
        self.pub_risk_grid = rospy.Publisher(self.risk_grid_topic, OccupancyGrid, queue_size=1)
        self.pub_explainability = rospy.Publisher(
            self.explainability_topic, ExplainabilityEvent, queue_size=20
        )

        self.sub_odom = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=20)
        self.sub_objects = rospy.Subscriber(self.objects_topic, TrackedObjectArray, self.objects_callback, queue_size=5)
        self.sub_grid = rospy.Subscriber(self.drivable_grid_topic, OccupancyGrid, self.grid_callback, queue_size=3)

        self.timer = rospy.Timer(rospy.Duration(0.1), self.on_timer)
        rospy.loginfo(
            "dynamic_risk_manager started | objects=%s behavior=%s risk_grid=%s",
            self.objects_topic,
            self.behavior_cmd_topic,
            self.risk_grid_topic,
        )

    @staticmethod
    def _twist_to_world(msg, yaw):
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        header_frame = str(msg.header.frame_id).strip()
        child_frame = str(msg.child_frame_id).strip()
        if child_frame and child_frame != header_frame:
            c = math.cos(yaw)
            s = math.sin(yaw)
            return c * vx - s * vy, s * vx + c * vy
        return vx, vy

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.odom_x = float(p.x)
        self.odom_y = float(p.y)
        self.odom_yaw = transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        self.odom_vx, self.odom_vy = self._twist_to_world(msg, self.odom_yaw)
        self.have_odom = True

    def objects_callback(self, msg):
        self.objects = list(msg.objects)

    def grid_callback(self, msg):
        self.grid_msg = msg

    def _world_to_robot(self, x, y):
        dx = x - self.odom_x
        dy = y - self.odom_y
        c = math.cos(self.odom_yaw)
        s = math.sin(self.odom_yaw)
        rx = c * dx + s * dy
        ry = -s * dx + c * dy
        return rx, ry

    def _is_pedestrian(self, label):
        t = (label or "").lower()
        return ("ped" in t) or ("person" in t) or ("walker" in t)

    @staticmethod
    def _object_speed(obj):
        return math.hypot(float(obj.twist.linear.x), float(obj.twist.linear.y))

    def _is_dynamic_object(self, obj):
        label = (obj.label or "").lower()
        if label.startswith("static_"):
            return False
        return self._object_speed(obj) >= self.dynamic_speed_thresh_mps

    @staticmethod
    def _behavior_priority(state):
        if state == "stop":
            return 2
        if state == "caution":
            return 1
        return 0

    def _behavior_on_count(self, state):
        if state == "stop":
            return self.behavior_stop_on_count
        if state == "caution":
            return self.behavior_caution_on_count
        return 1

    def _behavior_off_count(self, state):
        if state == "stop":
            return self.behavior_stop_off_count
        if state == "caution":
            return self.behavior_caution_off_count
        return 1

    def _behavior_hold_s(self, state):
        if state == "stop":
            return self.behavior_stop_hold_s
        if state == "caution":
            return self.behavior_caution_hold_s
        return 0.0

    def _update_behavior_state(self, raw_state, raw_reason):
        now = rospy.Time.now()
        if raw_state == self.behavior_raw_state:
            self.behavior_raw_count += 1
        else:
            self.behavior_raw_state = raw_state
            self.behavior_raw_count = 1

        current_prio = self._behavior_priority(self.behavior_state)
        raw_prio = self._behavior_priority(raw_state)

        if raw_state == self.behavior_state:
            if raw_state != "clear":
                self.behavior_state_reason = raw_reason
            return

        if raw_prio > current_prio:
            if self.behavior_raw_count >= self._behavior_on_count(raw_state):
                self.behavior_state = raw_state
                self.behavior_state_since = now
                self.behavior_state_reason = raw_reason
            return

        held_s = (now - self.behavior_state_since).to_sec() if self.behavior_state_since.to_sec() > 0.0 else float("inf")
        if held_s < self._behavior_hold_s(self.behavior_state):
            return

        if self.behavior_raw_count >= self._behavior_off_count(self.behavior_state):
            self.behavior_state = raw_state
            self.behavior_state_since = now
            self.behavior_state_reason = raw_reason if raw_state != "clear" else "clear"

    def _make_behavior_cmd(self, state, reason):
        cmd = BehaviorCommand()
        cmd.header.stamp = rospy.Time.now()
        cmd.stop = False
        cmd.speed_limit = float(self.default_speed_mps)
        cmd.reason = reason
        if state == "stop":
            cmd.stop = True
            cmd.speed_limit = 0.0
        elif state == "caution":
            cmd.stop = False
            cmd.speed_limit = min(cmd.speed_limit, self.caution_speed_mps)
            lower_reason = (reason or "").lower()
            if ("ped" in lower_reason) or ("person" in lower_reason):
                cmd.speed_limit = min(cmd.speed_limit, self.pedestrian_caution_speed_mps)
        return cmd

    def _publish_explainability(
        self,
        event_type,
        stamp=None,
        trigger_reason="",
        action_taken="",
        local_planning_active=False,
        stop_commanded=False,
        slowdown_commanded=False,
        speed_limit_mps=-1.0,
        closest_obstacle_dist_m=-1.0,
        obstacle_lateral_offset_m=-1.0,
        ttc_s=-1.0,
        tracked_object_id=-1,
        tracked_object_label="",
        summary_text="",
    ):
        msg = ExplainabilityEvent()
        msg.header.stamp = stamp if stamp is not None else rospy.Time.now()
        msg.source_node = "dynamic_risk_manager"
        msg.event_type = str(event_type)
        msg.decision_layer = "behavior_layer"
        msg.trigger_reason = str(trigger_reason)
        msg.action_taken = str(action_taken)
        msg.avoid_direction = "none"
        msg.local_planning_active = bool(local_planning_active)
        msg.stop_commanded = bool(stop_commanded)
        msg.slowdown_commanded = bool(slowdown_commanded)
        msg.speed_before_mps = -1.0
        msg.speed_after_mps = -1.0
        msg.speed_limit_mps = float(speed_limit_mps)
        msg.closest_obstacle_dist_m = float(closest_obstacle_dist_m)
        msg.obstacle_lateral_offset_m = float(obstacle_lateral_offset_m)
        msg.ttc_s = float(ttc_s)
        msg.tracked_object_id = int(tracked_object_id)
        msg.tracked_object_label = str(tracked_object_label)
        msg.summary_text = str(summary_text)

        key = (
            msg.event_type,
            msg.trigger_reason,
            msg.action_taken,
            msg.stop_commanded,
            msg.slowdown_commanded,
            round(float(msg.speed_limit_mps), 2),
            round(float(msg.ttc_s), 2),
            msg.tracked_object_id,
            msg.tracked_object_label,
        )
        if key == self._last_explain_key:
            return
        stamp_sec = msg.header.stamp.to_sec() if msg.header.stamp.to_sec() > 0.0 else rospy.get_time()
        self._last_explain_key = key
        self._last_explain_time = stamp_sec
        self.pub_explainability.publish(msg)

    def _evaluate_behavior(self):
        if not self.have_odom:
            self._update_behavior_state("clear", "clear")
            return self._make_behavior_cmd(self.behavior_state, self.behavior_state_reason), None

        best_ttc = None
        stop_reason = ""
        stop_obj_id = -1
        stop_obj_label = ""
        stop_rx = -1.0
        stop_ry = 0.0
        stop_dist = -1.0
        caution = False
        caution_reason = ""
        caution_ttc = None
        caution_obj_id = -1
        caution_obj_label = ""
        caution_rx = -1.0
        caution_ry = 0.0
        caution_dist = -1.0

        for obj in self.objects:
            if (not self.include_static_in_behavior) and (not self._is_dynamic_object(obj)):
                continue
            ox = float(obj.pose.position.x)
            oy = float(obj.pose.position.y)
            ovx = float(obj.twist.linear.x)
            ovy = float(obj.twist.linear.y)
            rx, ry = self._world_to_robot(ox, oy)
            if rx < -1.0:
                continue

            # Relative velocity in world; project towards object line-of-sight.
            rvx = ovx - self.odom_vx
            rvy = ovy - self.odom_vy
            dist = max(1e-3, math.hypot(ox - self.odom_x, oy - self.odom_y))
            closing = -((ox - self.odom_x) * rvx + (oy - self.odom_y) * rvy) / dist
            is_ped = self._is_pedestrian(obj.label)
            ttc = float("inf")
            if closing > 1e-3:
                ttc = dist / closing
            stop_ttc = self.ped_stop_ttc_s if is_ped else self.vehicle_stop_ttc_s
            lateral_stop = self.front_lateral_stop_m
            lateral_caution = self.front_lateral_caution_m

            stop_hit = rx >= 0.0 and abs(ry) <= lateral_stop and (
                ttc <= stop_ttc or (is_ped and dist <= self.pedestrian_stop_distance_m)
            )
            caution_hit = rx >= 0.0 and abs(ry) <= lateral_caution and (
                ttc <= self.caution_ttc_s or (is_ped and dist <= self.pedestrian_caution_distance_m)
            )

            if stop_hit:
                if (best_ttc is None) or (ttc < best_ttc):
                    best_ttc = ttc
                    if is_ped and dist <= self.pedestrian_stop_distance_m and ttc > stop_ttc:
                        stop_reason = "ped_distance_stop:{:.2f}m".format(dist)
                    elif is_ped and not math.isfinite(ttc):
                        stop_reason = "ped_distance_stop:{:.2f}m".format(dist)
                    else:
                        stop_reason = "ttc_stop:{}:{:.2f}s".format(
                            obj.label if obj.label else "obj", ttc
                        )
                    stop_obj_id = int(obj.id)
                    stop_obj_label = str(obj.label)
                    stop_rx = float(rx)
                    stop_ry = float(ry)
                    stop_dist = float(dist)
            elif caution_hit:
                caution = True
                if is_ped and dist <= self.pedestrian_caution_distance_m and ttc > self.caution_ttc_s:
                    candidate_caution_reason = "ped_distance_caution:{:.2f}m".format(dist)
                elif is_ped and not math.isfinite(ttc):
                    candidate_caution_reason = "ped_distance_caution:{:.2f}m".format(dist)
                else:
                    candidate_caution_reason = "ttc_caution"
                if caution_ttc is None or ttc < caution_ttc:
                    caution_ttc = ttc
                    caution_reason = candidate_caution_reason
                    caution_obj_id = int(obj.id)
                    caution_obj_label = str(obj.label)
                    caution_rx = float(rx)
                    caution_ry = float(ry)
                    caution_dist = float(dist)

        raw_state = "clear"
        raw_reason = "clear"
        if best_ttc is not None:
            raw_state = "stop"
            raw_reason = stop_reason
        elif caution:
            raw_state = "caution"
            raw_reason = caution_reason if caution_reason else "ttc_caution"

        self._update_behavior_state(raw_state, raw_reason)
        cmd = self._make_behavior_cmd(self.behavior_state, self.behavior_state_reason)
        cmd.header.stamp = rospy.Time.now()
        event_meta = {
            "raw_state": raw_state,
            "raw_reason": raw_reason,
            "behavior_state": self.behavior_state,
            "behavior_reason": self.behavior_state_reason,
            "stop_ttc": float(best_ttc) if best_ttc is not None else -1.0,
            "stop_obj_id": stop_obj_id,
            "stop_obj_label": stop_obj_label,
            "stop_rx": stop_rx,
            "stop_ry": stop_ry,
            "stop_dist": stop_dist,
            "caution_ttc": float(caution_ttc) if caution_ttc is not None else -1.0,
            "caution_obj_id": caution_obj_id,
            "caution_obj_label": caution_obj_label,
            "caution_rx": caution_rx,
            "caution_ry": caution_ry,
            "caution_dist": caution_dist,
        }
        return cmd, event_meta

    @staticmethod
    def _mark_disk(data, width, height, cx, cy, rad_cells, value):
        rr = rad_cells * rad_cells
        for dx in range(-rad_cells, rad_cells + 1):
            for dy in range(-rad_cells, rad_cells + 1):
                if dx * dx + dy * dy > rr:
                    continue
                x = cx + dx
                y = cy + dy
                if x < 0 or y < 0 or x >= width or y >= height:
                    continue
                idx = y * width + x
                if value > data[idx]:
                    data[idx] = value

    def _build_risk_grid(self):
        if self.grid_msg is None:
            return None
        g = self.grid_msg
        out = OccupancyGrid()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = g.header.frame_id if g.header.frame_id else "map"
        out.info = g.info
        w = int(g.info.width)
        h = int(g.info.height)
        res = float(g.info.resolution)
        ox = float(g.info.origin.position.x)
        oy = float(g.info.origin.position.y)
        data = [0] * (w * h)
        steps = max(1, int(math.floor(self.horizon_s / self.step_s)))
        for obj in self.objects:
            if (not self.include_static_in_risk_grid) and (not self._is_dynamic_object(obj)):
                continue
            obj_inflate_m = self.inflate_m
            if self._is_pedestrian(obj.label):
                obj_inflate_m += self.pedestrian_inflate_extra_m
            rad_cells = max(1, int(math.ceil(obj_inflate_m / max(1e-3, res))))
            x0 = float(obj.pose.position.x)
            y0 = float(obj.pose.position.y)
            vx = float(obj.twist.linear.x)
            vy = float(obj.twist.linear.y)
            for k in range(steps + 1):
                t = k * self.step_s
                x = x0 + vx * t
                y = y0 + vy * t
                gx = int(math.floor((x - ox) / res))
                gy = int(math.floor((y - oy) / res))
                value = max(20, 100 - int(80.0 * (t / max(1e-3, self.horizon_s))))
                self._mark_disk(data, w, h, gx, gy, rad_cells, value)

        out.data = data
        return out

    def on_timer(self, _evt):
        try:
            cmd, event_meta = self._evaluate_behavior()
            self.pub_behavior.publish(cmd)
            if event_meta is not None:
                behavior_state = str(event_meta["behavior_state"])
                if behavior_state == "stop":
                    stop_reason = str(event_meta["behavior_reason"])
                    if stop_reason.startswith("ped_distance_stop:"):
                        summary_text = (
                            "Dynamic risk manager requested a stop because '{}' entered the pedestrian distance stop zone at {:.2f}m."
                        ).format(
                            event_meta["stop_obj_label"] if event_meta["stop_obj_label"] else "pedestrian",
                            max(0.0, float(event_meta["stop_dist"])),
                        )
                    else:
                        summary_text = (
                            "Dynamic risk manager requested a stop because '{}' reached TTC {:.2f}s."
                        ).format(
                            event_meta["stop_obj_label"] if event_meta["stop_obj_label"] else "object",
                            max(0.0, float(event_meta["stop_ttc"])),
                        )
                    self._publish_explainability(
                        event_type="BEHAVIOR_STATE_CHANGE",
                        stamp=cmd.header.stamp,
                        trigger_reason=event_meta["behavior_reason"],
                        action_taken="stop",
                        stop_commanded=True,
                        speed_limit_mps=float(cmd.speed_limit),
                        closest_obstacle_dist_m=float(event_meta["stop_rx"]),
                        obstacle_lateral_offset_m=float(event_meta["stop_ry"]),
                        ttc_s=float(event_meta["stop_ttc"]),
                        tracked_object_id=int(event_meta["stop_obj_id"]),
                        tracked_object_label=event_meta["stop_obj_label"],
                        summary_text=summary_text,
                    )
                elif behavior_state == "caution":
                    caution_reason = str(event_meta["behavior_reason"])
                    if caution_reason.startswith("ped_distance_caution:"):
                        summary_text = (
                            "Dynamic risk manager requested a slowdown because '{}' entered the pedestrian distance caution zone at {:.2f}m."
                        ).format(
                            event_meta["caution_obj_label"] if event_meta["caution_obj_label"] else "pedestrian",
                            max(0.0, float(event_meta["caution_dist"])),
                        )
                    else:
                        summary_text = (
                            "Dynamic risk manager requested a slowdown because '{}' entered the caution TTC zone."
                        ).format(
                            event_meta["caution_obj_label"] if event_meta["caution_obj_label"] else "object"
                        )
                    self._publish_explainability(
                        event_type="BEHAVIOR_STATE_CHANGE",
                        stamp=cmd.header.stamp,
                        trigger_reason=event_meta["behavior_reason"],
                        action_taken="slowdown",
                        slowdown_commanded=True,
                        speed_limit_mps=float(cmd.speed_limit),
                        closest_obstacle_dist_m=float(event_meta["caution_rx"]),
                        obstacle_lateral_offset_m=float(event_meta["caution_ry"]),
                        ttc_s=float(event_meta["caution_ttc"]),
                        tracked_object_id=int(event_meta["caution_obj_id"]),
                        tracked_object_label=event_meta["caution_obj_label"],
                        summary_text=summary_text,
                    )
                else:
                    self._publish_explainability(
                        event_type="BEHAVIOR_STATE_CHANGE",
                        stamp=cmd.header.stamp,
                        trigger_reason=event_meta["behavior_reason"],
                        action_taken="clear",
                        speed_limit_mps=float(cmd.speed_limit),
                        summary_text="Dynamic risk manager cleared the stop/slowdown state.",
                    )
            risk_grid = self._build_risk_grid()
            if risk_grid is not None:
                self.pub_risk_grid.publish(risk_grid)
            if self.debug_risk_logging:
                moving_count = sum(1 for obj in self.objects if self._is_dynamic_object(obj))
                static_count = max(0, len(self.objects) - moving_count)
                rospy.loginfo_throttle(
                    self.debug_risk_log_period_s,
                    "dynamic_risk_manager: objects=%d moving=%d static=%d risk_static=%s behavior_static=%s raw=%s(%d) latched=%s stop=%s speed_limit=%.2f reason=%s",
                    len(self.objects),
                    moving_count,
                    static_count,
                    "on" if self.include_static_in_risk_grid else "off",
                    "on" if self.include_static_in_behavior else "off",
                    self.behavior_raw_state,
                    self.behavior_raw_count,
                    self.behavior_state,
                    "yes" if cmd.stop else "no",
                    float(cmd.speed_limit),
                    cmd.reason,
                )
        except Exception as e:
            rospy.logwarn_throttle(1.0, "dynamic_risk_manager error: %s", str(e))


def main():
    rospy.init_node("dynamic_risk_manager", anonymous=False)
    DynamicRiskManager()
    rospy.spin()


if __name__ == "__main__":
    main()
