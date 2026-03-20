#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rospy
import tf.transformations as transformations
from nav_msgs.msg import OccupancyGrid, Odometry

from dynamic_window_approach.msg import BehaviorCommand, TrackedObjectArray


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

        self.vehicle_stop_ttc_s = max(0.1, float(rospy.get_param("~vehicle_stop_ttc_s", 2.8)))
        self.ped_stop_ttc_s = max(0.1, float(rospy.get_param("~pedestrian_stop_ttc_s", 3.5)))
        self.caution_ttc_s = max(0.1, float(rospy.get_param("~caution_ttc_s", 5.0)))
        self.caution_speed_mps = max(0.05, float(rospy.get_param("~caution_speed_mps", 0.25)))
        self.default_speed_mps = max(0.05, float(rospy.get_param("~default_speed_mps", 0.55)))
        self.front_lateral_stop_m = max(0.1, float(rospy.get_param("~front_lateral_stop_m", 1.8)))
        self.front_lateral_caution_m = max(0.1, float(rospy.get_param("~front_lateral_caution_m", 2.5)))
        self.dynamic_speed_thresh_mps = max(
            0.01, float(rospy.get_param("~dynamic_speed_thresh_mps", 0.15))
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

        self.pub_behavior = rospy.Publisher(self.behavior_cmd_topic, BehaviorCommand, queue_size=5)
        self.pub_risk_grid = rospy.Publisher(self.risk_grid_topic, OccupancyGrid, queue_size=1)

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

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.odom_x = float(p.x)
        self.odom_y = float(p.y)
        self.odom_yaw = transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        self.odom_vx = float(msg.twist.twist.linear.x)
        self.odom_vy = float(msg.twist.twist.linear.y)
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

    def _evaluate_behavior(self):
        cmd = BehaviorCommand()
        cmd.header.stamp = rospy.Time.now()
        cmd.stop = False
        cmd.speed_limit = float(self.default_speed_mps)
        cmd.reason = "clear"

        if not self.have_odom:
            return cmd

        best_ttc = None
        stop_reason = ""
        caution = False

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
            if closing <= 1e-3:
                continue
            ttc = dist / closing
            is_ped = self._is_pedestrian(obj.label)
            stop_ttc = self.ped_stop_ttc_s if is_ped else self.vehicle_stop_ttc_s
            lateral_stop = self.front_lateral_stop_m
            lateral_caution = self.front_lateral_caution_m

            if rx >= 0.0 and abs(ry) <= lateral_stop and ttc <= stop_ttc:
                if (best_ttc is None) or (ttc < best_ttc):
                    best_ttc = ttc
                    stop_reason = "ttc_stop:{}:{:.2f}s".format(obj.label if obj.label else "obj", ttc)
            elif rx >= 0.0 and abs(ry) <= lateral_caution and ttc <= self.caution_ttc_s:
                caution = True

        if best_ttc is not None:
            cmd.stop = True
            cmd.speed_limit = 0.0
            cmd.reason = stop_reason
            return cmd

        if caution:
            cmd.stop = False
            cmd.speed_limit = min(cmd.speed_limit, self.caution_speed_mps)
            cmd.reason = "ttc_caution"
        return cmd

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
        rad_cells = max(1, int(math.ceil(self.inflate_m / max(1e-3, res))))

        steps = max(1, int(math.floor(self.horizon_s / self.step_s)))
        for obj in self.objects:
            if (not self.include_static_in_risk_grid) and (not self._is_dynamic_object(obj)):
                continue
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
            cmd = self._evaluate_behavior()
            self.pub_behavior.publish(cmd)
            risk_grid = self._build_risk_grid()
            if risk_grid is not None:
                self.pub_risk_grid.publish(risk_grid)
            if self.debug_risk_logging:
                moving_count = sum(1 for obj in self.objects if self._is_dynamic_object(obj))
                static_count = max(0, len(self.objects) - moving_count)
                rospy.loginfo_throttle(
                    self.debug_risk_log_period_s,
                    "dynamic_risk_manager: objects=%d moving=%d static=%d risk_static=%s behavior_static=%s stop=%s speed_limit=%.2f reason=%s",
                    len(self.objects),
                    moving_count,
                    static_count,
                    "on" if self.include_static_in_risk_grid else "off",
                    "on" if self.include_static_in_behavior else "off",
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
