#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Empty


class DrivableArea2DEditor:
    def __init__(self):
        self.window_name = rospy.get_param("~window_name", "Drivable Area 2D Editor")
        self.grid_topic = rospy.get_param("~grid_topic", "/lio_sam/drivable_area/grid")
        self.odom_topic = rospy.get_param("~odom_topic", "/lio_sam/mapping/odometry")
        self.add_point_topic = rospy.get_param("~add_point_topic", "/lio_sam/drivable_area/add_point")
        self.erase_point_topic = rospy.get_param("~erase_point_topic", "/lio_sam/drivable_area/erase_point")
        self.clear_topic = rospy.get_param("~clear_topic", "/lio_sam/drivable_area/clear")
        self.undo_topic = rospy.get_param("~undo_topic", "/lio_sam/drivable_area/undo")
        self.save_topic = rospy.get_param("~save_topic", "/lio_sam/drivable_area/save")
        self.load_topic = rospy.get_param("~load_topic", "/lio_sam/drivable_area/load")

        self.mode = str(rospy.get_param("~mode", "toggle")).strip().lower()
        if self.mode not in ("toggle", "add", "erase"):
            self.mode = "toggle"

        self.max_display_px = max(400, int(rospy.get_param("~max_display_px", 1000)))
        self.max_scale = max(1.0, float(rospy.get_param("~max_scale", 6.0)))
        self.drag_min_step_px = max(1, int(rospy.get_param("~drag_min_step_px", 3)))
        self.paint_throttle_hz = max(1.0, float(rospy.get_param("~paint_throttle_hz", 30.0)))
        self.publish_unknown_in_toggle_as_add = bool(
            rospy.get_param("~publish_unknown_in_toggle_as_add", True)
        )

        self._lock = threading.RLock()
        self._grid_msg = None
        self._odom_xy = None
        self._display_scale = 1.0
        self._dragging = False
        self._last_drag_px = None
        self._last_pub_time = rospy.Time(0)

        self.pub_add = rospy.Publisher(self.add_point_topic, PointStamped, queue_size=50)
        self.pub_erase = rospy.Publisher(self.erase_point_topic, PointStamped, queue_size=50)
        self.pub_clear = rospy.Publisher(self.clear_topic, Empty, queue_size=2)
        self.pub_undo = rospy.Publisher(self.undo_topic, Empty, queue_size=2)
        self.pub_save = rospy.Publisher(self.save_topic, Empty, queue_size=2)
        self.pub_load = rospy.Publisher(self.load_topic, Empty, queue_size=2)

        self.sub_grid = rospy.Subscriber(self.grid_topic, OccupancyGrid, self.grid_callback, queue_size=1)
        self.sub_odom = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=20)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        rospy.on_shutdown(self.on_shutdown)

        rospy.loginfo(
            "drivable_area_2d_editor started | grid=%s, mode=%s, controls=[a/e/t, c/u/s/l, q]",
            self.grid_topic,
            self.mode,
        )

    def on_shutdown(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def grid_callback(self, msg):
        with self._lock:
            self._grid_msg = msg

    def odom_callback(self, msg):
        with self._lock:
            self._odom_xy = (
                float(msg.pose.pose.position.x),
                float(msg.pose.pose.position.y),
            )

    @staticmethod
    def _empty_canvas():
        img = np.zeros((480, 800, 3), dtype=np.uint8)
        cv2.putText(img, "Waiting for /lio_sam/drivable_area/grid ...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
        return img

    def _build_display_image(self):
        with self._lock:
            grid = self._grid_msg
            odom_xy = self._odom_xy

        if grid is None or grid.info.width <= 0 or grid.info.height <= 0:
            return self._empty_canvas(), None

        w = int(grid.info.width)
        h = int(grid.info.height)
        data = np.array(grid.data, dtype=np.int16).reshape((h, w))

        # Flip y for image top-left convention
        data_img = np.flipud(data)

        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[data_img < 0] = (35, 35, 35)        # unknown
        img[data_img == 0] = (40, 40, 220)      # drivable (red-ish, BGR)
        img[data_img > 0] = (80, 80, 80)        # reserved

        if w >= h:
            scale = min(self.max_scale, float(self.max_display_px) / float(max(1, w)))
        else:
            scale = min(self.max_scale, float(self.max_display_px) / float(max(1, h)))
        scale = max(1.0, scale)
        self._display_scale = scale

        disp_w = max(1, int(round(w * scale)))
        disp_h = max(1, int(round(h * scale)))
        disp = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)

        # Draw robot position if available
        if odom_xy is not None:
            gx_f = (odom_xy[0] - grid.info.origin.position.x) / grid.info.resolution
            gy_f = (odom_xy[1] - grid.info.origin.position.y) / grid.info.resolution
            if 0.0 <= gx_f < w and 0.0 <= gy_f < h:
                gy_img_f = (h - 1) - gy_f
                px = int(round(gx_f * scale))
                py = int(round(gy_img_f * scale))
                cv2.circle(disp, (px, py), max(2, int(round(3 * scale))), (255, 255, 255), 2)

        self._draw_hud(disp)
        return disp, grid

    def _draw_hud(self, disp):
        cv2.putText(
            disp,
            "Mode: {} | [a]add [e]erase [t]toggle | [c]clear [u]undo [s]save [l]load [q]quit".format(self.mode),
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

    def _pixel_to_world(self, px, py):
        with self._lock:
            grid = self._grid_msg
            scale = self._display_scale
        if grid is None or scale <= 0.0:
            return None

        w = int(grid.info.width)
        h = int(grid.info.height)
        gx = int(px / scale)
        gy_img = int(py / scale)
        gy = (h - 1) - gy_img

        if gx < 0 or gy < 0 or gx >= w or gy >= h:
            return None

        x = grid.info.origin.position.x + (gx + 0.5) * grid.info.resolution
        y = grid.info.origin.position.y + (gy + 0.5) * grid.info.resolution
        return (x, y, gx, gy, grid)

    def _toggle_action_for_cell(self, gx, gy, grid):
        idx = gy * int(grid.info.width) + gx
        if idx < 0 or idx >= len(grid.data):
            return "add"
        val = int(grid.data[idx])
        if val == 0:
            return "erase"
        if val < 0:
            return "add" if self.publish_unknown_in_toggle_as_add else "erase"
        return "erase"

    def _publish_point(self, x, y, action, frame_id):
        now = rospy.Time.now()
        dt = now - self._last_pub_time
        if dt.to_sec() < (1.0 / self.paint_throttle_hz):
            return
        self._last_pub_time = now

        msg = PointStamped()
        msg.header.stamp = now
        msg.header.frame_id = frame_id
        msg.point.x = float(x)
        msg.point.y = float(y)
        msg.point.z = 0.0
        if action == "erase":
            self.pub_erase.publish(msg)
        else:
            self.pub_add.publish(msg)

    def _handle_point_action(self, px, py, force_action=None):
        mapped = self._pixel_to_world(px, py)
        if mapped is None:
            return
        x, y, gx, gy, grid = mapped
        action = force_action if force_action is not None else self.mode
        if action == "toggle":
            action = self._toggle_action_for_cell(gx, gy, grid)
        self._publish_point(x, y, action, grid.header.frame_id or "map")

    def on_mouse(self, event, x, y, flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._dragging = True
            self._last_drag_px = (x, y)
            self._handle_point_action(x, y, None)
            return
        if event == cv2.EVENT_LBUTTONUP:
            self._dragging = False
            self._last_drag_px = None
            return
        if event == cv2.EVENT_RBUTTONDOWN:
            self._handle_point_action(x, y, "erase")
            return
        if event == cv2.EVENT_MBUTTONDOWN:
            self._handle_point_action(x, y, "add")
            return
        if event == cv2.EVENT_MOUSEMOVE and self._dragging:
            if self._last_drag_px is None:
                self._last_drag_px = (x, y)
                self._handle_point_action(x, y, None)
                return
            dx = x - self._last_drag_px[0]
            dy = y - self._last_drag_px[1]
            if (dx * dx + dy * dy) >= (self.drag_min_step_px * self.drag_min_step_px):
                self._last_drag_px = (x, y)
                self._handle_point_action(x, y, None)

    def _handle_key(self, key):
        if key == ord("a"):
            self.mode = "add"
        elif key == ord("e"):
            self.mode = "erase"
        elif key == ord("t"):
            self.mode = "toggle"
        elif key == ord("c"):
            self.pub_clear.publish(Empty())
        elif key == ord("u"):
            self.pub_undo.publish(Empty())
        elif key == ord("s"):
            self.pub_save.publish(Empty())
        elif key == ord("l"):
            self.pub_load.publish(Empty())
        elif key == ord("q") or key == 27:
            rospy.signal_shutdown("2D editor closed by user")

    def spin(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            disp, _ = self._build_display_image()
            cv2.imshow(self.window_name, disp)
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                self._handle_key(key)
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("drivable_area_2d_editor")
    try:
        node = DrivableArea2DEditor()
        node.spin()
    except rospy.ROSInterruptException:
        pass
