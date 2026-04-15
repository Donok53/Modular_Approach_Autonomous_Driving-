#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Empty, String


LABEL_COLORS = {
    -1: (20, 20, 24),
    0: (70, 72, 78),     # observed
    40: (70, 120, 200),  # road
    80: (210, 230, 210), # sidewalk
    100: (220, 70, 60),  # curb
}


class SemanticBEV2DEditor:
    def __init__(self):
        self.window_name = rospy.get_param("~window_name", "Semantic BEV 2D Editor")
        self.grid_topic = rospy.get_param("~grid_topic", "/semantic_bev_editor/grid")
        self.paint_point_topic = rospy.get_param("~paint_point_topic", "/semantic_bev_editor/paint_point")
        self.mode_topic = rospy.get_param("~mode_topic", "/semantic_bev_editor/mode")
        self.clear_topic = rospy.get_param("~clear_topic", "/semantic_bev_editor/clear")
        self.undo_topic = rospy.get_param("~undo_topic", "/semantic_bev_editor/undo")
        self.save_topic = rospy.get_param("~save_topic", "/semantic_bev_editor/save")
        self.load_topic = rospy.get_param("~load_topic", "/semantic_bev_editor/load")

        self.max_display_px = max(400, int(rospy.get_param("~max_display_px", 1200)))
        self.max_scale = max(1.0, float(rospy.get_param("~max_scale", 6.0)))
        self.drag_min_step_px = max(1, int(rospy.get_param("~drag_min_step_px", 3)))
        self.paint_throttle_hz = max(1.0, float(rospy.get_param("~paint_throttle_hz", 30.0)))

        self.mode = "sidewalk"
        self._lock = threading.RLock()
        self._grid_msg = None
        self._display_scale = 1.0
        self._dragging = False
        self._last_drag_px = None
        self._last_pub_time = rospy.Time(0)

        self.pub_paint = rospy.Publisher(self.paint_point_topic, PointStamped, queue_size=50)
        self.pub_mode = rospy.Publisher(self.mode_topic, String, queue_size=10, latch=True)
        self.pub_clear = rospy.Publisher(self.clear_topic, Empty, queue_size=2)
        self.pub_undo = rospy.Publisher(self.undo_topic, Empty, queue_size=2)
        self.pub_save = rospy.Publisher(self.save_topic, Empty, queue_size=2)
        self.pub_load = rospy.Publisher(self.load_topic, Empty, queue_size=2)

        self.sub_grid = rospy.Subscriber(self.grid_topic, OccupancyGrid, self.grid_callback, queue_size=1)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        rospy.on_shutdown(self.on_shutdown)
        self._publish_mode()
        rospy.loginfo(
            "semantic_bev_2d_editor started | grid=%s | keys=[1 sidewalk, 2 road, 3 curb, 0 observed, u/s/l/c/q]",
            self.grid_topic,
        )

    def on_shutdown(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def _publish_mode(self):
        self.pub_mode.publish(String(data=self.mode))

    def grid_callback(self, msg):
        with self._lock:
            self._grid_msg = msg

    @staticmethod
    def _empty_canvas():
        img = np.zeros((480, 900, 3), dtype=np.uint8)
        cv2.putText(
            img,
            "Waiting for /semantic_bev_editor/grid ...",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (220, 220, 220),
            2,
        )
        return img

    def _draw_hud(self, disp):
        cv2.putText(
            disp,
            "Mode: {} | [1]sidewalk [2]road [3]curb [0]observed | [u]undo [s]save [l]load [c]reset [q]quit".format(self.mode),
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

    def _build_display_image(self):
        with self._lock:
            grid = self._grid_msg
        if grid is None or grid.info.width <= 0 or grid.info.height <= 0:
            return self._empty_canvas(), None

        w = int(grid.info.width)
        h = int(grid.info.height)
        data = np.array(grid.data, dtype=np.int16).reshape((h, w))
        data_img = np.flipud(data)

        img = np.zeros((h, w, 3), dtype=np.uint8)
        for val, color in LABEL_COLORS.items():
            img[data_img == val] = color

        if w >= h:
            scale = min(self.max_scale, float(self.max_display_px) / float(max(1, w)))
        else:
            scale = min(self.max_scale, float(self.max_display_px) / float(max(1, h)))
        scale = max(1.0, scale)
        self._display_scale = scale

        disp = cv2.resize(
            img,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_NEAREST,
        )
        self._draw_hud(disp)
        return disp, grid

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
        return x, y, grid.header.frame_id or "map"

    def _publish_point(self, x, y, frame_id):
        now = rospy.Time.now()
        if (now - self._last_pub_time).to_sec() < (1.0 / self.paint_throttle_hz):
            return
        self._last_pub_time = now
        msg = PointStamped()
        msg.header.stamp = now
        msg.header.frame_id = frame_id
        msg.point.x = float(x)
        msg.point.y = float(y)
        msg.point.z = 0.0
        self.pub_paint.publish(msg)

    def _handle_point_action(self, px, py, force_mode=None):
        mapped = self._pixel_to_world(px, py)
        if mapped is None:
            return
        x, y, frame_id = mapped
        if force_mode is not None and force_mode != self.mode:
            self.mode = force_mode
            self._publish_mode()
        self._publish_point(x, y, frame_id)

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
            self._handle_point_action(x, y, "observed")
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
        if key == ord("1"):
            self.mode = "sidewalk"
            self._publish_mode()
        elif key == ord("2"):
            self.mode = "road"
            self._publish_mode()
        elif key == ord("3"):
            self.mode = "curb"
            self._publish_mode()
        elif key == ord("0"):
            self.mode = "observed"
            self._publish_mode()
        elif key == ord("u"):
            self.pub_undo.publish(Empty())
        elif key == ord("s"):
            self.pub_save.publish(Empty())
        elif key == ord("l"):
            self.pub_load.publish(Empty())
        elif key == ord("c"):
            self.pub_clear.publish(Empty())
        elif key == ord("q") or key == 27:
            rospy.signal_shutdown("semantic 2d editor closed by user")

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
    rospy.init_node("semantic_bev_2d_editor")
    try:
        node = SemanticBEV2DEditor()
        node.spin()
    except rospy.ROSInterruptException:
        pass
