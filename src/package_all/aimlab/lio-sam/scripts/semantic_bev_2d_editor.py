#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import struct
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
BACKGROUND_POINT_COLOR = (142, 146, 152)


class SemanticBEV2DEditor:
    def __init__(self):
        self.window_name = rospy.get_param("~window_name", "Semantic BEV 2D Editor")
        self.grid_topic = rospy.get_param("~grid_topic", "/semantic_bev_editor/grid")
        self.paint_point_topic = rospy.get_param("~paint_point_topic", "/semantic_bev_editor/paint_point")
        self.erase_point_topic = rospy.get_param("~erase_point_topic", "/semantic_bev_editor/erase_point")
        self.mode_topic = rospy.get_param("~mode_topic", "/semantic_bev_editor/mode")
        self.clear_topic = rospy.get_param("~clear_topic", "/semantic_bev_editor/clear")
        self.undo_topic = rospy.get_param("~undo_topic", "/semantic_bev_editor/undo")
        self.save_topic = rospy.get_param("~save_topic", "/semantic_bev_editor/save")
        self.load_topic = rospy.get_param("~load_topic", "/semantic_bev_editor/load")
        self.sidewalk_only_editing = bool(rospy.get_param("~sidewalk_only_editing", True))
        self.background_enabled = bool(rospy.get_param("~background_enabled", True))
        self.background_alpha = min(0.95, max(0.0, float(rospy.get_param("~background_alpha", 0.35))))
        self.background_point_color = tuple(
            int(v) for v in rospy.get_param("~background_point_color_bgr", list(BACKGROUND_POINT_COLOR))
        )
        self.background_pcd_path = self._resolve_background_pcd_path(
            rospy.get_param("~background_pcd_path", "")
        )

        self.max_display_px = max(400, int(rospy.get_param("~max_display_px", 1200)))
        self.max_scale = max(1.0, float(rospy.get_param("~max_scale", 24.0)))
        self.zoom_step = max(1.01, float(rospy.get_param("~zoom_step", 1.25)))
        self.drag_min_step_px = max(1, int(rospy.get_param("~drag_min_step_px", 3)))
        self.paint_throttle_hz = max(1.0, float(rospy.get_param("~paint_throttle_hz", 30.0)))
        self.zoom_slider_max = max(100, int(round(self.max_scale * 100.0)))

        self.mode = "sidewalk"
        self._lock = threading.RLock()
        self._grid_msg = None
        self._fit_scale = 1.0
        self._display_scale = 1.0
        self._view_origin_px = (0, 0)
        self._dragging = False
        self._last_drag_px = None
        self._panning = False
        self._last_pan_px = None
        self._zoom_dragging = False
        self._last_zoom_px = None
        self._erasing = False
        self._last_erase_px = None
        self._last_pub_time = rospy.Time(0)
        self._trackbar_updating = False
        self._background_points_xy = None
        self._background_mask = None
        self._background_signature = None
        self.show_background = self.background_enabled

        self.pub_paint = rospy.Publisher(self.paint_point_topic, PointStamped, queue_size=50)
        self.pub_erase = rospy.Publisher(self.erase_point_topic, PointStamped, queue_size=50)
        self.pub_mode = rospy.Publisher(self.mode_topic, String, queue_size=10, latch=True)
        self.pub_clear = rospy.Publisher(self.clear_topic, Empty, queue_size=2)
        self.pub_undo = rospy.Publisher(self.undo_topic, Empty, queue_size=2)
        self.pub_save = rospy.Publisher(self.save_topic, Empty, queue_size=2)
        self.pub_load = rospy.Publisher(self.load_topic, Empty, queue_size=2)

        self.sub_grid = rospy.Subscriber(self.grid_topic, OccupancyGrid, self.grid_callback, queue_size=1)

        window_flags = cv2.WINDOW_NORMAL
        if hasattr(cv2, "WINDOW_GUI_EXPANDED"):
            window_flags |= cv2.WINDOW_GUI_EXPANDED
        cv2.namedWindow(self.window_name, window_flags)
        cv2.createTrackbar("Zoom %", self.window_name, 100, self.zoom_slider_max, self.on_trackbar_zoom)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        rospy.on_shutdown(self.on_shutdown)
        self._load_background_points()
        self._publish_mode()
        rospy.loginfo(
            "semantic_bev_2d_editor started | grid=%s | background=%s | sidewalk_only=%s | keys=[g background, wheel zoom, ctrl+drag zoom, middle drag pan, +/- zoom, f fit, u/s/l/c/q]",
            self.grid_topic,
            self.background_pcd_path if self.background_pcd_path else "off",
            str(self.sidewalk_only_editing),
        )

    @staticmethod
    def _pcd_type_to_struct(type_code, size):
        table = {
            ("F", 4): "f",
            ("F", 8): "d",
            ("I", 1): "b",
            ("I", 2): "h",
            ("I", 4): "i",
            ("I", 8): "q",
            ("U", 1): "B",
            ("U", 2): "H",
            ("U", 4): "I",
            ("U", 8): "Q",
        }
        return table.get((type_code, size))

    @staticmethod
    def _candidate_background_paths():
        return [
            "/home/byeongjae/code/Modular_Approach_Autonomous_Driving-/src/package_all/aimlab/lio-localizer/map/test/GlobalMap2D.pcd",
            "/home/byeongjae/code/Modular_Approach_Autonomous_Driving-/src/package_all/monitoring_delivery/latest/GlobalMap2D.pcd",
        ]

    def _resolve_background_pcd_path(self, configured):
        configured = os.path.expanduser(str(configured).strip())
        if configured:
            return configured
        for candidate in self._candidate_background_paths():
            if os.path.isfile(candidate):
                return candidate
        return ""

    @classmethod
    def _read_pcd_header(cls, path):
        header = {}
        with open(path, "rb") as f:
            while True:
                line = f.readline()
                if not line:
                    raise ValueError("pcd header ended unexpectedly: %s" % path)
                text = line.decode("ascii", errors="ignore").strip()
                if not text or text.startswith("#"):
                    continue
                parts = text.split()
                key = parts[0].upper()
                values = parts[1:]
                header[key] = values
                if key == "DATA":
                    header["DATA_START"] = f.tell()
                    break

        fields = header.get("FIELDS", [])
        size = [int(v) for v in header.get("SIZE", [])]
        types = header.get("TYPE", [])
        counts = [int(v) for v in header.get("COUNT", ["1"] * len(fields))]
        points = int(header.get("POINTS", [header.get("WIDTH", ["0"])[0]])[0])
        data_mode = header.get("DATA", [""])[0].lower()

        if not (len(fields) == len(size) == len(types) == len(counts)):
            raise ValueError("pcd header field metadata mismatch: %s" % path)

        offset = 0
        layout = []
        for name, item_size, item_type, item_count in zip(fields, size, types, counts):
            struct_code = cls._pcd_type_to_struct(item_type, item_size)
            if struct_code is None:
                raise ValueError("unsupported pcd field type %s/%s in %s" % (item_type, item_size, path))
            layout.append(
                {
                    "name": name,
                    "size": item_size,
                    "type": item_type,
                    "count": item_count,
                    "offset": offset,
                    "struct_code": struct_code,
                }
            )
            offset += item_size * item_count

        header["POINT_COUNT"] = points
        header["DATA_MODE"] = data_mode
        header["POINT_STEP"] = offset
        header["LAYOUT"] = layout
        return header

    @classmethod
    def _read_pcd_xy_points(cls, path):
        header = cls._read_pcd_header(path)
        fields = {item["name"]: item for item in header["LAYOUT"]}
        if "x" not in fields or "y" not in fields:
            raise ValueError("pcd missing x/y fields: %s" % path)

        def unpack_scalar(blob, spec):
            fmt = "<" + spec["struct_code"]
            return struct.unpack_from(fmt, blob, spec["offset"])[0]

        data_mode = header["DATA_MODE"]
        field_names = header.get("FIELDS", [])
        point_count = header["POINT_COUNT"]
        point_step = header["POINT_STEP"]
        points = []

        if data_mode == "ascii":
            with open(path, "r", encoding="ascii", errors="ignore") as f:
                started = False
                field_index = {name: idx for idx, name in enumerate(field_names)}
                for line in f:
                    if not started:
                        if line.strip().lower().startswith("data"):
                            started = True
                        continue
                    row = line.strip().split()
                    if not row:
                        continue
                    points.append((float(row[field_index["x"]]), float(row[field_index["y"]])))
            return np.asarray(points, dtype=np.float32)

        if data_mode != "binary":
            raise ValueError("unsupported pcd data mode '%s' in %s" % (data_mode, path))

        with open(path, "rb") as f:
            f.seek(header["DATA_START"])
            for _ in range(point_count):
                blob = f.read(point_step)
                if len(blob) < point_step:
                    break
                points.append(
                    (
                        float(unpack_scalar(blob, fields["x"])),
                        float(unpack_scalar(blob, fields["y"])),
                    )
                )
        return np.asarray(points, dtype=np.float32)

    def _load_background_points(self):
        self._background_points_xy = None
        self._background_mask = None
        self._background_signature = None
        if not self.background_enabled:
            return
        if not self.background_pcd_path or not os.path.isfile(self.background_pcd_path):
            rospy.logwarn("semantic_bev_2d_editor: background pcd not found: %s", self.background_pcd_path)
            return
        try:
            points_xy = self._read_pcd_xy_points(self.background_pcd_path)
        except Exception as e:
            rospy.logwarn("semantic_bev_2d_editor: failed to load background pcd '%s': %s", self.background_pcd_path, str(e))
            return
        self._background_points_xy = points_xy
        rospy.loginfo(
            "semantic_bev_2d_editor: loaded background GlobalMap2D (%d pts) from %s",
            int(points_xy.shape[0]),
            self.background_pcd_path,
        )

    def on_shutdown(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def _publish_mode(self):
        self.pub_mode.publish(String(data=self.mode))

    def on_trackbar_zoom(self, value):
        if self._trackbar_updating:
            return
        scale = max(self._fit_scale, min(self.max_scale, float(max(100, value)) / 100.0))
        self._set_scale_around_pixel(scale)

    def grid_callback(self, msg):
        reset_needed = False
        with self._lock:
            prev = self._grid_msg
            self._grid_msg = msg
            if prev is None:
                reset_needed = True
            else:
                if (
                    int(prev.info.width) != int(msg.info.width)
                    or int(prev.info.height) != int(msg.info.height)
                    or abs(float(prev.info.resolution) - float(msg.info.resolution)) > 1e-9
                ):
                    reset_needed = True
        if reset_needed:
            self._reset_view()

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
        if self.sidewalk_only_editing:
            help_text = "Sidewalk edit | BG:{} | left drag add sidewalk | right drag erase | [g]bg [u]undo [s]save [l]load [c]reset [f]fit [q]quit".format(
                "on" if self.show_background and self._background_points_xy is not None else "off"
            )
        else:
            help_text = "Mode: {} | BG:{} | slider/wheel zoom | middle drag pan | [1/2/3/0] label | [g]bg [u]undo [s]save [l]load [c]reset [f]fit [q]quit".format(
                self.mode,
                "on" if self.show_background and self._background_points_xy is not None else "off",
            )
        cv2.putText(
            disp,
            help_text,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

    def _ensure_background_mask(self, grid):
        if not self.show_background or self._background_points_xy is None or self._background_points_xy.size == 0:
            return None
        signature = (
            int(grid.info.width),
            int(grid.info.height),
            round(float(grid.info.resolution), 6),
            round(float(grid.info.origin.position.x), 6),
            round(float(grid.info.origin.position.y), 6),
        )
        if self._background_signature == signature and self._background_mask is not None:
            return self._background_mask

        width = int(grid.info.width)
        height = int(grid.info.height)
        resolution = float(grid.info.resolution)
        origin_x = float(grid.info.origin.position.x)
        origin_y = float(grid.info.origin.position.y)

        pts = self._background_points_xy
        gx = np.floor((pts[:, 0] - origin_x) / resolution).astype(np.int32)
        gy = np.floor((pts[:, 1] - origin_y) / resolution).astype(np.int32)
        valid = (gx >= 0) & (gy >= 0) & (gx < width) & (gy < height)

        mask = np.zeros((height, width), dtype=np.uint8)
        if np.any(valid):
            mask[gy[valid], gx[valid]] = 255
        mask = np.flipud(mask)
        self._background_mask = mask
        self._background_signature = signature
        return self._background_mask

    def _blend_background(self, img, grid):
        bg_mask = self._ensure_background_mask(grid)
        if bg_mask is None:
            return img
        blended = img.astype(np.float32)
        point_color = np.array(self.background_point_color, dtype=np.float32)
        mask = bg_mask > 0
        if np.any(mask):
            blended[mask] = (
                blended[mask] * (1.0 - self.background_alpha)
                + point_color * self.background_alpha
            )
        return np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def _mouse_wheel_delta(flags):
        if hasattr(cv2, "getMouseWheelDelta"):
            try:
                return int(cv2.getMouseWheelDelta(flags))
            except Exception:
                pass
        delta = (int(flags) >> 16) & 0xFFFF
        if delta >= 0x8000:
            delta -= 0x10000
        return int(delta)

    def _clamp_view_origin(self, origin_x, origin_y, scaled_w, scaled_h):
        viewport_w = min(self.max_display_px, scaled_w)
        viewport_h = min(self.max_display_px, scaled_h)
        max_x = max(0, scaled_w - viewport_w)
        max_y = max(0, scaled_h - viewport_h)
        origin_x = min(max(0, int(round(origin_x))), max_x)
        origin_y = min(max(0, int(round(origin_y))), max_y)
        return origin_x, origin_y

    def _set_scale_around_pixel(self, new_scale, anchor_x=None, anchor_y=None):
        with self._lock:
            grid = self._grid_msg
            old_scale = self._display_scale
            origin_x, origin_y = self._view_origin_px
        if grid is None:
            return
        w = int(grid.info.width)
        h = int(grid.info.height)
        old_scale = max(1.0, old_scale)
        new_scale = max(self._fit_scale, min(self.max_scale, new_scale))
        scaled_w = max(1, int(round(w * new_scale)))
        scaled_h = max(1, int(round(h * new_scale)))
        if anchor_x is None:
            anchor_x = 0.5 * min(self.max_display_px, max(1, int(round(w * old_scale))))
        if anchor_y is None:
            anchor_y = 0.5 * min(self.max_display_px, max(1, int(round(h * old_scale))))
        map_x = (origin_x + anchor_x) / old_scale
        map_y = (origin_y + anchor_y) / old_scale
        new_origin_x = map_x * new_scale - anchor_x
        new_origin_y = map_y * new_scale - anchor_y
        new_origin_x, new_origin_y = self._clamp_view_origin(new_origin_x, new_origin_y, scaled_w, scaled_h)
        with self._lock:
            self._display_scale = new_scale
            self._view_origin_px = (new_origin_x, new_origin_y)
        self._sync_trackbar_to_scale(new_scale)

    def _reset_view(self):
        with self._lock:
            grid = self._grid_msg
        if grid is None:
            return
        w = int(grid.info.width)
        h = int(grid.info.height)
        if w <= 0 or h <= 0:
            return
        if w >= h:
            fit = min(self.max_scale, float(self.max_display_px) / float(max(1, w)))
        else:
            fit = min(self.max_scale, float(self.max_display_px) / float(max(1, h)))
        fit = max(1.0, fit)
        scaled_w = max(1, int(round(w * fit)))
        scaled_h = max(1, int(round(h * fit)))
        origin_x, origin_y = self._clamp_view_origin(0, 0, scaled_w, scaled_h)
        with self._lock:
            self._fit_scale = fit
            self._display_scale = fit
            self._view_origin_px = (origin_x, origin_y)
        self._sync_trackbar_to_scale(fit)

    def _sync_trackbar_to_scale(self, scale):
        value = int(round(max(1.0, min(self.max_scale, scale)) * 100.0))
        value = min(self.zoom_slider_max, max(100, value))
        self._trackbar_updating = True
        try:
            cv2.setTrackbarPos("Zoom %", self.window_name, value)
        except Exception:
            pass
        finally:
            self._trackbar_updating = False

    def _build_display_image(self):
        with self._lock:
            grid = self._grid_msg
        if grid is None or grid.info.width <= 0 or grid.info.height <= 0:
            return self._empty_canvas(), None

        w = int(grid.info.width)
        h = int(grid.info.height)
        data = np.array(grid.data, dtype=np.int16).reshape((h, w))
        data_img = np.flipud(data)

        if self.sidewalk_only_editing:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img[:, :] = LABEL_COLORS[-1]
            img[data_img == 80] = LABEL_COLORS[80]
        else:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            for val, color in LABEL_COLORS.items():
                img[data_img == val] = color
        img = self._blend_background(img, grid)

        with self._lock:
            if self._display_scale <= 0.0:
                self._reset_view()
            current_scale = self._display_scale
            if self._fit_scale <= 0.0:
                self._fit_scale = current_scale
            origin_x, origin_y = self._view_origin_px

        disp_full = cv2.resize(
            img,
            (max(1, int(round(w * current_scale))), max(1, int(round(h * current_scale)))),
            interpolation=cv2.INTER_NEAREST,
        )
        scaled_h, scaled_w = disp_full.shape[:2]
        origin_x, origin_y = self._clamp_view_origin(origin_x, origin_y, scaled_w, scaled_h)
        viewport_w = min(self.max_display_px, scaled_w)
        viewport_h = min(self.max_display_px, scaled_h)
        disp = disp_full[origin_y:origin_y + viewport_h, origin_x:origin_x + viewport_w].copy()
        with self._lock:
            self._view_origin_px = (origin_x, origin_y)
            self._display_scale = current_scale
        self._draw_hud(disp)
        return disp, grid

    def _pixel_to_world(self, px, py):
        with self._lock:
            grid = self._grid_msg
            scale = self._display_scale
            origin_x, origin_y = self._view_origin_px
        if grid is None or scale <= 0.0:
            return None
        w = int(grid.info.width)
        h = int(grid.info.height)
        gx = int((px + origin_x) / scale)
        gy_img = int((py + origin_y) / scale)
        gy = (h - 1) - gy_img
        if gx < 0 or gy < 0 or gx >= w or gy >= h:
            return None
        x = grid.info.origin.position.x + (gx + 0.5) * grid.info.resolution
        y = grid.info.origin.position.y + (gy + 0.5) * grid.info.resolution
        return x, y, grid.header.frame_id or "map"

    def _publish_point(self, x, y, frame_id, erase=False):
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
        if erase:
            self.pub_erase.publish(msg)
        else:
            self.pub_paint.publish(msg)

    def _handle_point_action(self, px, py, force_mode=None, erase=False):
        mapped = self._pixel_to_world(px, py)
        if mapped is None:
            return
        x, y, frame_id = mapped
        if self.sidewalk_only_editing:
            self._publish_point(x, y, frame_id, erase=erase)
            return
        if force_mode is not None and force_mode != self.mode:
            self.mode = force_mode
            self._publish_mode()
        self._publish_point(x, y, frame_id, erase=erase)

    def on_mouse(self, event, x, y, flags, _userdata):
        if hasattr(cv2, "EVENT_MOUSEWHEEL") and event == cv2.EVENT_MOUSEWHEEL:
            delta = self._mouse_wheel_delta(flags)
            if delta > 0:
                self._set_scale_around_pixel(self._display_scale * self.zoom_step, x, y)
            elif delta < 0:
                self._set_scale_around_pixel(self._display_scale / self.zoom_step, x, y)
            return
        if event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_CTRLKEY):
            self._zoom_dragging = True
            self._last_zoom_px = (x, y)
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self._dragging = True
            self._last_drag_px = (x, y)
            self._handle_point_action(x, y, None, erase=False)
            return
        if event == cv2.EVENT_LBUTTONUP:
            self._zoom_dragging = False
            self._last_zoom_px = None
            self._dragging = False
            self._last_drag_px = None
            return
        if event == cv2.EVENT_RBUTTONDOWN:
            self._erasing = True
            self._last_erase_px = (x, y)
            self._handle_point_action(x, y, "observed", erase=True)
            return
        if event == cv2.EVENT_RBUTTONUP:
            self._erasing = False
            self._last_erase_px = None
            return
        if event == cv2.EVENT_MBUTTONDOWN:
            self._panning = True
            self._last_pan_px = (x, y)
            return
        if event == cv2.EVENT_MBUTTONUP:
            self._panning = False
            self._last_pan_px = None
            return
        if event == cv2.EVENT_MOUSEMOVE and self._zoom_dragging:
            if self._last_zoom_px is None:
                self._last_zoom_px = (x, y)
                return
            dy = y - self._last_zoom_px[1]
            self._last_zoom_px = (x, y)
            if abs(dy) < self.drag_min_step_px:
                return
            steps = max(1, int(abs(dy) / float(self.drag_min_step_px)))
            factor = self.zoom_step ** steps
            if dy < 0:
                self._set_scale_around_pixel(self._display_scale * factor, x, y)
            else:
                self._set_scale_around_pixel(self._display_scale / factor, x, y)
            return
        if event == cv2.EVENT_MOUSEMOVE and self._panning:
            if self._last_pan_px is None:
                self._last_pan_px = (x, y)
                return
            dx = x - self._last_pan_px[0]
            dy = y - self._last_pan_px[1]
            self._last_pan_px = (x, y)
            with self._lock:
                grid = self._grid_msg
                origin_x, origin_y = self._view_origin_px
                scale = self._display_scale
            if grid is None:
                return
            scaled_w = max(1, int(round(int(grid.info.width) * scale)))
            scaled_h = max(1, int(round(int(grid.info.height) * scale)))
            origin_x, origin_y = self._clamp_view_origin(origin_x - dx, origin_y - dy, scaled_w, scaled_h)
            with self._lock:
                self._view_origin_px = (origin_x, origin_y)
            return
        if event == cv2.EVENT_MOUSEMOVE and self._erasing:
            if self._last_erase_px is None:
                self._last_erase_px = (x, y)
                self._handle_point_action(x, y, "observed", erase=True)
                return
            dx = x - self._last_erase_px[0]
            dy = y - self._last_erase_px[1]
            if (dx * dx + dy * dy) >= (self.drag_min_step_px * self.drag_min_step_px):
                self._last_erase_px = (x, y)
                self._handle_point_action(x, y, "observed", erase=True)
            return
        if event == cv2.EVENT_MOUSEMOVE and self._dragging:
            if self._last_drag_px is None:
                self._last_drag_px = (x, y)
                self._handle_point_action(x, y, None, erase=False)
                return
            dx = x - self._last_drag_px[0]
            dy = y - self._last_drag_px[1]
            if (dx * dx + dy * dy) >= (self.drag_min_step_px * self.drag_min_step_px):
                self._last_drag_px = (x, y)
                self._handle_point_action(x, y, None, erase=False)

    def _handle_key(self, key):
        if (not self.sidewalk_only_editing) and key == ord("1"):
            self.mode = "sidewalk"
            self._publish_mode()
        elif (not self.sidewalk_only_editing) and key == ord("2"):
            self.mode = "road"
            self._publish_mode()
        elif (not self.sidewalk_only_editing) and key == ord("3"):
            self.mode = "curb"
            self._publish_mode()
        elif (not self.sidewalk_only_editing) and key == ord("0"):
            self.mode = "observed"
            self._publish_mode()
        elif key == ord("u"):
            self.pub_undo.publish(Empty())
        elif key == ord("+") or key == ord("="):
            self._set_scale_around_pixel(self._display_scale * self.zoom_step)
        elif key == ord("-") or key == ord("_"):
            self._set_scale_around_pixel(self._display_scale / self.zoom_step)
        elif key == ord("f"):
            self._reset_view()
        elif key == ord("g"):
            self.show_background = not self.show_background
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
