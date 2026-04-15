#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
import rosnode
import struct
import threading

import rospy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Empty, Header, String


LABEL_OBSERVED = "observed"
LABEL_SIDEWALK = "sidewalk"
LABEL_ROAD = "road"
LABEL_CURB = "curb"
EDITABLE_LABELS = (LABEL_SIDEWALK, LABEL_ROAD, LABEL_CURB)
ALL_LABELS = (LABEL_OBSERVED,) + EDITABLE_LABELS
GRID_VALUES = {
    LABEL_OBSERVED: 0,
    LABEL_ROAD: 40,
    LABEL_SIDEWALK: 80,
    LABEL_CURB: 100,
}
RGB_VALUES = {
    LABEL_OBSERVED: (70, 72, 78),
    LABEL_SIDEWALK: (210, 230, 210),
    LABEL_ROAD: (70, 120, 200),
    LABEL_CURB: (220, 70, 60),
}


def rgb_to_float(rgb):
    packed = (int(rgb[0]) << 16) | (int(rgb[1]) << 8) | int(rgb[2])
    return struct.unpack("f", struct.pack("I", packed))[0]


class SemanticBEVEditor:
    def __init__(self):
        default_state = (
            "/home/byeongjae/code/Modular_Approach_Autonomous_Driving-"
            "/generated/raw_semantic_bev_map_fullbag/raw_semantic_bev_state.json"
        )
        default_edited = (
            "/home/byeongjae/code/Modular_Approach_Autonomous_Driving-"
            "/generated/raw_semantic_bev_map_fullbag/raw_semantic_bev_state.edited.json"
        )
        default_override = (
            "/home/byeongjae/code/Modular_Approach_Autonomous_Driving-"
            "/generated/raw_semantic_bev_map_fullbag/raw_semantic_bev_override.edited.json"
        )
        default_drivable_state = os.path.expanduser("~/.ros/lio_sam_drivable_area_state.json")

        self.frame_id = rospy.get_param("~frame_id", "map")
        self.state_file_path = os.path.expanduser(rospy.get_param("~state_file_path", default_state))
        self.edited_state_file_path = os.path.expanduser(
            rospy.get_param("~edited_state_file_path", default_edited)
        )
        self.override_file_path = os.path.expanduser(
            rospy.get_param("~override_file_path", default_override)
        )
        self.grid_topic = rospy.get_param("~grid_topic", "/semantic_bev_editor/grid")
        self.cloud_topic = rospy.get_param("~cloud_topic", "/semantic_bev_editor/cloud")
        self.paint_point_topic = rospy.get_param("~paint_point_topic", "/semantic_bev_editor/paint_point")
        self.erase_point_topic = rospy.get_param("~erase_point_topic", "/semantic_bev_editor/erase_point")
        self.export_drivable_state_on_save = bool(rospy.get_param("~export_drivable_state_on_save", True))
        self.drivable_state_file_path = os.path.expanduser(
            rospy.get_param("~drivable_state_file_path", default_drivable_state)
        )
        self.drivable_load_topic = rospy.get_param("~drivable_load_topic", "/lio_sam/drivable_area/load")
        self.notify_drivable_reload_on_save = bool(rospy.get_param("~notify_drivable_reload_on_save", True))
        self.mode_topic = rospy.get_param("~mode_topic", "/semantic_bev_editor/mode")
        self.clear_topic = rospy.get_param("~clear_topic", "/semantic_bev_editor/clear")
        self.undo_topic = rospy.get_param("~undo_topic", "/semantic_bev_editor/undo")
        self.save_topic = rospy.get_param("~save_topic", "/semantic_bev_editor/save")
        self.load_topic = rospy.get_param("~load_topic", "/semantic_bev_editor/load")
        self.clicked_point_topic = rospy.get_param("~clicked_point_topic", "/semantic_bev_editor/clicked_point")
        self.use_clicked_point = bool(rospy.get_param("~use_clicked_point", True))
        self.sidewalk_only_editing = bool(rospy.get_param("~sidewalk_only_editing", True))
        self.clicked_point_mode = str(
            rospy.get_param("~clicked_point_mode", "toggle" if self.sidewalk_only_editing else "add")
        ).strip().lower()
        if self.clicked_point_mode not in ("add", "erase", "toggle"):
            rospy.logwarn("semantic_bev_editor unknown clicked_point_mode='%s', fallback to 'toggle'", self.clicked_point_mode)
            self.clicked_point_mode = "toggle"
        self.publish_period_s = max(0.1, float(rospy.get_param("~publish_period_s", 0.5)))
        self.brush_radius_m = max(0.05, float(rospy.get_param("~brush_radius_m", 0.8)))
        self.max_history = max(1, int(rospy.get_param("~max_history", 30)))
        self.auto_load_edited_state = bool(rospy.get_param("~auto_load_edited_state", True))
        self.save_on_shutdown = bool(rospy.get_param("~save_on_shutdown", True))
        self.watch_node_name = str(rospy.get_param("~watch_node_name", "")).strip()
        self.save_on_watch_node_exit = bool(rospy.get_param("~save_on_watch_node_exit", False))
        self.watch_node_check_period_s = max(0.5, float(rospy.get_param("~watch_node_check_period_s", 1.0)))

        self._lock = threading.RLock()
        self._history = []
        self._dirty = True
        self._base_payload = None
        self._base_label_map = {}
        self._observed = {}
        self._classes = {label: set() for label in EDITABLE_LABELS}
        self._ix_min = 0
        self._ix_max = -1
        self._iy_min = 0
        self._iy_max = -1
        self.grid_resolution_m = 0.2
        self.mode = LABEL_SIDEWALK
        self._watch_node_seen = False
        self._watch_node_saved = False

        self._load_base_state()
        if self.auto_load_edited_state and os.path.isfile(self.edited_state_file_path):
            self._load_state_from_path(self.edited_state_file_path, keep_as_base=False)

        self.pub_grid = rospy.Publisher(self.grid_topic, OccupancyGrid, queue_size=1, latch=True)
        self.pub_cloud = rospy.Publisher(self.cloud_topic, PointCloud2, queue_size=1, latch=True)
        self.pub_drivable_reload = rospy.Publisher(self.drivable_load_topic, Empty, queue_size=1)

        self.sub_paint = rospy.Subscriber(self.paint_point_topic, PointStamped, self.paint_point_callback, queue_size=50)
        self.sub_erase = rospy.Subscriber(self.erase_point_topic, PointStamped, self.erase_point_callback, queue_size=50)
        self.sub_mode = rospy.Subscriber(self.mode_topic, String, self.mode_callback, queue_size=10)
        self.sub_clear = rospy.Subscriber(self.clear_topic, Empty, self.clear_callback, queue_size=2)
        self.sub_undo = rospy.Subscriber(self.undo_topic, Empty, self.undo_callback, queue_size=2)
        self.sub_save = rospy.Subscriber(self.save_topic, Empty, self.save_callback, queue_size=2)
        self.sub_load = rospy.Subscriber(self.load_topic, Empty, self.load_callback, queue_size=2)
        self.sub_click = None
        if self.use_clicked_point:
            self.sub_click = rospy.Subscriber(
                self.clicked_point_topic, PointStamped, self.clicked_point_callback, queue_size=50
            )

        rospy.Timer(rospy.Duration(self.publish_period_s), self.on_timer)
        if self.save_on_watch_node_exit and self.watch_node_name:
            rospy.Timer(rospy.Duration(self.watch_node_check_period_s), self._watch_node_timer)
        rospy.on_shutdown(self.on_shutdown)
        rospy.loginfo(
            "semantic_bev_editor started | state=%s, edited=%s, override=%s, mode=%s",
            self.state_file_path,
            self.edited_state_file_path,
            self.override_file_path,
            self.mode,
        )

    def _load_base_state(self):
        if not os.path.isfile(self.state_file_path):
            rospy.logwarn("semantic_bev_editor base state missing: %s", self.state_file_path)
            with self._lock:
                self.grid_resolution_m = 0.2
                self._observed = {}
                self._classes = {label: set() for label in EDITABLE_LABELS}
                self._ix_min = 0
                self._ix_max = -1
                self._iy_min = 0
                self._iy_max = -1
                self._dirty = True
                self._base_payload = {"meta": {"grid_resolution": 0.2}, "classes": {}, "observed_cells": []}
                self._base_label_map = {}
                self._history = []
            return
        self._load_state_from_path(self.state_file_path, keep_as_base=True)

    def _load_state_from_path(self, path, keep_as_base):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        meta = payload.get("meta", {})
        grid_resolution = float(meta.get("grid_resolution", 0.2))
        observed_cells = payload.get("observed_cells", [])

        observed = {}
        base_label_map = {}
        ix_min = None
        ix_max = None
        iy_min = None
        iy_max = None
        for row in observed_cells:
            if len(row) < 4:
                continue
            ix = int(row[0])
            iy = int(row[1])
            z = float(row[2])
            label = str(row[3]).strip().lower()
            if label not in ALL_LABELS:
                label = LABEL_OBSERVED
            key = (ix, iy)
            observed[key] = z
            base_label_map[key] = label
            ix_min = ix if ix_min is None else min(ix_min, ix)
            ix_max = ix if ix_max is None else max(ix_max, ix)
            iy_min = iy if iy_min is None else min(iy_min, iy)
            iy_max = iy if iy_max is None else max(iy_max, iy)

        classes = {label: set() for label in EDITABLE_LABELS}
        for label in EDITABLE_LABELS:
            for row in payload.get("classes", {}).get(label, []):
                if len(row) < 2:
                    continue
                key = (int(row[0]), int(row[1]))
                if key in observed:
                    classes[label].add(key)

        with self._lock:
            self.grid_resolution_m = grid_resolution
            self._observed = observed
            self._classes = classes
            self._ix_min = ix_min if ix_min is not None else 0
            self._ix_max = ix_max if ix_max is not None else -1
            self._iy_min = iy_min if iy_min is not None else 0
            self._iy_max = iy_max if iy_max is not None else -1
            self._dirty = True
            if keep_as_base:
                self._base_payload = payload
                self._base_label_map = dict(base_label_map)
            self._history = []

    def _label_of_key_locked(self, key):
        for label in EDITABLE_LABELS:
            if key in self._classes[label]:
                return label
        return LABEL_OBSERVED

    def _set_key_label_locked(self, key, label):
        for editable in EDITABLE_LABELS:
            self._classes[editable].discard(key)
        if label in EDITABLE_LABELS:
            self._classes[label].add(key)

    def _key_to_center(self, ix, iy):
        r = self.grid_resolution_m
        return (ix + 0.5) * r, (iy + 0.5) * r

    def _world_to_candidate_keys(self, x, y):
        r = self.grid_resolution_m
        radius = self.brush_radius_m
        ix_min = int(math.floor((x - radius) / r))
        ix_max = int(math.floor((x + radius) / r))
        iy_min = int(math.floor((y - radius) / r))
        iy_max = int(math.floor((y + radius) / r))
        out = []
        rr = radius * radius
        for ix in range(ix_min, ix_max + 1):
            for iy in range(iy_min, iy_max + 1):
                key = (ix, iy)
                if key not in self._observed:
                    continue
                cx, cy = self._key_to_center(ix, iy)
                dx = cx - x
                dy = cy - y
                if (dx * dx + dy * dy) <= rr:
                    out.append(key)
        return out

    def _apply_label_to_point(self, x, y, label, source):
        with self._lock:
            keys = self._world_to_candidate_keys(x, y)
            if not keys:
                rospy.loginfo_throttle(1.0, "semantic_bev_editor %s: no observed cells under brush", source)
                return
            diff = []
            for key in keys:
                before = self._label_of_key_locked(key)
                if before == label:
                    continue
                self._set_key_label_locked(key, label)
                diff.append((key, before, label))
            if not diff:
                return
            self._history.append(diff)
            if len(self._history) > self.max_history:
                self._history.pop(0)
            self._dirty = True

    def paint_point_callback(self, msg):
        label = LABEL_SIDEWALK if self.sidewalk_only_editing else self.mode
        self._apply_label_to_point(float(msg.point.x), float(msg.point.y), label, "paint_point")

    def erase_point_callback(self, msg):
        self._apply_label_to_point(float(msg.point.x), float(msg.point.y), LABEL_OBSERVED, "erase_point")

    def clicked_point_callback(self, msg):
        x = float(msg.point.x)
        y = float(msg.point.y)
        if self.sidewalk_only_editing:
            with self._lock:
                keys = self._world_to_candidate_keys(x, y)
                if not keys:
                    rospy.loginfo_throttle(1.0, "semantic_bev_editor clicked_point: no observed cells under brush")
                    return
                if self.clicked_point_mode == "toggle":
                    allow = not any(self._label_of_key_locked(key) == LABEL_SIDEWALK for key in keys)
                else:
                    allow = (self.clicked_point_mode == "add")
            label = LABEL_SIDEWALK if allow else LABEL_OBSERVED
        else:
            label = self.mode
        self._apply_label_to_point(x, y, label, "clicked_point")

    def mode_callback(self, msg):
        mode = str(msg.data).strip().lower()
        if self.sidewalk_only_editing:
            if mode not in ("add", "erase", "toggle"):
                rospy.logwarn("semantic_bev_editor mode ignored: '%s' (use add|erase|toggle)", mode)
                return
            if mode == self.clicked_point_mode:
                return
            self.clicked_point_mode = mode
            rospy.loginfo("semantic_bev_editor clicked_point_mode set to '%s'", self.clicked_point_mode)
            return
        if mode not in ALL_LABELS:
            rospy.logwarn("semantic_bev_editor unknown mode: %s", mode)
            return
        self.mode = mode
        rospy.loginfo("semantic_bev_editor mode set to '%s'", self.mode)

    def clear_callback(self, _msg):
        with self._lock:
            diff = []
            for key in self._observed.keys():
                before = self._label_of_key_locked(key)
                after = self._base_label_map.get(key, LABEL_OBSERVED)
                if before == after:
                    continue
                self._set_key_label_locked(key, after)
                diff.append((key, before, after))
            if diff:
                self._history.append(diff)
                if len(self._history) > self.max_history:
                    self._history.pop(0)
                self._dirty = True
        rospy.loginfo("semantic_bev_editor restored current labels to base state")

    def undo_callback(self, _msg):
        with self._lock:
            if not self._history:
                rospy.loginfo("semantic_bev_editor undo: no history")
                return
            diff = self._history.pop()
            for key, before, _after in reversed(diff):
                self._set_key_label_locked(key, before)
            self._dirty = True
        rospy.loginfo("semantic_bev_editor undo applied")

    def _build_current_payload_locked(self):
        payload = {
            "meta": dict(self._base_payload.get("meta", {})) if self._base_payload else {},
            "classes": {
                label: [[ix, iy] for ix, iy in sorted(self._classes[label])]
                for label in EDITABLE_LABELS
            },
            "observed_cells": [
                [ix, iy, float(self._observed[(ix, iy)]), self._label_of_key_locked((ix, iy))]
                for ix, iy in sorted(self._observed.keys())
            ],
        }
        payload["meta"]["source_override_json"] = self.override_file_path
        return payload

    def _build_override_payload_locked(self):
        payload = {
            "description": "Edited by semantic_bev_editor.py",
            "paint": {
                LABEL_SIDEWALK: {"cells": [], "rectangles": []},
                LABEL_ROAD: {"cells": [], "rectangles": []},
                LABEL_CURB: {"cells": [], "rectangles": []},
            },
            "erase_cells": [],
            "erase_rectangles": [],
        }
        for key in sorted(self._observed.keys()):
            base = self._base_label_map.get(key, LABEL_OBSERVED)
            current = self._label_of_key_locked(key)
            if base == current:
                continue
            cell = [int(key[0]), int(key[1])]
            if current == LABEL_OBSERVED:
                payload["erase_cells"].append(cell)
            else:
                payload["paint"][current]["cells"].append(cell)
        return payload

    def _build_drivable_state_payload_locked(self):
        sidewalk_keys = sorted(self._classes[LABEL_SIDEWALK])
        cells = []
        z_values = []
        for key in sidewalk_keys:
            if key not in self._observed:
                continue
            z = float(self._observed[key])
            cells.append([int(key[0]), int(key[1]), z])
            z_values.append(z)

        last_seed_xy = None
        if sidewalk_keys:
            seed_ix, seed_iy = sidewalk_keys[len(sidewalk_keys) // 2]
            last_seed_xy = list(self._key_to_center(seed_ix, seed_iy))

        if z_values:
            last_odom_z = float(sum(z_values) / float(len(z_values)))
        else:
            last_odom_z = 0.0

        return {
            "version": 1,
            "grid_resolution_m": float(self.grid_resolution_m),
            "cells": cells,
            "risk_cells": [],
            "last_seed_xy": last_seed_xy,
            "last_odom_z": last_odom_z,
            "saved_at": float(rospy.Time.now().to_sec()),
        }

    def save_callback(self, _msg):
        self._save_all(log_prefix="topic")

    def _save_all(self, log_prefix="manual"):
        with self._lock:
            state_payload = self._build_current_payload_locked()
            override_payload = self._build_override_payload_locked()
            drivable_payload = self._build_drivable_state_payload_locked()
        state_dir = os.path.dirname(self.edited_state_file_path)
        override_dir = os.path.dirname(self.override_file_path)
        if state_dir:
            os.makedirs(state_dir, exist_ok=True)
        if override_dir:
            os.makedirs(override_dir, exist_ok=True)
        with open(self.edited_state_file_path, "w", encoding="utf-8") as f:
            json.dump(state_payload, f, indent=2)
        with open(self.override_file_path, "w", encoding="utf-8") as f:
            json.dump(override_payload, f, indent=2)
        if self.export_drivable_state_on_save:
            drivable_dir = os.path.dirname(self.drivable_state_file_path)
            if drivable_dir:
                os.makedirs(drivable_dir, exist_ok=True)
            with open(self.drivable_state_file_path, "w", encoding="utf-8") as f:
                json.dump(drivable_payload, f, indent=2)
            if self.notify_drivable_reload_on_save:
                self.pub_drivable_reload.publish(Empty())
        rospy.loginfo(
            "semantic_bev_editor saved (%s): %s | override: %s | drivable: %s",
            log_prefix,
            self.edited_state_file_path,
            self.override_file_path,
            self.drivable_state_file_path if self.export_drivable_state_on_save else "disabled",
        )

    def load_callback(self, _msg):
        if os.path.isfile(self.edited_state_file_path):
            self._load_state_from_path(self.edited_state_file_path, keep_as_base=False)
            rospy.loginfo("semantic_bev_editor loaded edited state: %s", self.edited_state_file_path)
        else:
            self._load_base_state()
            rospy.loginfo("semantic_bev_editor edited state missing, restored base state")

    def _build_grid_msg_locked(self):
        grid = OccupancyGrid()
        grid.header.stamp = rospy.Time.now()
        grid.header.frame_id = self.frame_id
        width = max(0, self._ix_max - self._ix_min + 1)
        height = max(0, self._iy_max - self._iy_min + 1)
        grid.info.resolution = self.grid_resolution_m
        grid.info.width = width
        grid.info.height = height
        grid.info.origin.position.x = self._ix_min * self.grid_resolution_m
        grid.info.origin.position.y = self._iy_min * self.grid_resolution_m
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.w = 1.0
        data = [-1] * (width * height)
        for (ix, iy), _z in self._observed.items():
            gx = ix - self._ix_min
            gy = iy - self._iy_min
            if gx < 0 or gy < 0 or gx >= width or gy >= height:
                continue
            idx = gy * width + gx
            label = self._label_of_key_locked((ix, iy))
            if self.sidewalk_only_editing:
                data[idx] = GRID_VALUES[LABEL_SIDEWALK] if label == LABEL_SIDEWALK else -1
            else:
                data[idx] = GRID_VALUES[label]
        grid.data = data
        return grid

    def _build_cloud_msg_locked(self):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.frame_id
        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("rgb", 12, PointField.FLOAT32, 1),
        ]
        points = []
        for key in sorted(self._observed.keys()):
            label = self._label_of_key_locked(key)
            if self.sidewalk_only_editing and label != LABEL_SIDEWALK:
                continue
            ix, iy = key
            x, y = self._key_to_center(ix, iy)
            z = float(self._observed[key])
            rgb = rgb_to_float(RGB_VALUES[label])
            points.append((x, y, z, rgb))
        return point_cloud2.create_cloud(header, fields, points)

    def on_timer(self, _evt):
        with self._lock:
            if not self._dirty:
                return
            grid = self._build_grid_msg_locked()
            cloud = self._build_cloud_msg_locked()
            self._dirty = False
        self.pub_grid.publish(grid)
        self.pub_cloud.publish(cloud)

    def _watch_node_timer(self, _evt):
        try:
            names = set(rosnode.get_node_names())
        except Exception:
            return
        if self.watch_node_name in names:
            self._watch_node_seen = True
            return
        if self._watch_node_seen and (not self._watch_node_saved):
            self._save_all(log_prefix="watch_node_exit")
            self._watch_node_saved = True

    def on_shutdown(self):
        if self.save_on_shutdown:
            try:
                self._save_all(log_prefix="shutdown")
            except Exception as e:
                rospy.logwarn("semantic_bev_editor shutdown save failed: %s", str(e))


if __name__ == "__main__":
    rospy.init_node("semantic_bev_editor")
    try:
        SemanticBEVEditor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
