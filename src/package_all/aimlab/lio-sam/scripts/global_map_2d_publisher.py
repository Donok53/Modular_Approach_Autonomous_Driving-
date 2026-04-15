#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import struct

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2


class GlobalMap2DPublisher:
    def __init__(self):
        self.pcd_path = os.path.expanduser(
            rospy.get_param(
                "~pcd_path",
                "/home/byeongjae/code/Modular_Approach_Autonomous_Driving-/src/package_all/aimlab/lio-localizer/map/test/GlobalMap2D.pcd",
            )
        )
        self.topic = rospy.get_param("~topic", "/lio_localizer/localization/global_map_2d")
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.z_offset = float(rospy.get_param("~z_offset", -0.05))
        self.reload_period_s = max(0.5, float(rospy.get_param("~reload_period_s", 2.0)))

        self._last_mtime_ns = None
        self._last_cloud = None
        self._pub = rospy.Publisher(self.topic, PointCloud2, queue_size=1, latch=True)
        self._timer = rospy.Timer(rospy.Duration(self.reload_period_s), self._on_timer)
        self._reload_if_needed(force=True)

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
    def _read_pcd_points(cls, path):
        header = cls._read_pcd_header(path)
        fields = {item["name"]: item for item in header["LAYOUT"]}
        if "x" not in fields or "y" not in fields or "z" not in fields:
            raise ValueError("pcd missing x/y/z fields: %s" % path)

        points = []
        data_mode = header["DATA_MODE"]
        point_count = header["POINT_COUNT"]
        point_step = header["POINT_STEP"]
        field_names = header.get("FIELDS", [])

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
                    x = float(row[field_index["x"]])
                    y = float(row[field_index["y"]])
                    z = float(row[field_index["z"]])
                    intensity = float(row[field_index["intensity"]]) if "intensity" in field_index else 0.0
                    points.append((x, y, z, intensity))
            return points

        if data_mode != "binary":
            raise ValueError("unsupported pcd data mode '%s' in %s" % (data_mode, path))

        def unpack_scalar(blob, spec):
            fmt = "<" + spec["struct_code"]
            return struct.unpack_from(fmt, blob, spec["offset"])[0]

        with open(path, "rb") as f:
            f.seek(header["DATA_START"])
            for _ in range(point_count):
                blob = f.read(point_step)
                if len(blob) < point_step:
                    break
                x = float(unpack_scalar(blob, fields["x"]))
                y = float(unpack_scalar(blob, fields["y"]))
                z = float(unpack_scalar(blob, fields["z"]))
                intensity = float(unpack_scalar(blob, fields["intensity"])) if "intensity" in fields else 0.0
                points.append((x, y, z, intensity))
        return points

    def _build_cloud(self, points):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.frame_id
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        return pc2.create_cloud(header, fields, points)

    def _reload_if_needed(self, force=False):
        if not os.path.isfile(self.pcd_path):
            rospy.logwarn_throttle(5.0, "global_map_2d_publisher: missing pcd: %s" % self.pcd_path)
            return
        try:
            mtime_ns = os.stat(self.pcd_path).st_mtime_ns
        except OSError:
            return
        if not force and self._last_mtime_ns == mtime_ns and self._last_cloud is not None:
            return
        try:
            raw_points = self._read_pcd_points(self.pcd_path)
        except Exception as e:
            rospy.logwarn("global_map_2d_publisher: failed to read %s: %s", self.pcd_path, str(e))
            return
        points = [(x, y, z + self.z_offset, intensity) for x, y, z, intensity in raw_points]
        self._last_cloud = self._build_cloud(points)
        self._last_mtime_ns = mtime_ns
        self._pub.publish(self._last_cloud)
        rospy.loginfo(
            "global_map_2d_publisher: published %d pts from %s on %s (z_offset=%.3f)",
            len(points),
            self.pcd_path,
            self.topic,
            self.z_offset,
        )

    def _on_timer(self, _event):
        self._reload_if_needed(force=False)
        if self._last_cloud is not None:
            self._last_cloud.header.stamp = rospy.Time.now()
            self._pub.publish(self._last_cloud)


if __name__ == "__main__":
    rospy.init_node("global_map_2d_publisher")
    GlobalMap2DPublisher()
    rospy.spin()
