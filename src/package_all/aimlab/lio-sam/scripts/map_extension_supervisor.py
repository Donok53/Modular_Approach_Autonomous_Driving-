#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import signal
import shutil
import struct
import subprocess
import threading
import time

import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Point
from lio_sam.srv import save_map, save_mapRequest
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Empty, Header, String
from std_srvs.srv import Trigger, TriggerResponse
from visualization_msgs.msg import Marker

try:
    from interactive_markers.interactive_marker_server import InteractiveMarkerServer
    from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl
except Exception:
    InteractiveMarkerServer = None
    InteractiveMarker = None
    InteractiveMarkerControl = None


class MapExtensionSupervisor:
    def __init__(self):
        self.fixed_frame = rospy.get_param("~fixed_frame", "map")
        self.localizer_odom_topic = rospy.get_param(
            "~localizer_odom_topic", "/lio_localizer/odometry/optimization"
        )
        self.mapping_odom_topic = rospy.get_param("~mapping_odom_topic", "/lio_sam/mapping/odometry")
        self.mapping_path_topic = rospy.get_param("~mapping_path_topic", "/lio_sam/mapping/path")
        self.preview_source_cloud_topic = rospy.get_param(
            "~preview_source_cloud_topic", "/lio_sam/mapping/cloud_registered_raw"
        )
        self.preview_cloud_topic = rospy.get_param("~preview_cloud_topic", "/map_extension/preview_cloud")
        self.transformed_path_topic = rospy.get_param(
            "~transformed_path_topic", "/map_extension/transformed_path"
        )
        self.status_topic = rospy.get_param("~status_topic", "/map_extension/status")
        self.status_marker_topic = rospy.get_param(
            "~status_marker_topic", "/map_extension/status_marker"
        )
        self.reload_topic = rospy.get_param("~astar_reload_topic", "/astar/reload_map")

        self.start_service_name = rospy.get_param("~start_service", "/map_extension/start")
        self.finish_service_name = rospy.get_param("~finish_service", "/map_extension/finish")
        self.cancel_service_name = rospy.get_param("~cancel_service", "/map_extension/cancel")
        self.save_map_service_name = rospy.get_param("~save_map_service", "/lio_sam/save_map")
        self.map_sync_service_name = rospy.get_param(
            "~map_sync_service", "/lio_sam_map_sync/sync_now"
        )
        self.map_sync_disable_shutdown_service_name = rospy.get_param(
            "~map_sync_disable_shutdown_service", "/lio_sam_map_sync/disable_shutdown_sync"
        )
        self.trajectory_save_service_name = rospy.get_param(
            "~trajectory_save_service", "/trajectory_osm_exporter/save_now"
        )

        self.mapping_launch_file = os.path.expanduser(rospy.get_param("~mapping_launch_file", ""))
        self.mapping_launch_args = self._param_list(rospy.get_param("~mapping_launch_args", []))
        self.roslaunch_executable = rospy.get_param(
            "~roslaunch_executable", "/opt/ros/noetic/bin/roslaunch"
        )
        self.source_dir = os.path.expanduser(rospy.get_param("~source_dir", "~/Downloads/test"))
        self.transform_pcd_files = self._param_list(
            rospy.get_param(
                "~transform_pcd_files",
                ["GlobalMap.pcd", "CornerMap.pcd", "SurfMap.pcd"],
            )
        )
        self.clear_source_dir_on_start = bool(rospy.get_param("~clear_source_dir_on_start", False))

        self.localization_max_age_s = max(0.1, float(rospy.get_param("~localization_max_age_s", 1.0)))
        self.require_stationary_start = bool(rospy.get_param("~require_stationary_start", True))
        self.start_max_linear_speed_mps = max(
            0.0, float(rospy.get_param("~start_max_linear_speed_mps", 0.12))
        )
        self.start_max_angular_speed_rps = max(
            0.0, float(rospy.get_param("~start_max_angular_speed_rps", 0.20))
        )
        self.service_timeout_s = max(0.5, float(rospy.get_param("~service_timeout_s", 30.0)))
        self.require_trajectory_save = bool(rospy.get_param("~require_trajectory_save", False))
        self.save_map_resolution = max(0.0, float(rospy.get_param("~save_map_resolution", 0.12)))
        self.preview_voxel_leaf_size = max(
            0.01, float(rospy.get_param("~preview_voxel_leaf_size", 0.12))
        )
        self.preview_point_stride = max(1, int(rospy.get_param("~preview_point_stride", 4)))
        self.preview_max_points = max(1000, int(rospy.get_param("~preview_max_points", 150000)))
        self.preview_publish_hz = max(0.2, float(rospy.get_param("~preview_publish_hz", 2.0)))
        self.control_refresh_s = max(0.2, float(rospy.get_param("~control_refresh_s", 1.0)))
        self.enable_interactive_controls = bool(rospy.get_param("~enable_interactive_controls", False))

        self._lock = threading.RLock()
        self._launch_parent = None
        self._running = False
        self._state = "idle"
        self._last_message = "ready"
        self._last_localizer_odom = None
        self._last_localizer_time = rospy.Time(0)
        self._start_localizer_matrix = None
        self._start_localizer_quat = (0.0, 0.0, 0.0, 1.0)
        self._session_to_map = None
        self._session_quat = None
        self._preview_points = {}
        self._last_preview_cloud = None

        self.pub_preview = rospy.Publisher(self.preview_cloud_topic, PointCloud2, queue_size=1, latch=True)
        self.pub_path = rospy.Publisher(self.transformed_path_topic, Path, queue_size=1, latch=True)
        self.pub_status = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)
        self.pub_status_marker = rospy.Publisher(
            self.status_marker_topic, Marker, queue_size=1, latch=True
        )
        self.pub_reload = rospy.Publisher(self.reload_topic, Empty, queue_size=1)

        self.sub_localizer = rospy.Subscriber(
            self.localizer_odom_topic, Odometry, self._on_localizer_odom, queue_size=10
        )
        self.sub_mapping_odom = rospy.Subscriber(
            self.mapping_odom_topic, Odometry, self._on_mapping_odom, queue_size=10
        )
        self.sub_mapping_path = rospy.Subscriber(
            self.mapping_path_topic, Path, self._on_mapping_path, queue_size=2
        )
        self.sub_preview_cloud = rospy.Subscriber(
            self.preview_source_cloud_topic, PointCloud2, self._on_preview_cloud, queue_size=2
        )

        self.srv_start = rospy.Service(self.start_service_name, Trigger, self.handle_start)
        self.srv_finish = rospy.Service(self.finish_service_name, Trigger, self.handle_finish)
        self.srv_cancel = rospy.Service(self.cancel_service_name, Trigger, self.handle_cancel)

        self.marker_server = None
        if self.enable_interactive_controls and InteractiveMarkerServer is not None:
            self.marker_server = InteractiveMarkerServer("map_extension_controls")
            self._refresh_interactive_markers()
        elif self.enable_interactive_controls:
            rospy.logwarn("map_extension_supervisor: interactive_markers is unavailable; services still work")

        self.preview_timer = rospy.Timer(
            rospy.Duration(1.0 / self.preview_publish_hz), self._publish_preview_timer
        )
        self.control_timer = rospy.Timer(rospy.Duration(self.control_refresh_s), self._control_timer_cb)
        rospy.on_shutdown(self._shutdown_mapping_launch)

        rospy.loginfo(
            "map_extension_supervisor started | start=%s finish=%s cancel=%s preview=%s scene_controls=%s",
            self.start_service_name,
            self.finish_service_name,
            self.cancel_service_name,
            self.preview_cloud_topic,
            str(self.enable_interactive_controls),
        )
        self._publish_status("idle", "waiting for localization")

    @staticmethod
    def _param_list(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value]
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return [str(value)]

    @staticmethod
    def _quat_normalize(q):
        n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        if n <= 1e-12:
            return (0.0, 0.0, 0.0, 1.0)
        return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)

    @classmethod
    def _quat_inverse(cls, q):
        q = cls._quat_normalize(q)
        return (-q[0], -q[1], -q[2], q[3])

    @classmethod
    def _quat_multiply(cls, q0, q1):
        x0, y0, z0, w0 = q0
        x1, y1, z1, w1 = q1
        return cls._quat_normalize(
            (
                w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
                w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            )
        )

    @classmethod
    def _matrix_from_translation_quaternion(cls, translation, quat):
        x, y, z, w = cls._quat_normalize(quat)
        tx, ty, tz = translation
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), tx],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), ty],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), tz],
            [0.0, 0.0, 0.0, 1.0],
        ]

    @staticmethod
    def _matrix_multiply(a, b):
        out = [[0.0 for _ in range(4)] for _ in range(4)]
        for r in range(4):
            for c in range(4):
                out[r][c] = sum(a[r][k] * b[k][c] for k in range(4))
        return out

    @staticmethod
    def _matrix_inverse_rigid(m):
        out = [
            [m[0][0], m[1][0], m[2][0], 0.0],
            [m[0][1], m[1][1], m[2][1], 0.0],
            [m[0][2], m[1][2], m[2][2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        tx, ty, tz = m[0][3], m[1][3], m[2][3]
        out[0][3] = -(out[0][0] * tx + out[0][1] * ty + out[0][2] * tz)
        out[1][3] = -(out[1][0] * tx + out[1][1] * ty + out[1][2] * tz)
        out[2][3] = -(out[2][0] * tx + out[2][1] * ty + out[2][2] * tz)
        return out

    @staticmethod
    def _copy_matrix(m):
        return [row[:] for row in m]

    @classmethod
    def _pose_quat_from_odom(cls, msg):
        q = msg.pose.pose.orientation
        return cls._quat_normalize((float(q.x), float(q.y), float(q.z), float(q.w)))

    @classmethod
    def _pose_matrix_from_odom(cls, msg):
        q = msg.pose.pose.orientation
        p = msg.pose.pose.position
        return cls._matrix_from_translation_quaternion(
            (float(p.x), float(p.y), float(p.z)),
            (float(q.x), float(q.y), float(q.z), float(q.w)),
        )

    @staticmethod
    def _transform_xyz(matrix, x, y, z):
        return (
            matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
            matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
            matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
        )

    @staticmethod
    def _cloud_fields():
        return [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

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
                header[key] = parts[1:]
                if key == "DATA":
                    header["DATA_START"] = f.tell()
                    break

        fields = header.get("FIELDS", [])
        sizes = [int(v) for v in header.get("SIZE", [])]
        types = header.get("TYPE", [])
        counts = [int(v) for v in header.get("COUNT", ["1"] * len(fields))]
        if not (len(fields) == len(sizes) == len(types) == len(counts)):
            raise ValueError("pcd field metadata mismatch: %s" % path)
        point_count = int(header.get("POINTS", [header.get("WIDTH", ["0"])[0]])[0])
        data_mode = header.get("DATA", [""])[0].lower()
        offset = 0
        layout = []
        for name, size, type_code, count in zip(fields, sizes, types, counts):
            struct_code = cls._pcd_type_to_struct(type_code, size)
            if struct_code is None:
                raise ValueError("unsupported pcd field type %s/%s in %s" % (type_code, size, path))
            layout.append(
                {
                    "name": name,
                    "offset": offset,
                    "size": size,
                    "type": type_code,
                    "count": count,
                    "struct_code": struct_code,
                }
            )
            offset += size * count
        header["POINT_COUNT"] = point_count
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
        field_names = header.get("FIELDS", [])
        points = []
        if header["DATA_MODE"] == "ascii":
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
        if header["DATA_MODE"] != "binary":
            raise ValueError("unsupported pcd mode %s in %s" % (header["DATA_MODE"], path))

        def unpack_scalar(blob, spec):
            return struct.unpack_from("<" + spec["struct_code"], blob, spec["offset"])[0]

        with open(path, "rb") as f:
            f.seek(header["DATA_START"])
            for _ in range(header["POINT_COUNT"]):
                blob = f.read(header["POINT_STEP"])
                if len(blob) < header["POINT_STEP"]:
                    break
                x = float(unpack_scalar(blob, fields["x"]))
                y = float(unpack_scalar(blob, fields["y"]))
                z = float(unpack_scalar(blob, fields["z"]))
                intensity = float(unpack_scalar(blob, fields["intensity"])) if "intensity" in fields else 0.0
                points.append((x, y, z, intensity))
        return points

    @staticmethod
    def _write_pcd(path, points):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        header = (
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\n"
            "FIELDS x y z intensity\n"
            "SIZE 4 4 4 4\n"
            "TYPE F F F F\n"
            "COUNT 1 1 1 1\n"
            "WIDTH {n}\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            "POINTS {n}\n"
            "DATA binary\n"
        ).format(n=len(points))
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            for x, y, z, intensity in points:
                f.write(struct.pack("<ffff", float(x), float(y), float(z), float(intensity)))

    def _on_localizer_odom(self, msg):
        with self._lock:
            self._last_localizer_odom = msg
            self._last_localizer_time = rospy.Time.now()

    def _on_mapping_odom(self, msg):
        with self._lock:
            if not self._running or self._session_to_map is not None:
                return
            if self._start_localizer_matrix is None:
                return
            session_start = self._pose_matrix_from_odom(msg)
            self._session_to_map = self._matrix_multiply(
                self._start_localizer_matrix, self._matrix_inverse_rigid(session_start)
            )
            self._session_quat = self._quat_multiply(
                self._start_localizer_quat, self._quat_inverse(self._pose_quat_from_odom(msg))
            )
            self._publish_status("running", "extension transform ready")
            rospy.loginfo("map_extension_supervisor: session-to-map transform is ready")

    def _on_mapping_path(self, msg):
        with self._lock:
            if not self._running or self._session_to_map is None:
                return
            matrix = self._copy_matrix(self._session_to_map)
            q_session = tuple(self._session_quat)
        out = Path()
        out.header = msg.header
        out.header.frame_id = self.fixed_frame
        out.header.stamp = rospy.Time.now()
        for ps in msg.poses:
            pose = ps.pose
            tx, ty, tz = self._transform_xyz(
                matrix,
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
            )
            q_in = (
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            )
            q_out = self._quat_multiply(q_session, q_in)
            new_ps = type(ps)()
            new_ps.header = ps.header
            new_ps.header.frame_id = self.fixed_frame
            new_ps.header.stamp = out.header.stamp
            new_ps.pose.position.x = tx
            new_ps.pose.position.y = ty
            new_ps.pose.position.z = tz
            new_ps.pose.orientation.x = q_out[0]
            new_ps.pose.orientation.y = q_out[1]
            new_ps.pose.orientation.z = q_out[2]
            new_ps.pose.orientation.w = q_out[3]
            out.poses.append(new_ps)
        self.pub_path.publish(out)

    def _on_preview_cloud(self, msg):
        with self._lock:
            if not self._running or self._session_to_map is None:
                return
            matrix = self._copy_matrix(self._session_to_map)
            leaf = self.preview_voxel_leaf_size
            stride = self.preview_point_stride
        field_names = [field.name for field in msg.fields]
        use_intensity = "intensity" in field_names
        names = ("x", "y", "z", "intensity") if use_intensity else ("x", "y", "z")
        try:
            iterator = pc2.read_points(msg, field_names=names, skip_nans=True)
            local_updates = {}
            for index, row in enumerate(iterator):
                if index % stride:
                    continue
                x, y, z = float(row[0]), float(row[1]), float(row[2])
                if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                    continue
                tx, ty, tz = self._transform_xyz(matrix, x, y, z)
                intensity = float(row[3]) if use_intensity and len(row) > 3 else 0.0
                key = (int(math.floor(tx / leaf)), int(math.floor(ty / leaf)), int(math.floor(tz / leaf)))
                local_updates[key] = (tx, ty, tz, intensity)
        except Exception as e:
            rospy.logwarn_throttle(2.0, "map_extension_supervisor: preview cloud transform failed: %s", str(e))
            return

        with self._lock:
            self._preview_points.update(local_updates)
            overflow = len(self._preview_points) - self.preview_max_points
            if overflow > 0:
                for key in list(self._preview_points.keys())[:overflow]:
                    self._preview_points.pop(key, None)

    def _publish_preview_timer(self, _event):
        with self._lock:
            points = list(self._preview_points.values())
            last_cloud = self._last_preview_cloud
        if points:
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = self.fixed_frame
            cloud = pc2.create_cloud(header, self._cloud_fields(), points)
            with self._lock:
                self._last_preview_cloud = cloud
            self.pub_preview.publish(cloud)
        elif last_cloud is not None:
            last_cloud.header.stamp = rospy.Time.now()
            self.pub_preview.publish(last_cloud)

    def _clear_preview_outputs(self):
        with self._lock:
            self._preview_points.clear()
            self._last_preview_cloud = None
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.fixed_frame
        self.pub_preview.publish(pc2.create_cloud(header, self._cloud_fields(), []))
        path = Path()
        path.header = header
        self.pub_path.publish(path)

    def _localized_ready(self):
        with self._lock:
            odom = self._last_localizer_odom
            stamp = self._last_localizer_time
        if odom is None:
            return False, "no localization odometry yet"
        age = (rospy.Time.now() - stamp).to_sec()
        if age > self.localization_max_age_s:
            return False, "localization odometry stale %.2fs" % age
        if self.require_stationary_start:
            twist = odom.twist.twist
            linear_speed = math.sqrt(
                twist.linear.x * twist.linear.x
                + twist.linear.y * twist.linear.y
                + twist.linear.z * twist.linear.z
            )
            angular_speed = math.sqrt(
                twist.angular.x * twist.angular.x
                + twist.angular.y * twist.angular.y
                + twist.angular.z * twist.angular.z
            )
            if linear_speed > self.start_max_linear_speed_mps:
                return (
                    False,
                    "robot moving too fast for map extension start %.2fm/s > %.2fm/s"
                    % (linear_speed, self.start_max_linear_speed_mps),
                )
            if angular_speed > self.start_max_angular_speed_rps:
                return (
                    False,
                    "robot rotating too fast for map extension start %.2frad/s > %.2frad/s"
                    % (angular_speed, self.start_max_angular_speed_rps),
                )
        return True, "localized age %.2fs" % age

    def _start_mapping_launch(self):
        if not self.mapping_launch_file:
            raise RuntimeError("mapping_launch_file is empty")
        if not os.path.isfile(self.mapping_launch_file):
            raise RuntimeError("mapping_launch_file missing: %s" % self.mapping_launch_file)
        command = [self.roslaunch_executable, self.mapping_launch_file] + list(self.mapping_launch_args)
        self._launch_parent = subprocess.Popen(command, preexec_fn=os.setsid)
        rospy.loginfo(
            "map_extension_supervisor: started mapping launch pid=%s cmd=%s",
            self._launch_parent.pid,
            " ".join(command),
        )

    def _shutdown_mapping_launch(self):
        with self._lock:
            parent = self._launch_parent
            self._launch_parent = None
            self._running = False
        if parent is not None:
            try:
                pgid = os.getpgid(parent.pid)
                os.killpg(pgid, signal.SIGINT)
                try:
                    parent.wait(timeout=8.0)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGTERM)
                    parent.wait(timeout=4.0)
            except Exception as e:
                rospy.logwarn("map_extension_supervisor: mapping launch shutdown failed: %s", str(e))

    def _call_trigger(self, service_name, timeout_s=None):
        timeout_s = self.service_timeout_s if timeout_s is None else timeout_s
        rospy.wait_for_service(service_name, timeout=timeout_s)
        proxy = rospy.ServiceProxy(service_name, Trigger)
        return proxy()

    def _disable_child_shutdown_sync(self):
        try:
            resp = self._call_trigger(self.map_sync_disable_shutdown_service_name, timeout_s=1.0)
            if not resp.success:
                rospy.logwarn("map_extension_supervisor: disable shutdown sync returned false: %s", resp.message)
        except Exception as e:
            rospy.logwarn("map_extension_supervisor: shutdown sync disable service unavailable: %s", str(e))

    def _call_save_map(self):
        rospy.wait_for_service(self.save_map_service_name, timeout=self.service_timeout_s)
        proxy = rospy.ServiceProxy(self.save_map_service_name, save_map)
        req = save_mapRequest()
        req.resolution = self.save_map_resolution
        req.destination = ""
        rospy.loginfo(
            "map_extension_supervisor: calling save_map resolution=%.3f destination=default",
            req.resolution,
        )
        return proxy(req)

    def _transform_source_pcds(self):
        with self._lock:
            matrix = None if self._session_to_map is None else self._copy_matrix(self._session_to_map)
        if matrix is None:
            raise RuntimeError("session-to-map transform is not ready")
        transformed = 0
        for rel in self.transform_pcd_files:
            path = os.path.join(self.source_dir, rel)
            if not os.path.isfile(path):
                rospy.logwarn("map_extension_supervisor: source pcd missing, skip transform: %s", path)
                continue
            points = self._read_pcd_points(path)
            out = []
            for x, y, z, intensity in points:
                tx, ty, tz = self._transform_xyz(matrix, x, y, z)
                out.append((tx, ty, tz, intensity))
            self._write_pcd(path, out)
            transformed += 1
            rospy.loginfo(
                "map_extension_supervisor: transformed %s into %s frame (%d pts)",
                rel,
                self.fixed_frame,
                len(out),
            )
        return transformed

    def _clear_source_dir(self):
        if not self.clear_source_dir_on_start:
            return
        if os.path.isdir(self.source_dir):
            shutil.rmtree(self.source_dir)
        os.makedirs(self.source_dir, exist_ok=True)

    def handle_start(self, _req):
        with self._lock:
            if self._running:
                return TriggerResponse(False, "map extension is already running")
        ready, message = self._localized_ready()
        if not ready:
            self._publish_status("waiting_localization", message)
            return TriggerResponse(False, message)
        try:
            self._clear_preview_outputs()
            with self._lock:
                self._start_localizer_matrix = self._pose_matrix_from_odom(self._last_localizer_odom)
                self._start_localizer_quat = self._pose_quat_from_odom(self._last_localizer_odom)
                self._session_to_map = None
                self._session_quat = None
                self._running = True
            self._clear_source_dir()
            self._publish_status("starting", "starting mapping extension")
            self._start_mapping_launch()
            self._publish_status("running", "waiting for mapping odometry")
            return TriggerResponse(True, "map extension started")
        except Exception as e:
            with self._lock:
                self._running = False
            self._publish_status("error", str(e))
            rospy.logwarn("map_extension_supervisor: start failed: %s", str(e))
            return TriggerResponse(False, str(e))

    def handle_finish(self, _req):
        with self._lock:
            running = self._running
            transform_ready = self._session_to_map is not None
        if not running:
            return TriggerResponse(False, "map extension is not running")
        if not transform_ready:
            return TriggerResponse(False, "mapping odometry was not received yet")
        try:
            self._publish_status("saving", "saving extension trajectory")
            try:
                traj_resp = self._call_trigger(self.trajectory_save_service_name, timeout_s=5.0)
                if not traj_resp.success:
                    msg = "trajectory save skipped: %s" % traj_resp.message
                    if self.require_trajectory_save:
                        raise RuntimeError(msg)
                    rospy.logwarn("map_extension_supervisor: %s", msg)
            except Exception as e:
                if self.require_trajectory_save:
                    raise
                rospy.logwarn("map_extension_supervisor: trajectory save service unavailable: %s", str(e))

            self._publish_status("saving", "saving extension map")
            save_resp = self._call_save_map()
            if not getattr(save_resp, "success", False):
                raise RuntimeError("save_map returned failure")

            self._publish_status("saving", "transforming extension pcds")
            transformed = self._transform_source_pcds()
            if transformed <= 0:
                raise RuntimeError("no extension pcd files were transformed")

            self._publish_status("syncing", "merging extension assets")
            sync_resp = self._call_trigger(self.map_sync_service_name)
            if not sync_resp.success:
                raise RuntimeError("map sync failed: %s" % sync_resp.message)

            self.pub_reload.publish(Empty())
            self._clear_preview_outputs()
            self._shutdown_mapping_launch()
            self._publish_status("idle", "extension saved and merged")
            return TriggerResponse(True, "map extension saved and merged")
        except Exception as e:
            self._publish_status("error", str(e))
            rospy.logwarn("map_extension_supervisor: finish failed: %s", str(e))
            return TriggerResponse(False, str(e))

    def handle_cancel(self, _req):
        with self._lock:
            was_running = self._running
        self._clear_preview_outputs()
        if not was_running:
            self._publish_status("idle", "cancel ignored; extension was not running")
            return TriggerResponse(True, "extension was not running")
        self._publish_status("cancelling", "stopping mapping extension")
        self._disable_child_shutdown_sync()
        self._shutdown_mapping_launch()
        self._publish_status("idle", "extension cancelled")
        return TriggerResponse(True, "map extension cancelled")

    def _button_feedback(self, name):
        if name == "start":
            self.handle_start(None)
        elif name == "finish":
            self.handle_finish(None)
        elif name == "cancel":
            self.handle_cancel(None)

    def _current_control_origin(self):
        with self._lock:
            odom = self._last_localizer_odom
        if odom is None:
            return 0.0, 0.0, 0.7
        p = odom.pose.pose.position
        return float(p.x), float(p.y), float(p.z) + 1.0

    def _make_button_marker(self, name, label, color, offset_y):
        marker = InteractiveMarker()
        marker.header.frame_id = self.fixed_frame
        marker.header.stamp = rospy.Time.now()
        marker.name = "map_extension_" + name
        marker.description = label
        marker.scale = 0.8
        x, y, z = self._current_control_origin()
        marker.pose.position.x = x
        marker.pose.position.y = y + offset_y
        marker.pose.position.z = z

        cube = Marker()
        cube.type = Marker.CUBE
        cube.scale.x = 0.55
        cube.scale.y = 0.28
        cube.scale.z = 0.12
        cube.color.r, cube.color.g, cube.color.b, cube.color.a = color

        text = Marker()
        text.type = Marker.TEXT_VIEW_FACING
        text.pose.position.z = 0.22
        text.scale.z = 0.16
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 1.0
        text.text = label

        control = InteractiveMarkerControl()
        control.always_visible = True
        control.interaction_mode = InteractiveMarkerControl.BUTTON
        control.markers.append(cube)
        control.markers.append(text)
        marker.controls.append(control)
        return marker

    def _refresh_interactive_markers(self):
        if self.marker_server is None:
            return
        buttons = [
            ("start", "START EXT", (0.05, 0.55, 0.20, 0.92), -0.45),
            ("finish", "SAVE EXT", (0.05, 0.25, 0.85, 0.92), 0.0),
            ("cancel", "CANCEL", (0.75, 0.10, 0.10, 0.92), 0.45),
        ]
        for name, label, color, offset_y in buttons:
            marker = self._make_button_marker(name, label, color, offset_y)
            self.marker_server.insert(marker, lambda feedback, n=name: self._button_feedback(n))
        self.marker_server.applyChanges()

    def _control_timer_cb(self, _event):
        self._refresh_interactive_markers()
        with self._lock:
            state = self._state
            message = self._last_message
        self._publish_status(state, message)

    def _publish_status(self, state, message):
        with self._lock:
            self._state = state
            self._last_message = message
        text = "%s | %s" % (state, message)
        self.pub_status.publish(String(data=text))

        marker = Marker()
        marker.header.frame_id = self.fixed_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "map_extension_status"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        x, y, z = self._current_control_origin()
        marker.pose.position = Point(x, y, z + 0.45)
        marker.scale.z = 0.22
        marker.color.a = 1.0
        if state in ("running", "syncing", "saving"):
            marker.color.r, marker.color.g, marker.color.b = 0.1, 0.9, 0.4
        elif state == "error":
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.15, 0.1
        else:
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.9, 0.1
        marker.text = "Map Extension: " + text
        self.pub_status_marker.publish(marker)


def main():
    rospy.init_node("map_extension_supervisor", anonymous=False)
    MapExtensionSupervisor()
    rospy.spin()


if __name__ == "__main__":
    main()
