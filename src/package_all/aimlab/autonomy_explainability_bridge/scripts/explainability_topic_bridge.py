#!/usr/bin/env python3
import json
import math
import threading
import time

import rospy
from dynamic_window_approach.msg import BehaviorCommand, TrackedObjectArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray


def _stamp_to_float(stamp):
    if stamp is None:
        return 0.0
    try:
        return float(stamp.to_sec())
    except Exception:
        return 0.0


def _round(value, digits=3):
    if value is None:
        return None
    try:
        if not math.isfinite(float(value)):
            return None
        return round(float(value), digits)
    except Exception:
        return None


def _quat_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _point_distance_xy(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


class ExplainabilityTopicBridge:
    def __init__(self):
        self.publish_hz = max(0.2, float(rospy.get_param("~publish_hz", 5.0)))
        self.manifest_publish_period_s = max(
            1.0, float(rospy.get_param("~manifest_publish_period_s", 10.0))
        )
        self.stale_after_s = max(0.1, float(rospy.get_param("~stale_after_s", 1.5)))
        self.overlay_threshold = int(rospy.get_param("~overlay_threshold", 50))
        self.max_pointcloud_summary_points = max(
            1, int(rospy.get_param("~max_pointcloud_summary_points", 2000))
        )

        self.topic_manifest_topic = rospy.get_param(
            "~topic_manifest_topic", "/xai/topic_manifest"
        )
        self.planner_snapshot_topic = rospy.get_param(
            "~planner_snapshot_topic", "/xai/planner_snapshot"
        )
        self.event_log_topic = rospy.get_param("~event_log_topic", "/xai/event_log")

        self.topics = {
            "pose": rospy.get_param("~pose_topic", "/lio_localizer/odometry/optimization"),
            "raw_lidar": rospy.get_param("~raw_lidar_topic", "/ouster/points"),
            "obstacle_pointcloud": rospy.get_param(
                "~obstacle_pointcloud_topic", "/planning/linefit_ground/non_ground_cloud"
            ),
            "cmd_vel": rospy.get_param("~cmd_vel_topic", "/cmd_vel"),
            "behavior_cmd": rospy.get_param(
                "~behavior_cmd_topic", "/planning/behavior_cmd"
            ),
            "emergency_stop": rospy.get_param(
                "~emergency_stop_topic", "/planning/emergency_stop"
            ),
            "global_obstacle_caution": rospy.get_param(
                "~global_obstacle_caution_topic", "/planning/global_obstacle_caution"
            ),
            "path_blocked": rospy.get_param("~path_blocked_topic", "/astar/path_blocked"),
            "global_path": rospy.get_param("~global_path_topic", "/astar/path"),
            "candidate_paths": rospy.get_param(
                "~candidate_paths_topic", "/astar/candidate_paths"
            ),
            "tracking_reference": rospy.get_param(
                "~tracking_reference_topic", "/planning/tracking_reference_path"
            ),
            "active_path": rospy.get_param("~active_path_topic", "/planning/active_path"),
            "local_path": rospy.get_param("~local_path_topic", "/planning/local_path"),
            "avoidance_path": rospy.get_param(
                "~avoidance_path_topic", "/planning/avoidance_path"
            ),
            "global_obstacle_overlay": rospy.get_param(
                "~global_obstacle_overlay_topic", "/planning/global_obstacle_overlay"
            ),
            "global_obstacle_overlay_boxes": rospy.get_param(
                "~global_obstacle_overlay_boxes_topic",
                "/planning/global_obstacle_overlay_boxes",
            ),
            "near_field_raw_overlay_hits": rospy.get_param(
                "~near_field_raw_overlay_hits_topic",
                "/planning/near_field_raw_overlay_hits",
            ),
            "near_field_stop_hits": rospy.get_param(
                "~near_field_stop_hits_topic", "/planning/near_field_stop_hits"
            ),
            "near_field_raw_debug_cloud": rospy.get_param(
                "~near_field_raw_debug_cloud_topic", "/planning/near_field_raw_debug_cloud"
            ),
            "near_field_stop_marker": rospy.get_param(
                "~near_field_stop_marker_topic", "/planning/near_field_stop_marker"
            ),
            "drivable_grid": rospy.get_param(
                "~drivable_grid_topic", "/lio_sam/drivable_area/grid"
            ),
            "tracked_objects": rospy.get_param(
                "~tracked_objects_topic", "/perception/tracked_objects"
            ),
        }
        self.watch_raw_lidar = bool(rospy.get_param("~watch_raw_lidar", False))
        self.watch_obstacle_pointcloud = bool(
            rospy.get_param("~watch_obstacle_pointcloud", False)
        )

        self.latest = {}
        self.latest_stamp = {}
        self.path_change_seq = 0
        self.previous_path_points = []
        self.latest_path_change = {
            "changed": False,
            "direction": "unknown",
            "lateral_shift_m": 0.0,
        }
        self.event_seq = 0
        self.last_event_signature = None

        self.manifest_pub = rospy.Publisher(
            self.topic_manifest_topic, String, queue_size=1, latch=True
        )
        self.snapshot_pub = rospy.Publisher(
            self.planner_snapshot_topic, String, queue_size=1
        )
        self.event_pub = rospy.Publisher(
            self.event_log_topic, String, queue_size=50, latch=True
        )

        self._subscribe()
        self._publish_manifest()
        self.publisher_thread = threading.Thread(target=self._publisher_loop)
        self.publisher_thread.daemon = True
        self.publisher_thread.start()
        rospy.loginfo(
            "autonomy_explainability_bridge started | manifest=%s snapshot=%s hz=%.1f",
            self.topic_manifest_topic,
            self.planner_snapshot_topic,
            self.publish_hz,
        )

    def _publisher_loop(self):
        publish_period_s = 1.0 / self.publish_hz
        last_manifest_wall = time.monotonic()
        while not rospy.is_shutdown():
            try:
                self._publish_snapshot(None)
                now_wall = time.monotonic()
                if (now_wall - last_manifest_wall) >= self.manifest_publish_period_s:
                    last_manifest_wall = now_wall
                    self._publish_manifest()
            except Exception as exc:
                rospy.logwarn("autonomy_explainability_bridge publish loop error: %s", str(exc))
            time.sleep(publish_period_s)

    def _subscribe(self):
        rospy.Subscriber(self.topics["pose"], Odometry, self._store_cb("pose"), queue_size=10)
        rospy.Subscriber(
            self.topics["cmd_vel"], Twist, self._store_cb("cmd_vel"), queue_size=10
        )
        rospy.Subscriber(
            self.topics["behavior_cmd"],
            BehaviorCommand,
            self._store_cb("behavior_cmd"),
            queue_size=10,
        )
        rospy.Subscriber(
            self.topics["emergency_stop"],
            Bool,
            self._store_cb("emergency_stop"),
            queue_size=10,
        )
        rospy.Subscriber(
            self.topics["global_obstacle_caution"],
            Bool,
            self._store_cb("global_obstacle_caution"),
            queue_size=10,
        )
        rospy.Subscriber(
            self.topics["path_blocked"],
            Bool,
            self._store_cb("path_blocked"),
            queue_size=10,
        )
        rospy.Subscriber(
            self.topics["global_path"], Path, self._path_cb, queue_size=5
        )
        rospy.Subscriber(
            self.topics["candidate_paths"],
            MarkerArray,
            self._store_cb("candidate_paths"),
            queue_size=5,
        )
        rospy.Subscriber(
            self.topics["tracking_reference"],
            Path,
            self._store_cb("tracking_reference"),
            queue_size=5,
        )
        rospy.Subscriber(
            self.topics["active_path"], Path, self._store_cb("active_path"), queue_size=5
        )
        rospy.Subscriber(
            self.topics["local_path"], Path, self._store_cb("local_path"), queue_size=5
        )
        rospy.Subscriber(
            self.topics["avoidance_path"],
            Path,
            self._store_cb("avoidance_path"),
            queue_size=5,
        )
        rospy.Subscriber(
            self.topics["global_obstacle_overlay"],
            OccupancyGrid,
            self._store_cb("global_obstacle_overlay"),
            queue_size=2,
        )
        rospy.Subscriber(
            self.topics["global_obstacle_overlay_boxes"],
            MarkerArray,
            self._store_cb("global_obstacle_overlay_boxes"),
            queue_size=2,
        )
        rospy.Subscriber(
            self.topics["near_field_raw_overlay_hits"],
            PointCloud2,
            self._store_cb("near_field_raw_overlay_hits"),
            queue_size=2,
        )
        rospy.Subscriber(
            self.topics["near_field_stop_hits"],
            PointCloud2,
            self._store_cb("near_field_stop_hits"),
            queue_size=2,
        )
        rospy.Subscriber(
            self.topics["near_field_raw_debug_cloud"],
            PointCloud2,
            self._store_cb("near_field_raw_debug_cloud"),
            queue_size=1,
        )
        rospy.Subscriber(
            self.topics["drivable_grid"],
            OccupancyGrid,
            self._store_cb("drivable_grid"),
            queue_size=2,
        )
        rospy.Subscriber(
            self.topics["tracked_objects"],
            TrackedObjectArray,
            self._store_cb("tracked_objects"),
            queue_size=5,
        )
        if self.watch_raw_lidar:
            rospy.Subscriber(
                self.topics["raw_lidar"],
                PointCloud2,
                self._store_cb("raw_lidar"),
                queue_size=1,
            )
        if self.watch_obstacle_pointcloud:
            rospy.Subscriber(
                self.topics["obstacle_pointcloud"],
                PointCloud2,
                self._store_cb("obstacle_pointcloud"),
                queue_size=1,
            )

    def _store_cb(self, key):
        def cb(msg):
            self.latest[key] = msg
            self.latest_stamp[key] = self._message_stamp(msg)

        return cb

    def _path_cb(self, msg):
        new_points = self._path_points(msg)
        self.latest_path_change = self._path_change_summary(
            self.previous_path_points, new_points
        )
        if self.latest_path_change["changed"]:
            self.path_change_seq += 1
        self.previous_path_points = new_points
        self.latest["global_path"] = msg
        self.latest_stamp["global_path"] = self._message_stamp(msg)

    def _message_stamp(self, msg):
        header = getattr(msg, "header", None)
        if header is not None:
            stamp = _stamp_to_float(getattr(header, "stamp", None))
            if stamp > 0.0:
                return stamp
        return rospy.Time.now().to_sec()

    def _topic_status(self, key, now_sec):
        stamp = float(self.latest_stamp.get(key, 0.0))
        if stamp <= 0.0:
            return {"topic": self.topics.get(key, ""), "received": False, "age_s": None}
        age = max(0.0, now_sec - stamp)
        return {
            "topic": self.topics.get(key, ""),
            "received": True,
            "stamp": _round(stamp, 3),
            "age_s": _round(age, 3),
            "fresh": age <= self.stale_after_s,
        }

    def _publish_manifest(self, _event=None):
        manifest = {
            "schema": "autonomy_explainability_bridge/TopicManifest@1",
            "generated_by": rospy.get_name(),
            "generated_at": _round(rospy.Time.now().to_sec(), 3),
            "summary_topics": [
                {
                    "topic": self.topic_manifest_topic,
                    "type": "std_msgs/String",
                    "payload": "JSON TopicManifest@1",
                    "use": "새 컴퓨터가 어떤 원본 토픽을 어떤 목적으로 구독해야 하는지 알려주는 latched 계약서",
                },
                {
                    "topic": self.planner_snapshot_topic,
                    "type": "std_msgs/String",
                    "payload": "JSON PlannerSnapshot@1",
                    "use": "주행 판단, 제어 방향, 장애물 evidence의 최신 요약",
                },
                {
                    "topic": self.event_log_topic,
                    "type": "std_msgs/String",
                    "payload": "JSON EventLog@1",
                    "use": "학습/분석용 상태 변화 이벤트. ROS/bag time 기준으로 언제 어떤 판단 변화가 생겼는지 기록",
                },
            ],
            "source_topics": self._manifest_source_topics(),
            "notes": [
                "이 bridge는 read-only 관찰자이며 /cmd_vel, goal, planner decision을 publish하지 않는다.",
                "3D detector는 /ouster/points, /tf, /tf_static, pose topic을 직접 구독하는 것을 권장한다.",
                "custom type 구독이 어려운 원격 서비스는 /xai/planner_snapshot JSON만으로 현재 판단을 읽을 수 있다.",
            ],
        }
        self.manifest_pub.publish(
            String(data=json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        )

    def _manifest_source_topics(self):
        return [
            self._topic_contract(
                "raw_lidar",
                "sensor_msgs/PointCloud2",
                "primary_detection_input",
                "3D detector의 원본 입력. 좌표 해석을 위해 /tf, /tf_static, pose topic과 함께 사용.",
                consumers=["external_3d_detector"],
            ),
            self._topic_contract(
                "pose",
                "nav_msgs/Odometry",
                "localization_context",
                "map/base_link 위치와 yaw. map 기준 장애물 위치와 설명 문장 생성에 필요.",
            ),
            {
                "topic": "/tf",
                "type": "tf2_msgs/TFMessage",
                "role": "transform_context",
                "use": "LiDAR frame, base_link, map 변환",
                "priority": "required",
            },
            {
                "topic": "/tf_static",
                "type": "tf2_msgs/TFMessage",
                "role": "transform_context",
                "use": "os_sensor -> base_link 같은 static transform",
                "priority": "required",
            },
            self._topic_contract(
                "obstacle_pointcloud",
                "sensor_msgs/PointCloud2",
                "planner_obstacle_input",
                "ground/self filtering 이후 planner가 보는 obstacle point cloud.",
            ),
            self._topic_contract(
                "global_obstacle_overlay",
                "nav_msgs/OccupancyGrid",
                "global_planner_obstacle_evidence",
                "A* candidate/global path 평가에 들어가는 map-frame obstacle overlay.",
            ),
            self._topic_contract(
                "global_obstacle_overlay_boxes",
                "visualization_msgs/MarkerArray",
                "planner_obstacle_boxes",
                "overlay에서 생성된 obstacle box. DWA emergency stop 판단에도 사용됨.",
            ),
            self._topic_contract(
                "global_obstacle_caution",
                "std_msgs/Bool",
                "slowdown_trigger",
                "planning_slowdown_manager가 observe/slowdown 판단에 쓰는 caution evidence.",
            ),
            self._topic_contract(
                "near_field_raw_overlay_hits",
                "sensor_msgs/PointCloud2",
                "raw_lidar_near_evidence",
                "raw LiDAR ROI에서 global overlay에 반영된 근거리 hit cloud.",
            ),
            self._topic_contract(
                "near_field_stop_hits",
                "sensor_msgs/PointCloud2",
                "emergency_stop_near_evidence",
                "DWA emergency stop ROI에 잡힌 근거리 hit cloud.",
            ),
            self._topic_contract(
                "near_field_raw_debug_cloud",
                "sensor_msgs/PointCloud2",
                "debug_near_raw_cloud",
                "raw near-field ROI debug cloud. detector 검증용 보조 자료.",
                priority="optional",
            ),
            self._topic_contract(
                "global_path",
                "nav_msgs/Path",
                "selected_global_path",
                "A*가 선택한 현재 global path.",
            ),
            self._topic_contract(
                "candidate_paths",
                "visualization_msgs/MarkerArray",
                "candidate_global_paths",
                "A* candidate path 시각화. path 변경 방향 추정 보조.",
            ),
            self._topic_contract(
                "tracking_reference",
                "nav_msgs/Path",
                "controller_reference_path",
                "DWA가 추종 중인 reference path.",
            ),
            self._topic_contract(
                "path_blocked",
                "std_msgs/Bool",
                "global_path_blocked_state",
                "A*가 현재 path blocked 여부를 판단한 결과.",
            ),
            self._topic_contract(
                "behavior_cmd",
                "dynamic_window_approach/BehaviorCommand",
                "behavior_speed_stop_command",
                "planning_slowdown_manager의 stop/speed_limit/reason.",
            ),
            self._topic_contract(
                "emergency_stop",
                "std_msgs/Bool",
                "emergency_stop_state",
                "DWA 최종 emergency stop 상태.",
            ),
            self._topic_contract(
                "cmd_vel",
                "geometry_msgs/Twist",
                "actual_control_command",
                "실제 제어 출력. 회피 의도라고 부르기보다는 실제 조향/회전 방향으로 해석.",
            ),
            self._topic_contract(
                "drivable_grid",
                "nav_msgs/OccupancyGrid",
                "map_context",
                "drivable area. goal/path/obstacle가 주행 가능 영역 안인지 설명할 때 사용.",
                priority="context",
            ),
            self._topic_contract(
                "tracked_objects",
                "dynamic_window_approach/TrackedObjectArray",
                "legacy_tracked_objects",
                "기존 tracker가 켜진 경우의 object 상태. 기본 run.launch에서는 보통 비활성.",
                priority="optional",
            ),
        ]

    def _topic_contract(self, key, msg_type, role, use, priority="recommended", consumers=None):
        item = {
            "topic": self.topics.get(key, ""),
            "type": msg_type,
            "role": role,
            "use": use,
            "priority": priority,
        }
        if consumers:
            item["consumers"] = consumers
        return item

    def _publish_snapshot(self, _event):
        now_sec = rospy.Time.now().to_sec()
        snapshot = {
            "schema": "autonomy_explainability_bridge/PlannerSnapshot@1",
            "stamp": _round(now_sec, 3),
            "wall_time": _round(time.time(), 3),
            "time_basis": "stamp uses ROS time; with rosbag --clock this matches bag time",
            "node": rospy.get_name(),
            "freshness": {
                key: self._topic_status(key, now_sec)
                for key in (
                    "pose",
                    "cmd_vel",
                    "behavior_cmd",
                    "emergency_stop",
                    "global_obstacle_caution",
                    "path_blocked",
                    "global_path",
                    "candidate_paths",
                    "global_obstacle_overlay",
                    "global_obstacle_overlay_boxes",
                    "near_field_raw_overlay_hits",
                    "near_field_stop_hits",
                    "tracked_objects",
                )
            },
            "robot": self._robot_summary(),
            "decision": self._decision_summary(),
            "control": self._control_summary(),
            "planning": self._planning_summary(),
            "obstacle_evidence": self._obstacle_summary(),
        }
        self.snapshot_pub.publish(
            String(data=json.dumps(snapshot, ensure_ascii=False, sort_keys=True))
        )
        self._maybe_publish_event(snapshot)

    def _maybe_publish_event(self, snapshot):
        signature = self._event_signature(snapshot)
        if signature == self.last_event_signature:
            return

        changed_fields = []
        event_type = "initial_state"
        if self.last_event_signature is not None:
            event_type = "state_change"
            for key in sorted(signature.keys()):
                if signature.get(key) != self.last_event_signature.get(key):
                    changed_fields.append(key)
        else:
            changed_fields = sorted(signature.keys())

        self.last_event_signature = dict(signature)
        self.event_seq += 1
        event = {
            "schema": "autonomy_explainability_bridge/EventLog@1",
            "seq": int(self.event_seq),
            "event_type": event_type,
            "event_label": (
                "initial_state"
                if event_type == "initial_state"
                else self._event_label(snapshot, changed_fields)
            ),
            "stamp": snapshot.get("stamp"),
            "wall_time": _round(time.time(), 3),
            "time_basis": "stamp uses ROS time; with rosbag --clock this matches bag time",
            "changed_fields": changed_fields,
            "signature": signature,
            "source_stamps": self._source_stamps(snapshot),
            "decision": snapshot.get("decision", {}),
            "control": snapshot.get("control", {}),
            "planning": {
                "global_path": snapshot.get("planning", {}).get("global_path", {}),
                "path_change": snapshot.get("planning", {}).get("path_change", {}),
            },
            "obstacle_evidence": {
                "global_overlay_boxes": snapshot.get("obstacle_evidence", {}).get(
                    "global_overlay_boxes", {}
                ),
                "near_field_raw_overlay_hits": snapshot.get("obstacle_evidence", {}).get(
                    "near_field_raw_overlay_hits", {}
                ),
                "near_field_stop_hits": snapshot.get("obstacle_evidence", {}).get(
                    "near_field_stop_hits", {}
                ),
            },
        }
        self.event_pub.publish(
            String(data=json.dumps(event, ensure_ascii=False, sort_keys=True))
        )

    def _event_signature(self, snapshot):
        decision = snapshot.get("decision", {})
        behavior = decision.get("behavior", {})
        control = snapshot.get("control", {})
        planning = snapshot.get("planning", {})
        obstacle = snapshot.get("obstacle_evidence", {})
        path = planning.get("global_path", {})
        path_change = planning.get("path_change", {})
        boxes = obstacle.get("global_overlay_boxes", {})
        raw_hits = obstacle.get("near_field_raw_overlay_hits", {})
        stop_hits = obstacle.get("near_field_stop_hits", {})
        return {
            "behavior_reason": behavior.get("reason"),
            "behavior_stop": behavior.get("stop"),
            "speed_limit_mps": behavior.get("speed_limit_mps"),
            "emergency_stop": decision.get("emergency_stop", {}).get("value"),
            "global_obstacle_caution": decision.get("global_obstacle_caution", {}).get(
                "value"
            ),
            "path_blocked": decision.get("path_blocked", {}).get("value"),
            "motion_state": control.get("motion_state"),
            "steering_direction": control.get("steering_direction"),
            "global_path_received": path.get("received"),
            "global_path_points": path.get("points"),
            "path_change_seq": path_change.get("seq"),
            "path_change_direction": path_change.get("latest", {}).get("direction"),
            "overlay_box_count": boxes.get("box_count"),
            "near_raw_hit_present": (raw_hits.get("reported_points") or 0) > 0,
            "near_stop_hit_present": (stop_hits.get("reported_points") or 0) > 0,
        }

    def _event_label(self, snapshot, changed_fields):
        decision = snapshot.get("decision", {})
        behavior = decision.get("behavior", {})
        if decision.get("emergency_stop", {}).get("value") is True:
            return "emergency_stop"
        if behavior.get("stop") is True:
            return "behavior_stop"
        if decision.get("global_obstacle_caution", {}).get("value") is True:
            return "global_obstacle_caution"
        if decision.get("path_blocked", {}).get("value") is True:
            return "path_blocked"
        if "path_change_seq" in changed_fields:
            return "path_update"
        if "motion_state" in changed_fields or "steering_direction" in changed_fields:
            return "control_update"
        if behavior.get("reason") and behavior.get("reason") != "clear":
            return "behavior_reason"
        return "state_update"

    def _source_stamps(self, snapshot):
        out = {}
        for key, status in snapshot.get("freshness", {}).items():
            if isinstance(status, dict) and status.get("received"):
                out[key] = status.get("stamp")
        return out

    def _robot_summary(self):
        msg = self.latest.get("pose")
        if msg is None:
            return {"received": False, "topic": self.topics["pose"]}
        pose = msg.pose.pose
        twist = msg.twist.twist
        return {
            "received": True,
            "topic": self.topics["pose"],
            "frame_id": getattr(msg.header, "frame_id", ""),
            "child_frame_id": getattr(msg, "child_frame_id", ""),
            "position": {
                "x": _round(pose.position.x),
                "y": _round(pose.position.y),
                "z": _round(pose.position.z),
            },
            "yaw_rad": _round(_quat_to_yaw(pose.orientation), 4),
            "speed_mps": _round(
                math.hypot(float(twist.linear.x), float(twist.linear.y))
            ),
            "linear_velocity": {
                "x": _round(twist.linear.x),
                "y": _round(twist.linear.y),
                "z": _round(twist.linear.z),
            },
            "angular_velocity": {
                "x": _round(twist.angular.x),
                "y": _round(twist.angular.y),
                "z": _round(twist.angular.z),
            },
        }

    def _decision_summary(self):
        behavior = self.latest.get("behavior_cmd")
        return {
            "behavior": {
                "received": behavior is not None,
                "stop": bool(getattr(behavior, "stop", False)) if behavior else None,
                "speed_limit_mps": _round(getattr(behavior, "speed_limit", None)),
                "reason": getattr(behavior, "reason", None) if behavior else None,
            },
            "emergency_stop": self._bool_value("emergency_stop"),
            "global_obstacle_caution": self._bool_value("global_obstacle_caution"),
            "path_blocked": self._bool_value("path_blocked"),
        }

    def _bool_value(self, key):
        msg = self.latest.get(key)
        if msg is None:
            return {"received": False, "value": None, "topic": self.topics.get(key, "")}
        return {"received": True, "value": bool(msg.data), "topic": self.topics.get(key, "")}

    def _control_summary(self):
        msg = self.latest.get("cmd_vel")
        if msg is None:
            return {
                "received": False,
                "topic": self.topics["cmd_vel"],
                "steering_direction": "unknown",
                "motion_state": "unknown",
            }
        linear_x = float(msg.linear.x)
        angular_z = float(msg.angular.z)
        abs_linear = abs(linear_x)
        abs_angular = abs(angular_z)
        if abs_linear < 0.01 and abs_angular < 0.03:
            motion_state = "stopped"
        elif abs_linear < 0.01:
            motion_state = "rotate_left" if angular_z > 0.0 else "rotate_right"
        elif linear_x < -0.01:
            motion_state = "reverse"
        elif angular_z > 0.05:
            motion_state = "forward_left"
        elif angular_z < -0.05:
            motion_state = "forward_right"
        else:
            motion_state = "forward_straight"

        if angular_z > 0.05:
            steering_direction = "left"
        elif angular_z < -0.05:
            steering_direction = "right"
        else:
            steering_direction = "straight_or_stop"

        return {
            "received": True,
            "topic": self.topics["cmd_vel"],
            "linear_x_mps": _round(linear_x),
            "angular_z_radps": _round(angular_z),
            "steering_direction": steering_direction,
            "motion_state": motion_state,
            "note": "actual control output; do not treat as obstacle avoidance intent without obstacle/path-change evidence",
        }

    def _planning_summary(self):
        path = self.latest.get("global_path")
        tracking = self.latest.get("tracking_reference")
        candidates = self.latest.get("candidate_paths")
        active_path = self.latest.get("active_path")
        local_path = self.latest.get("local_path")
        avoidance_path = self.latest.get("avoidance_path")
        return {
            "global_path": self._path_summary(path, self.topics["global_path"]),
            "tracking_reference": self._path_summary(
                tracking, self.topics["tracking_reference"]
            ),
            "active_path": self._path_summary(active_path, self.topics["active_path"]),
            "local_path": self._path_summary(local_path, self.topics["local_path"]),
            "avoidance_path": self._path_summary(
                avoidance_path, self.topics["avoidance_path"]
            ),
            "candidate_paths": self._candidate_summary(candidates),
            "path_change": {
                "seq": int(self.path_change_seq),
                "latest": self.latest_path_change,
                "direction_semantics": "left/right is lateral shift of selected global path in robot heading frame, not a direct actuator command",
            },
        }

    def _path_summary(self, msg, topic):
        if msg is None:
            return {"received": False, "topic": topic}
        points = self._path_points(msg)
        length = 0.0
        for i in range(1, len(points)):
            length += _point_distance_xy(points[i - 1], points[i])
        out = {
            "received": True,
            "topic": topic,
            "frame_id": getattr(msg.header, "frame_id", ""),
            "points": len(points),
            "length_m": _round(length),
        }
        if points:
            out["start"] = {"x": _round(points[0][0]), "y": _round(points[0][1])}
            out["end"] = {"x": _round(points[-1][0]), "y": _round(points[-1][1])}
        return out

    def _path_points(self, msg):
        if msg is None:
            return []
        points = []
        for pose_stamped in getattr(msg, "poses", []):
            p = pose_stamped.pose.position
            points.append((float(p.x), float(p.y)))
        return points

    def _path_change_summary(self, old_points, new_points):
        if not old_points or not new_points:
            return {"changed": False, "direction": "unknown", "lateral_shift_m": 0.0}
        old_anchor = self._lookahead_point(old_points)
        new_anchor = self._lookahead_point(new_points)
        distance = _point_distance_xy(old_anchor, new_anchor)
        if distance < 0.15:
            return {
                "changed": False,
                "direction": "straight_or_same",
                "lateral_shift_m": _round(distance),
            }

        robot = self.latest.get("pose")
        if robot is None:
            direction = "unknown"
            lateral = 0.0
        else:
            pose = robot.pose.pose
            yaw = _quat_to_yaw(pose.orientation)
            left_x = -math.sin(yaw)
            left_y = math.cos(yaw)
            dx = new_anchor[0] - old_anchor[0]
            dy = new_anchor[1] - old_anchor[1]
            lateral = (dx * left_x) + (dy * left_y)
            if lateral > 0.15:
                direction = "left"
            elif lateral < -0.15:
                direction = "right"
            else:
                direction = "forward_or_back"
        return {
            "changed": True,
            "direction": direction,
            "lateral_shift_m": _round(lateral),
            "anchor_shift_m": _round(distance),
        }

    def _lookahead_point(self, points):
        if not points:
            return (0.0, 0.0)
        if len(points) == 1:
            return points[0]
        target_distance = 2.0
        accum = 0.0
        for i in range(1, len(points)):
            segment = _point_distance_xy(points[i - 1], points[i])
            accum += segment
            if accum >= target_distance:
                return points[i]
        return points[-1]

    def _candidate_summary(self, msg):
        if msg is None:
            return {"received": False, "topic": self.topics["candidate_paths"]}
        add_markers = [
            m
            for m in msg.markers
            if int(m.action) == Marker.ADD
            and (not m.ns or "candidate" in str(m.ns) or "astar" in str(m.ns))
        ]
        return {
            "received": True,
            "topic": self.topics["candidate_paths"],
            "marker_count": len(msg.markers),
            "candidate_marker_count": len(add_markers),
        }

    def _obstacle_summary(self):
        return {
            "global_overlay": self._overlay_summary(),
            "global_overlay_boxes": self._box_summary(),
            "near_field_raw_overlay_hits": self._pointcloud_summary(
                "near_field_raw_overlay_hits"
            ),
            "near_field_stop_hits": self._pointcloud_summary("near_field_stop_hits"),
            "near_field_raw_debug_cloud": self._pointcloud_summary(
                "near_field_raw_debug_cloud", read_points=False
            ),
            "tracked_objects": self._tracked_object_summary(),
        }

    def _overlay_summary(self):
        msg = self.latest.get("global_obstacle_overlay")
        if msg is None:
            return {"received": False, "topic": self.topics["global_obstacle_overlay"]}
        total = len(msg.data)
        occupied = 0
        if total:
            occupied = sum(1 for v in msg.data if int(v) >= self.overlay_threshold)
        return {
            "received": True,
            "topic": self.topics["global_obstacle_overlay"],
            "frame_id": getattr(msg.header, "frame_id", ""),
            "resolution_m": _round(msg.info.resolution),
            "width": int(msg.info.width),
            "height": int(msg.info.height),
            "occupied_cells_ge_threshold": int(occupied),
            "threshold": int(self.overlay_threshold),
        }

    def _box_summary(self):
        msg = self.latest.get("global_obstacle_overlay_boxes")
        if msg is None:
            return {
                "received": False,
                "topic": self.topics["global_obstacle_overlay_boxes"],
            }
        boxes = [
            marker
            for marker in msg.markers
            if int(marker.action) == Marker.ADD and int(marker.type) == Marker.CUBE
        ]
        nearest = None
        robot_xy = self._robot_xy()
        for marker in boxes:
            p = marker.pose.position
            distance = None
            bearing = None
            if robot_xy is not None:
                dx = float(p.x) - robot_xy[0]
                dy = float(p.y) - robot_xy[1]
                distance = math.hypot(dx, dy)
                yaw = self._robot_yaw()
                if yaw is not None:
                    bearing = math.atan2(dy, dx) - yaw
                    bearing = math.atan2(math.sin(bearing), math.cos(bearing))
            item = {
                "id": int(marker.id),
                "frame_id": marker.header.frame_id,
                "position": {"x": _round(p.x), "y": _round(p.y), "z": _round(p.z)},
                "size": {
                    "x": _round(marker.scale.x),
                    "y": _round(marker.scale.y),
                    "z": _round(marker.scale.z),
                },
                "distance_from_robot_m": _round(distance),
                "bearing_from_robot_rad": _round(bearing, 4),
                "locked_color_hint": bool(marker.color.g > 0.4),
            }
            if nearest is None:
                nearest = item
                continue
            item_dist = item["distance_from_robot_m"]
            nearest_dist = nearest["distance_from_robot_m"]
            if item_dist is not None and (
                nearest_dist is None or item_dist < nearest_dist
            ):
                nearest = item
        return {
            "received": True,
            "topic": self.topics["global_obstacle_overlay_boxes"],
            "box_count": len(boxes),
            "nearest_box": nearest,
        }

    def _pointcloud_summary(self, key, read_points=True):
        msg = self.latest.get(key)
        if msg is None:
            return {"received": False, "topic": self.topics.get(key, "")}
        reported = int(msg.width) * max(1, int(msg.height))
        out = {
            "received": True,
            "topic": self.topics.get(key, ""),
            "frame_id": msg.header.frame_id,
            "reported_points": reported,
        }
        if not read_points or reported <= 0:
            return out

        count = 0
        sx = sy = sz = 0.0
        min_x = None
        min_range = None
        for point in point_cloud2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True
        ):
            x = float(point[0])
            y = float(point[1])
            z = float(point[2])
            count += 1
            sx += x
            sy += y
            sz += z
            rng = math.sqrt((x * x) + (y * y) + (z * z))
            min_x = x if min_x is None else min(min_x, x)
            min_range = rng if min_range is None else min(min_range, rng)
            if count >= self.max_pointcloud_summary_points:
                break
        out["sampled_points"] = count
        out["min_x_m"] = _round(min_x)
        out["min_range_m"] = _round(min_range)
        if count > 0:
            out["sample_centroid"] = {
                "x": _round(sx / count),
                "y": _round(sy / count),
                "z": _round(sz / count),
            }
        return out

    def _tracked_object_summary(self):
        msg = self.latest.get("tracked_objects")
        if msg is None:
            return {"received": False, "topic": self.topics["tracked_objects"]}
        labels = {}
        for obj in msg.objects:
            label = str(obj.label) if obj.label else "unknown"
            labels[label] = labels.get(label, 0) + 1
        return {
            "received": True,
            "topic": self.topics["tracked_objects"],
            "frame_id": msg.header.frame_id,
            "object_count": len(msg.objects),
            "labels": labels,
        }

    def _robot_xy(self):
        msg = self.latest.get("pose")
        if msg is None:
            return None
        p = msg.pose.pose.position
        return float(p.x), float(p.y)

    def _robot_yaw(self):
        msg = self.latest.get("pose")
        if msg is None:
            return None
        return _quat_to_yaw(msg.pose.pose.orientation)


def main():
    rospy.init_node("autonomy_explainability_bridge")
    ExplainabilityTopicBridge()
    rospy.spin()


if __name__ == "__main__":
    main()
