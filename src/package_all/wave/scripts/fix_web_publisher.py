#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fix_web_publisher.py  (ROS1, Python3)
- GPS 있으면 패스스루, 없으면 로컬라이제이션 포즈로 위경도 추정해 'fix_web' 발행
- pose_topic은 다음 중 하나를 지원: nav_msgs/Path, geometry_msgs/PoseStamped,
  nav_msgs/Odometry, geometry_msgs/PoseWithCovarianceStamped
- 같은 토픽에 다중 타입 구독하지 않도록 '자동 타입 감지'로 딱 한 타입만 구독

Params
------
~gps_in_topic:             "/gps/fix"
~pose_topic:               "/lio_localizer/odometry/optimization_path"
~pose_topic_type:          ""        # "", "Path", "PoseStamped", "Odometry", "PoseWithCovarianceStamped"
~pose_topic_wait_sec:      10.0      # 자동 감지 최대 대기 시간(초)
~fix_out_topic:            "fix_web"
~mode_out_topic:           "fix_web/mode"
~gps_fresh_timeout_sec:     1.5
~init_lat:                  37.566536
~init_lon:                 126.977966
~init_alt:                   0.0
~use_alt_from_pose:          false
~map_yaw_deg:                0.0
~force_fallback:             false
~publish_init_if_no_pose:    true
"""

import math
import time
import threading

import rospy
import rostopic
from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import String
from sensor_msgs.msg import NavSatFix, NavSatStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path


def deg2rad(d): return d * math.pi / 180.0
def rad2deg(r): return r * 180.0 / math.pi


class FixWebPublisher:
    def __init__(self):
        # ---------- Params ----------
        self.gps_in_topic = rospy.get_param("~gps_in_topic", "/gps/fix")
        self.pose_topic = rospy.get_param("~pose_topic", "/lio_localizer/odometry")
        self.pose_topic_type = rospy.get_param("~pose_topic_type", "").strip()  # optional override
        self.pose_topic_wait_sec = float(rospy.get_param("~pose_topic_wait_sec", 10.0))

        self.fix_out_topic = rospy.get_param("~fix_out_topic", "fix_web")
        self.mode_out_topic = rospy.get_param("~mode_out_topic", "fix_web/mode")

        self.gps_fresh_timeout = float(rospy.get_param("~gps_fresh_timeout_sec", 1.5))
        self.init_lat = float(rospy.get_param("~init_lat", float('nan')))
        self.init_lon = float(rospy.get_param("~init_lon", float('nan')))
        self.init_alt = float(rospy.get_param("~init_alt", 0.0))
        self.use_alt_from_pose = bool(rospy.get_param("~use_alt_from_pose", False))
        self.map_yaw_deg = float(rospy.get_param("~map_yaw_deg", 0.0))
        self.force_fallback = bool(rospy.get_param("~force_fallback", False))
        self.publish_init_if_no_pose = bool(rospy.get_param("~publish_init_if_no_pose", True))

        # ---------- Internal state ----------
        self.lock = threading.Lock()
        self.last_gps_msg = None
        self.last_gps_stamp = None

        self.have_pose0 = False
        self.pose0_x = 0.0
        self.pose0_y = 0.0
        self.pose0_z = 0.0

        self.last_pose_x = None
        self.last_pose_y = None
        self.last_pose_z = None
        self.last_pose_stamp = None

        # map -> ENU 회전행렬(좌표만)
        yaw = deg2rad(self.map_yaw_deg)
        cy, sy = math.cos(yaw), math.sin(yaw)
        self.Rm2enu = ((cy, -sy), (sy, cy))

        # ---------- Pubs/Subs ----------
        self.fix_pub = rospy.Publisher(self.fix_out_topic, NavSatFix, queue_size=10, latch=True)
        self.mode_pub = rospy.Publisher(self.mode_out_topic, String, queue_size=10, latch=True)
        rospy.Subscriber(self.gps_in_topic, NavSatFix, self.gps_cb, queue_size=10)

        # pose 구독자는 '단 한 개'만 생성 (타입 자동 감지 또는 강제 지정)
        self._setup_pose_subscriber()

        # ---------- Services ----------
        rospy.Service("~fix_web_reset", Empty, self.handle_reset)

        rospy.loginfo("[fix_web_publisher] started. gps_in=%s pose=%s out=%s",
                      self.gps_in_topic, self.pose_topic, self.fix_out_topic)
        if math.isnan(self.init_lat) or math.isnan(self.init_lon):
            rospy.logwarn("[fix_web_publisher] init_lat/lon 미설정: GPS 미수신시 fallback 발행 불가")

        # 시작 직후, pose가 없어도 1회 발행
        if self.publish_init_if_no_pose:
            rospy.Timer(rospy.Duration(1.0), self._oneshot_publish_init, oneshot=True)

    # ---------- Pose subscriber setup (single-type only) ----------
    def _setup_pose_subscriber(self):
        # 강제 타입 지정이 있으면 우선
        forced = self.pose_topic_type.lower()
        if forced in ("path", "odometry", "posestamped", "posewithcovariancestamped"):
            self._subscribe_by_name(forced)
            return

        # 자동 감지
        deadline = time.time() + max(0.0, self.pose_topic_wait_sec)
        clazz = None
        while not rospy.is_shutdown() and time.time() < deadline and clazz is None:
            try:
                clazz, _, _ = rostopic.get_topic_class(self.pose_topic, blocking=False)
            except Exception:
                clazz = None
            if clazz is None:
                rospy.sleep(0.2)

        if clazz is None:
            rospy.logwarn("[fix_web_publisher] pose_topic 타입 자동 감지 실패: %s (대기 %.1fs). "
                          "임시로 Path로 시도합니다. 필요시 ~pose_topic_type으로 강제 지정하세요.",
                          self.pose_topic, self.pose_topic_wait_sec)
            self._subscribe_by_name("path")
            return

        # 감지된 타입에 따라 구독
        if clazz == Path:
            self._subscribe(Path, self.path_cb, "Path")
        elif clazz == Odometry:
            self._subscribe(Odometry, self.odom_cb, "Odometry")
        elif clazz == PoseStamped:
            self._subscribe(PoseStamped, self.pose_stamped_cb, "PoseStamped")
        elif clazz == PoseWithCovarianceStamped:
            self._subscribe(PoseWithCovarianceStamped, self.pwc_cb, "PoseWithCovarianceStamped")
        else:
            rospy.logwarn("[fix_web_publisher] 미지원 타입 %s 감지됨. 임시로 Path로 시도합니다.", clazz)
            self._subscribe(Path, self.path_cb, "Path")

    def _subscribe_by_name(self, name: str):
        name = name.lower()
        if name == "path":
            self._subscribe(Path, self.path_cb, "Path(Forced)")
        elif name == "odometry":
            self._subscribe(Odometry, self.odom_cb, "Odometry(Forced)")
        elif name == "posestamped":
            self._subscribe(PoseStamped, self.pose_stamped_cb, "PoseStamped(Forced)")
        elif name == "posewithcovariancestamped":
            self._subscribe(PoseWithCovarianceStamped, self.pwc_cb, "PoseWithCovarianceStamped(Forced)")
        else:
            rospy.logwarn("[fix_web_publisher] 알 수 없는 강제 타입 %s → Path로 시도", name)
            self._subscribe(Path, self.path_cb, "Path(Forced-Unknown)")

    def _subscribe(self, cls, cb, label: str):
        rospy.Subscriber(self.pose_topic, cls, cb, queue_size=50)
        rospy.loginfo("[fix_web_publisher] pose_topic 구독 타입: %s", label)

    # ---------- Callbacks ----------
    def gps_cb(self, msg: NavSatFix):
        with self.lock:
            self.last_gps_msg = msg
            self.last_gps_stamp = rospy.Time.now()
        self.publish_fix()

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self._update_pose_and_publish(p.x, p.y, p.z, msg.header.stamp)

    def pose_stamped_cb(self, msg: PoseStamped):
        p = msg.pose.position
        self._update_pose_and_publish(p.x, p.y, p.z, msg.header.stamp)

    def pwc_cb(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose.position
        self._update_pose_and_publish(p.x, p.y, p.z, msg.header.stamp)

    def path_cb(self, msg: Path):
        if not msg.poses:
            return
        last = msg.poses[-1]
        p = last.pose.position
        stamp = last.header.stamp if last.header.stamp != rospy.Time(0) else msg.header.stamp
        self._update_pose_and_publish(p.x, p.y, p.z, stamp)

    def _update_pose_and_publish(self, x, y, z, stamp):
        with self.lock:
            if not self.have_pose0:
                self.pose0_x, self.pose0_y, self.pose0_z = x, y, z
                self.have_pose0 = True
                rospy.loginfo("[fix_web_publisher] 기준 포즈 고정 p0=(%.3f, %.3f, %.3f)", x, y, z)
            self.last_pose_x, self.last_pose_y, self.last_pose_z = x, y, z
            self.last_pose_stamp = stamp
        self.publish_fix()

    # ---------- Service ----------
    def handle_reset(self, _req):
        with self.lock:
            self.have_pose0 = False
            self.pose0_x = self.pose0_y = self.pose0_z = 0.0
            self.last_pose_x = self.last_pose_y = self.last_pose_z = None
            self.last_pose_stamp = None
        rospy.logwarn("[fix_web_publisher] 기준 포즈와 최근 pose 리셋됨. 다음 pose 수신 시 p0 재설정")
        return EmptyResponse()

    # ---------- One-shot initial publish ----------
    def _oneshot_publish_init(self, _evt):
        with self.lock:
            need = (self.last_gps_msg is None) and (not self.have_pose0)
        if need:
            self.publish_fix(force_init_only=True)

    # ---------- Core publishing ----------
    def publish_fix(self, force_init_only: bool = False):
        mode = None
        out_msg = None
        now = rospy.Time.now()

        with self.lock:
            gps_fresh = False
            if (self.last_gps_msg is not None) and (self.last_gps_stamp is not None):
                gps_fresh = (now - self.last_gps_stamp).to_sec() <= self.gps_fresh_timeout

            if (not self.force_fallback) and (not force_init_only) and gps_fresh:
                mode = "GPS_PASSTHROUGH"
                out_msg = self._clone_navsat(self.last_gps_msg)
            else:
                if math.isnan(self.init_lat) or math.isnan(self.init_lon):
                    rospy.logwarn_throttle(5.0, "[fix_web_publisher] init_lat/lon 미설정: fallback 불가")
                    return

                mode = "LOC_FALLBACK"

                if force_init_only or (not self.have_pose0) or (self.last_pose_x is None):
                    out_msg = NavSatFix()
                    out_msg.header.stamp = now
                    out_msg.header.frame_id = "wgs84"
                    out_msg.status.status = NavSatStatus.STATUS_FIX
                    out_msg.status.service = NavSatStatus.SERVICE_GPS
                    out_msg.latitude = self.init_lat
                    out_msg.longitude = self.init_lon
                    out_msg.altitude = self.init_alt
                    out_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
                else:
                    lat, lon, alt, stamp = self._compute_latlon_from_pose()
                    out_msg = NavSatFix()
                    out_msg.header.stamp = stamp if stamp and stamp != rospy.Time(0) else now
                    out_msg.header.frame_id = "wgs84"
                    out_msg.status.status = NavSatStatus.STATUS_FIX
                    out_msg.status.service = NavSatStatus.SERVICE_GPS
                    out_msg.latitude = lat
                    out_msg.longitude = lon
                    out_msg.altitude = alt
                    out_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN

        if out_msg is not None:
            self.fix_pub.publish(out_msg)
        if mode is not None:
            self.mode_pub.publish(String(data=mode))

    # ---------- Helpers ----------
    def _clone_navsat(self, msg_in: NavSatFix) -> NavSatFix:
        m = NavSatFix()
        m.header = msg_in.header
        m.status = msg_in.status
        m.latitude = msg_in.latitude
        m.longitude = msg_in.longitude
        m.altitude = msg_in.altitude
        m.position_covariance = msg_in.position_covariance
        m.position_covariance_type = msg_in.position_covariance_type
        return m

    def _compute_latlon_from_pose(self):
        x = self.last_pose_x
        y = self.last_pose_y
        z = self.last_pose_z
        stamp = self.last_pose_stamp

        dx = x - self.pose0_x
        dy = y - self.pose0_y

        # map → ENU
        e = self.Rm2enu[0][0] * dx + self.Rm2enu[0][1] * dy
        n = self.Rm2enu[1][0] * dx + self.Rm2enu[1][1] * dy

        R = 6378137.0
        lat0_rad = deg2rad(self.init_lat)
        cos_lat0 = max(1e-9, math.cos(lat0_rad))

        dlat_deg = rad2deg(n / R)
        dlon_deg = rad2deg(e / (R * cos_lat0))

        lat = self.init_lat + dlat_deg
        lon = self.init_lon + dlon_deg

        if self.use_alt_from_pose and (z is not None):
            alt = self.init_alt + (z - self.pose0_z)
        else:
            alt = self.init_alt

        return lat, lon, alt, stamp


def main():
    rospy.init_node("fix_web_publisher")
    _ = FixWebPublisher()
    rospy.spin()


if __name__ == "__main__":
    main()
