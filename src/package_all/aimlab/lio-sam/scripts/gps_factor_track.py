#!/usr/bin/env python3
import rosbag
import csv
import math

# 수정: 파일 경로
bag_path = "/home/aimlab/waveai_ws/bags/2025-05-08-16-33-39.bag"
output_csv = "/home/aimlab/waveai_ws/bags/likely_used_gps_poses.csv"

# 파라미터
TIME_SYNC_THRESHOLD = 0.2  # pose - gps 시간 허용 범위 (초)
GPS_COV_THRESHOLD = 2.0    # GPS covariance 허용 기준
MIN_GPS_DISTANCE = 5.0     # 최소 거리 간격 (SLAM 좌표 기준, meter)

def distance(p1, p2):
    return math.sqrt((p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

# 데이터 저장 리스트
slam_poses = []
gps_msgs = []

# bag 파일에서 pose, gps 데이터 추출
with rosbag.Bag(bag_path, "r") as bag:
    for topic, msg, t in bag.read_messages():
        if topic == "/lio_sam/mapping/pose":
            slam_poses.append((t.to_sec(), msg.pose.position.x, msg.pose.position.y, msg.pose.position.z))
        elif topic == "/smc_2000/fix":
            cov_x = msg.position_covariance[0]
            cov_y = msg.position_covariance[4]
            cov_z = msg.position_covariance[8]
            if cov_x < GPS_COV_THRESHOLD and cov_y < GPS_COV_THRESHOLD and cov_z < GPS_COV_THRESHOLD:
                gps_msgs.append((t.to_sec(), msg.latitude, msg.longitude, msg.altitude))

# pose - gps 매칭 및 거리 조건 필터링
matched_gps = []
last_gps = None

for pose_time, x, y, z in slam_poses:
    for gps_time, lat, lon, alt in gps_msgs:
        if abs(gps_time - pose_time) < TIME_SYNC_THRESHOLD:
            if last_gps is None or distance(last_gps, (gps_time, x, y)) > MIN_GPS_DISTANCE:
                matched_gps.append((gps_time, lat, lon, alt))
                last_gps = (gps_time, x, y)
            break

# 결과 저장
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "latitude", "longitude", "altitude"])
    writer.writerows(matched_gps)

print(f"[✓] Saved GPS-used pose candidates to: {output_csv}")
