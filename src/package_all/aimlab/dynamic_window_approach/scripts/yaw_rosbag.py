import rosbag
import matplotlib.pyplot as plt
import numpy as np

def extract_yaw_rate_from_cmd_vel(bag_file):
    bag = rosbag.Bag(bag_file)
    timestamps = []
    yaw_rates = []
    for topic, msg, t in bag.read_messages(topics=['/cmd_vel']):
        timestamps.append(t.to_sec())
        yaw_rates.append(msg.angular.z)
    bag.close()
    return np.array(timestamps), np.array(yaw_rates)

# 두 개의 bag 파일에서 데이터 추출
ts_before, yaw_before = extract_yaw_rate_from_cmd_vel('/home/aimlab/2025-05-26-22-26-05.bag')
ts_after, yaw_after = extract_yaw_rate_from_cmd_vel('/home/aimlab/2025-05-26-22-57-30.bag')

# 시간 정렬 보정 (시작 시간 기준 정렬)
ts_before -= ts_before[0]
ts_after -= ts_after[0]

# 8초 이후 데이터만 필터링
time_threshold = 8.5
mask_ts_before = ts_before >= time_threshold
ts_before = ts_before[mask_ts_before]
yaw_before = yaw_before[mask_ts_before]

mask_ts_after = ts_after >= time_threshold
ts_after = ts_after[mask_ts_after]
yaw_after = yaw_after[mask_ts_after]

# ------------------------------
# 📊 1. 시각화: 시간에 따른 yaw rate
plt.plot(ts_before, yaw_before, label='Before Spline', alpha=0.7)
plt.plot(ts_after, yaw_after, label='After Spline', alpha=0.7)
plt.xlabel("Time [s]")
plt.ylabel("Yaw Rate [rad/s]")
plt.title("Yaw Rate over Time")
plt.grid()
plt.legend()
plt.show()

# ------------------------------
# 📈 2. 평균, 표준편차, 최대값 비교
def stats(label, data):
    print(f"{label}: mean={np.mean(data):.3f}, std={np.std(data):.3f}, max={np.max(np.abs(data)):.3f}")

stats("Before Spline", yaw_before)
stats("After Spline", yaw_after)

# ------------------------------
# 🚨 3. 스파이크(급회전) 탐지: |yaw_rate| > 1.0 rad/s
def count_spikes(data, threshold=1.0):
    return np.sum(np.abs(data) > threshold)

print("Spike Count (|yaw_rate| > 1.0 rad/s)")
print("Before Spline:", count_spikes(yaw_before))
print("After Spline:", count_spikes(yaw_after))

# ------------------------------
# 🎯 4. 커브 구간만 분석 (곡률 임계값 이상만)
CURVE_THRESHOLD = 0.1  # rad/s 이상이면 커브라고 가정

# 마스크 생성
mask_before = np.abs(yaw_before) > CURVE_THRESHOLD
mask_after = np.abs(yaw_after) > CURVE_THRESHOLD

# 커브 구간만 추출
curve_yaw_before = yaw_before[mask_before]
curve_yaw_after = yaw_after[mask_after]

# 통계 출력
def print_stats(label, data):
    print(f"{label}:")
    if len(data) == 0:
        print("  ⚠️ No curve data")
    else:
        print(f"  mean={np.mean(data):.4f}, std={np.std(data):.4f}, max={np.max(np.abs(data)):.4f}, count={len(data)}")

print_stats("Before Spline (curve only)", curve_yaw_before)
print_stats("After Spline (curve only)", curve_yaw_after)

# 시각화
plt.hist(curve_yaw_before, bins=40, alpha=0.5, label='Before (Curves)')
plt.hist(curve_yaw_after, bins=40, alpha=0.5, label='After (Curves)')
plt.legend()
plt.xlabel("Yaw Rate [rad/s]")
plt.ylabel("Frequency")
plt.title("Yaw Rate Distribution (Curve Sections Only, After 8s)")
plt.grid()
plt.show()