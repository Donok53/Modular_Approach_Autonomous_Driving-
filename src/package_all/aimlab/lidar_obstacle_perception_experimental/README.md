# LiDAR Obstacle Perception Experimental

실험용 LiDAR+IMU 기반 장애물 인식 패키지입니다. 기존 자율주행 스택을 건드리지 않고 다음 단계를 독립적으로 확인하는 용도입니다.

- `ground removal`
- `정적 포인트맵 누적`
- `동적 장애물 추적`
- `로컬 장애물 occupancy grid`
- `RViz 시각화`

## 입력 토픽

- `pointcloud_topic`
  - 기본값: `/lio_localizer/localization/cloud_deskewed`
- `imu_topic`
  - 기본값: `/imu_correct`
- `odom_topic`
  - 기본값: `/lio_localizer/odometry/optimization`

## 출력 토픽

- `/experimental/lidar_obstacle_perception/ground_cloud`
- `/experimental/lidar_obstacle_perception/non_ground_cloud`
- `/experimental/lidar_obstacle_perception/static_cloud`
- `/experimental/lidar_obstacle_perception/dynamic_cloud`
- `/experimental/lidar_obstacle_perception/dynamic_markers`
- `/experimental/lidar_obstacle_perception/local_obstacle_grid`

## 실행

```bash
roslaunch lidar_obstacle_perception_experimental lidar_obstacle_perception_experimental.launch
```

## Ground Removal Debug

정적/동적 분리를 잠시 무시하고 raw LiDAR 기준으로 ground removal만 확인하려면 아래 launch를 쓰면 됩니다.

```bash
roslaunch lidar_obstacle_perception_experimental lidar_ground_removal_debug.launch
```

`monitoring_full.bag` 기준 예시는 아래처럼 쓰면 됩니다.

```bash
roslaunch lidar_obstacle_perception_experimental lidar_ground_removal_debug.launch \
  pointcloud_topic:=/ouster/points \
  imu_topic:=/imu/data
```

이 모드에서는 RViz에 아래 3개만 보입니다.

- `Raw LiDAR`
- `Ground Cloud`
- `Non Ground Cloud`

이 debug launch는 `ground_only_mode=true`로 실행되어, 클러스터링/트래킹/static map/grid 생성을 건너뜁니다.
그래서 raw 대비 `Ground/Non Ground` 지연을 줄이고 ground removal 파라미터만 보기에 적합합니다.

## Linefit Ground Removal Debug

검증된 C++ ground segmentation 구현으로 같은 구성을 보고 싶으면 아래 launch를 쓰면 됩니다.

```bash
roslaunch lidar_obstacle_perception_experimental linefit_ground_removal_debug.launch \
  pointcloud_topic:=/ouster/points \
  imu_topic:=/imu/data
```

이 launch는:

- `linefit_ground_segmentation_ros`를 사용해 ground/non-ground를 분리하고
- `imu_gravity_frame_broadcaster.py`로 IMU 기반 gravity-aligned frame을 제공하며
- RViz에서 `Raw LiDAR / Ground / Non Ground`를 동일한 색 구성으로 보여줍니다.

참고 구현:

- linefit_ground_segmentation: https://github.com/lorenwel/linefit_ground_segmentation

차체가 `non-ground`로 계속 보이는지 확인하려면 아래 launch를 별도로 쓰면 됩니다.
이 launch는 주행용 `run.launch`에는 영향을 주지 않고, 필터 전/후 cloud를 RViz에서 비교합니다.

```bash
roslaunch lidar_obstacle_perception_experimental linefit_self_filter_debug.launch \
  pointcloud_topic:=/ouster/points \
  imu_topic:=/imu/data
```

RViz 색상:

- `Non Ground Raw`: 빨간색, 차체 제거 전
- `Non Ground Self-Filtered`: 노란색, 차체 footprint 제거 후

먼저 조정할 파라미터:

```bash
roslaunch lidar_obstacle_perception_experimental linefit_self_filter_debug.launch \
  pointcloud_topic:=/ouster/points \
  imu_topic:=/imu/data \
  self_filter_padding_m:=0.04 \
  self_filter_center_x_m:=0.0 \
  self_filter_center_y_m:=0.0
```

오르막에서 앞쪽 차체가 더 많이 보이는 현상은 IMU pitch 기반 footprint 보정으로 처리합니다.
기본값은 pitch가 음수일 때 오르막으로 보고 앞쪽 mask를 늘립니다.

```bash
roslaunch lidar_obstacle_perception_experimental linefit_self_filter_debug.launch \
  pointcloud_topic:=/ouster/points \
  imu_topic:=/imu/data \
  self_filter_padding_m:=0.06 \
  pitch_uphill_sign:=-1.0 \
  pitch_adjust_primary_gain_m_per_deg:=0.02
```

로그에서 오르막인데 `front`가 늘지 않고 `rear`가 늘면 pitch 부호가 반대입니다.
그때는 `pitch_uphill_sign:=1.0`으로 바꿔서 비교하면 됩니다.

경사에서 바닥 잔상이 `non-ground`로 얇게 튀면 slope residual filter를 같이 조정합니다.
기본은 IMU pitch가 경사라고 판단될 때만 켜집니다.

```bash
roslaunch lidar_obstacle_perception_experimental linefit_self_filter_debug.launch \
  pointcloud_topic:=/ouster/points \
  imu_topic:=/imu/data \
  self_filter_padding_m:=0.06 \
  slope_filter_max_range_m:=3.0 \
  slope_filter_cell_size_m:=0.18 \
  slope_filter_min_z_span_m:=0.14
```

더 강하게 줄이고 싶으면 `slope_filter_min_z_span_m`을 `0.18` 정도로 올리고,
실제 가까운 물체가 같이 사라지면 `0.10` 정도로 낮춥니다.

오르막/내리막에서 바닥이 `non-ground`로 뜨면 아래 파라미터를 먼저 조정하면 됩니다.

- `max_slope`
- `max_start_height`
- `max_long_height`
- `max_dist_to_line`

예:

```bash
roslaunch lidar_obstacle_perception_experimental linefit_ground_removal_debug.launch \
  pointcloud_topic:=/ouster/points \
  imu_topic:=/imu/data \
  max_slope:=0.50 \
  max_start_height:=0.36 \
  max_long_height:=0.20 \
  max_dist_to_line:=0.09
```

튜닝할 때 주로 보는 파라미터:

- `ground_cell_size_m`
- `ground_clearance_m`
- `ground_range_clearance_per_m`
- `ground_pitch_clearance_gain_m`
- `max_range_m`

## Patchwork++ Ground Removal Debug

공식 ROS1 Patchwork++ 구현으로 같은 구성을 보고 싶으면 아래 launch를 쓰면 됩니다.

```bash
roslaunch lidar_obstacle_perception_experimental patchworkpp_ground_removal_debug.launch \
  pointcloud_topic:=/ouster/points
```

이 launch는:

- `patchworkpp` 공식 ROS package의 `demo` 노드를 사용하고
- `Raw LiDAR / Ground / Non Ground`를 같은 색 구성으로 보여주며
- 기본 센서 높이는 `0.525m`로 맞춰져 있습니다.

튜닝 예:

```bash
roslaunch lidar_obstacle_perception_experimental patchworkpp_ground_removal_debug.launch \
  pointcloud_topic:=/ouster/points \
  sensor_height_m:=0.525 \
  max_range_m:=25 \
  min_range_m:=1.0 \
  th_seeds:=0.30 \
  th_dist:=0.125
```

## Linefit Static / Dynamic Debug

ground는 linefit으로 먼저 제거하고, 그 `non-ground`만 받아서 정적/동적 point cloud로 나누려면 아래 launch를 쓰면 됩니다.

```bash
roslaunch lidar_obstacle_perception_experimental linefit_static_dynamic_debug.launch \
  pointcloud_topic:=/ouster/points \
  imu_topic:=/imu/data \
  odom_topic:=/lio_localizer/odometry/optimization
```

이 RViz 구성은 기본적으로:

- `Ground Cloud`: off
- `Static Obstacle Cloud`: 노란색
- `Dynamic Obstacle Cloud`: 빨간색

으로 보이게 맞춰져 있습니다.

이 launch는 시각화가 복잡해지지 않도록 `static_cloud`를 누적맵이 아니라 `현재 프레임 기준 static point cloud`로 publish합니다.
즉 내부적으로는 추적/판정을 유지하되, RViz에는 시간축 흐름대로 보이게 한 모드입니다.

## 시각화 권장 해석

- `ground_cloud`
  - ground removal 결과로 떨어져 나간 바닥
- `static_cloud`
  - 정적 구조물과 정적 장애물 누적 포인트맵
- `dynamic_cloud`
  - 현재 프레임에서 동적으로 추정된 장애물 포인트
- `dynamic_markers`
  - 동적 장애물 bounding box와 속도 화살표
- `local_obstacle_grid`
  - planner에 넣기 쉬운 로컬 장애물 grid

## 현재 버전의 한계

- ground removal은 height-map 기반 베이스라인입니다.
- 동적 분류는 딥러닝이 아니라 `클러스터 + 최근 이동량 + 속도` 기반 휴리스틱입니다.
- static map은 TTL 기반 누적이므로 장시간 정차한 동적 객체가 정적으로 남을 수 있습니다.

## Noetic 설계 문서

Noetic 기준으로 ground removal, local obstacle layer, static/dynamic separation을
어떻게 단계적으로 붙일지에 대한 설계는 아래 문서를 참고하면 됩니다.

- [NOETIC_PERCEPTION_DESIGN.md](./NOETIC_PERCEPTION_DESIGN.md)
