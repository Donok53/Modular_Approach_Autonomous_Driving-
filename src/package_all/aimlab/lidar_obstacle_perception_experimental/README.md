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
