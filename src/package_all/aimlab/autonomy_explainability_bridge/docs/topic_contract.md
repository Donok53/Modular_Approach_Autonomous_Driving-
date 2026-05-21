# Autonomy Explainability Topic Contract

이 패키지는 자율주행 판단을 바꾸지 않는 read-only bridge입니다. 새 컴퓨터에서 3D object detector와 설명 생성기를 돌릴 때, 어떤 ROS 토픽을 어떤 의미로 사용해야 하는지 정리하고 현재 상태를 JSON으로 요약해서 내보냅니다.

## 실행

기본 사용 방식은 기존 자율주행 launch 하나만 실행하는 것입니다.

```bash
roslaunch src/package_all/run.launch
```

`run.launch`는 기본값으로 explainability bridge를 함께 띄우며, 아래 토픽을 같이 publish합니다.

필요할 때만 끌 수 있습니다.

```bash
roslaunch src/package_all/run.launch enable_explainability_bridge:=false
```

bridge만 단독으로 시험하고 싶을 때는 아래 launch도 사용할 수 있습니다.

```bash
roslaunch autonomy_explainability_bridge explainability_bridge.launch
```

기본 출력은 아래 두 토픽입니다.

| Topic | Type | 내용 |
| --- | --- | --- |
| `/xai/topic_manifest` | `std_msgs/String` JSON | 새 컴퓨터가 구독해야 할 원본 토픽 목록과 사용 목적 |
| `/xai/planner_snapshot` | `std_msgs/String` JSON | 최신 주행 판단, 제어 방향, obstacle evidence 요약 |
| `/xai/event_log` | `std_msgs/String` JSON | 상태 변화 이벤트. 언제 판단/제어/evidence가 바뀌었는지 기록 |

원격 컴퓨터가 `dynamic_window_approach/BehaviorCommand` 같은 custom message를 빌드하지 않아도, `/xai/planner_snapshot`은 `std_msgs/String`이라 바로 읽을 수 있습니다.

## 새 컴퓨터가 우선 구독할 토픽

### 3D Object Detection 입력

| Topic | Type | 목적 |
| --- | --- | --- |
| `/ouster/points` | `sensor_msgs/PointCloud2` | detector 원본 LiDAR 입력 |
| `/tf` | `tf2_msgs/TFMessage` | LiDAR, base_link, map 변환 |
| `/tf_static` | `tf2_msgs/TFMessage` | `os_sensor -> base_link` 같은 static transform |
| `/lio_localizer/odometry/optimization` | `nav_msgs/Odometry` | map 기준 로봇 위치와 yaw |
| `/planning/linefit_ground/non_ground_cloud` | `sensor_msgs/PointCloud2` | ground/self filtering 이후 planner obstacle 입력 |

3D detector는 `/ouster/points`를 직접 보고, 결과는 나중에 별도 토픽으로 publish하는 구조가 좋습니다. 예: `/xai/detected_objects_3d`.

### 로봇 판단과 행동

| Topic | Type | 의미 |
| --- | --- | --- |
| `/planning/behavior_cmd` | `dynamic_window_approach/BehaviorCommand` | `stop`, `speed_limit`, `reason` |
| `/planning/explainability` | `dynamic_window_approach/ExplainabilityEvent` | planner/control/behavior layer가 판단 이유를 event 형태로 publish |
| `/planning/emergency_stop` | `std_msgs/Bool` | DWA 최종 emergency stop 여부 |
| `/astar/path_blocked` | `std_msgs/Bool` | A* 기준 현재 global path blocked 여부 |
| `/planning/global_obstacle_caution` | `std_msgs/Bool` | global obstacle overlay 기반 slowdown caution |
| `/cmd_vel` | `geometry_msgs/Twist` | 실제 제어 출력. 회피 의도라기보다 실제 조향/회전 방향 |
| `/xai/planner_snapshot` | `std_msgs/String` JSON | 위 토픽들을 한 번에 읽기 쉬운 요약 |
| `/xai/event_log` | `std_msgs/String` JSON | 학습/분석용 이벤트 스트림. 상태가 바뀐 순간만 기록 |

현재 코드에서 안전하게 쓸 수 있는 reason은 `/planning/behavior_cmd.reason`에 들어오는 값과 DWA의 상태에서 관찰되는 값입니다. 예전 코드의 `selected action` 이름을 그대로 쓰면 현재 런타임과 어긋날 수 있습니다.

### 장애물 Evidence

| Topic | Type | 의미 |
| --- | --- | --- |
| `/planning/global_obstacle_overlay` | `nav_msgs/OccupancyGrid` | A* global path/candidate 평가에 들어가는 obstacle overlay |
| `/planning/global_obstacle_overlay_boxes` | `visualization_msgs/MarkerArray` | overlay box. DWA emergency stop 판단에도 사용 |
| `/planning/near_field_raw_overlay_hits` | `sensor_msgs/PointCloud2` | raw LiDAR ROI에서 global overlay에 반영된 근거리 hit |
| `/planning/near_field_stop_hits` | `sensor_msgs/PointCloud2` | DWA emergency stop ROI에서 잡힌 근거리 hit |
| `/planning/near_field_raw_debug_cloud` | `sensor_msgs/PointCloud2` | raw near-field ROI debug cloud |

`global_obstacle_overlay`는 emergency stop 전용이 아닙니다. A*가 path 후보를 평가할 때 쓰는 map-frame evidence입니다. 반면 `global_obstacle_overlay_boxes`는 DWA emergency stop 쪽에서도 쓰입니다.

### 경로와 회피 방향 추정

| Topic | Type | 의미 |
| --- | --- | --- |
| `/astar/path` | `nav_msgs/Path` | 현재 선택된 global path |
| `/astar/candidate_paths` | `visualization_msgs/MarkerArray` | A* candidate path 시각화 |
| `/planning/tracking_reference_path` | `nav_msgs/Path` | DWA가 실제 추종 중인 reference path |
| `/planning/path_mode` | `std_msgs/String` | local replanner가 요청한 mode (`hold`, `follow_local`, `follow_avoidance` 등) |
| `/planning/active_path` | `nav_msgs/Path` | optional active path |
| `/planning/local_path` | `nav_msgs/Path` | optional local path |
| `/planning/avoidance_path` | `nav_msgs/Path` | optional avoidance path |

회피 방향은 두 가지로 나눠서 저장하는 것이 좋습니다.

| 이름 | 추천 source | 설명 |
| --- | --- | --- |
| `actual_steering_direction` | `/cmd_vel.angular.z` | 실제 제어 출력 기준 left/right/straight/stop |
| `path_change_direction` | `/astar/path` 변화량 | 새 global path가 이전 path 대비 좌/우로 이동했는지 |

`/cmd_vel`만 보고 "장애물을 피하려고 왼쪽으로 갔다"라고 말하면 위험합니다. goal 정렬 때문에 제자리 회전하는 경우도 있기 때문입니다. obstacle evidence와 path change가 같이 있을 때만 회피 설명으로 묶는 편이 좋습니다.

## `/xai/planner_snapshot` 주요 필드

| Field | 의미 |
| --- | --- |
| `stamp` | ROS time. bag 재생 시 `/clock` 기준 bag timestamp |
| `wall_time` | bridge 컴퓨터의 wall-clock timestamp |
| `robot` | odometry 기반 위치, yaw, 속도 |
| `decision.behavior` | `/planning/behavior_cmd` 요약 |
| `decision.emergency_stop` | `/planning/emergency_stop` 요약 |
| `decision.global_obstacle_caution` | `/planning/global_obstacle_caution` 요약 |
| `decision.path_blocked` | `/astar/path_blocked` 요약 |
| `control` | `/cmd_vel` 기반 실제 조향/회전 방향 |
| `planning.global_path` | `/astar/path` 포인트 수, 길이, 시작/끝 |
| `planning.path_mode` | `/planning/path_mode` 현재 값 |
| `planning.local_path`, `planning.avoidance_path` | local/avoidance path 포인트 수, 길이, 시작/끝 |
| `explainability` | 최신 `/planning/explainability` event 요약. source node, trigger reason, action taken, summary 포함 |
| `planning.path_change` | 이전 path 대비 lateral shift 방향 |
| `obstacle_evidence.global_overlay` | overlay grid 크기와 occupied cell 수 |
| `obstacle_evidence.global_overlay_boxes` | box 수, 가장 가까운 box 위치/거리 |
| `obstacle_evidence.near_field_*` | 근거리 hit cloud 요약 |

## `/xai/event_log` 주요 필드

`/xai/planner_snapshot`은 상태를 계속 찍는 stream이고, `/xai/event_log`는 상태가 바뀌는 순간만 찍는 stream입니다. 학습 데이터 인덱스를 만들 때는 `/xai/event_log`를 기준으로 삼고, 필요하면 같은 `stamp` 근처의 `/ouster/points`, detector 결과, `/xai/planner_snapshot`을 join하는 방식이 좋습니다.

| Field | 의미 |
| --- | --- |
| `seq` | event sequence number |
| `event_type` | `initial_state` 또는 `state_change` |
| `event_label` | `control_update`, `path_update`, `global_obstacle_caution`, `emergency_stop` 등 |
| `stamp` | ROS time. bag 재생 시 bag timestamp |
| `wall_time` | bridge 컴퓨터 wall-clock timestamp |
| `changed_fields` | 이전 event 대비 바뀐 필드 목록 |
| `signature` | event trigger에 쓰인 compact state |
| `source_stamps` | 각 원본 토픽의 최신 stamp |
| `decision` | 해당 순간의 behavior/emergency/caution/path_blocked |
| `control` | 해당 순간의 `/cmd_vel` 요약 |
| `planning` | global path와 path change 요약 |
| `explainability` | 해당 순간의 최신 planner/control/behavior explainability event |
| `obstacle_evidence` | obstacle box와 near-field hit 요약 |

## 설명 생성 시 추천 문장 구조

장애물 detector가 object label을 붙인 뒤에는 아래 순서로 연결하면 됩니다.

```text
로봇은 map 기준 (x, y)의 [object label] evidence를 감지했습니다.
해당 evidence는 [global_obstacle_overlay/path_blocked/caution/emergency_stop]에 반영되었습니다.
그 결과 global path가 [left/right] 방향으로 변경되었고,
현재 제어 출력은 [actual_steering_direction]입니다.
```

이번 bridge는 label을 만들지 않습니다. label은 새 컴퓨터의 3D detector가 붙이고, 이 bridge는 로봇 쪽 planner evidence와 제어 상태를 시간 기준으로 맞춰 설명 서비스가 사용할 수 있게 정리합니다.
