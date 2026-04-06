# LiDAR Height Estimator

Flat-floor PointCloud2 data from a live topic or `rosbag play` is used to estimate:

- LiDAR height above the floor
- observed nearest ground return distance
- optional theoretical first-ground-hit distance if the lowest vertical beam angle is known

## What it does

The node:

1. filters the incoming cloud to keep likely ground candidates
2. excludes points inside the robot body footprint near the sensor
3. fits a ground plane with RANSAC
4. refits the plane with inliers
5. reports the plane distance from the sensor origin as the LiDAR height

This works best when:

- the robot is parked on a flat floor
- LiDAR pitch is close to 0 deg
- there are enough visible ground points around the robot

## Package location

`src/lidar_height_estimator`

## Build

From the workspace root:

```bash
catkin_make
source devel/setup.bash
```

## Run with live data

```bash
roslaunch lidar_height_estimator estimate_height.launch input_topic:=/ouster/points
```

## Run with a rosbag

Terminal 1:

```bash
roslaunch lidar_height_estimator estimate_height.launch input_topic:=/ouster/points
```

Terminal 2:

```bash
rosbag play your_file.bag
```

## Output topics

- `~height_m`: latest single-frame height estimate
- `~height_median_m`: median over the rolling history window
- `~height_std_m`: rolling standard deviation
- `~observed_ground_start_m`: nearest observed ground return distance in the current cloud
- `~theoretical_ground_start_m`: optional, only published if `lowest_vertical_angle_deg` is provided
- `~status_text`: short text summary
- `~ground_points`: optional debug cloud of ground inliers

## Recommended starting params for your robot

These match a Scout Mini-sized body with a center-mounted LiDAR:

- `self_exclusion_half_length_m:=0.45`
- `self_exclusion_half_width_m:=0.40`
- `min_range_m:=0.6`
- `max_range_m:=12.0`
- `max_plane_tilt_deg:=8.0`

## Notes

- `observed_ground_start_m` is not the same as the true blind-zone distance. It is the nearest ground point actually seen in the current scene.
- If you know the LiDAR's lowest vertical angle, set `lowest_vertical_angle_deg` to also estimate the theoretical first-ground-hit distance from the estimated height.
- If the estimate is noisy, increase `process_every_n`, `input_decimation`, or `history_size`, and make sure the robot is stationary on a flat floor.
