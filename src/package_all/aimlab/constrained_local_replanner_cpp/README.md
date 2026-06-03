# constrained_local_replanner_cpp

C++ port-in-progress of `dynamic_window_approach/scripts/constrained_local_replanner.py`.

This is **NOT** a drop-in replacement for the Python node yet. It is a
foundation that ports the hottest inner loops to native C++ so the Python
node's 2-3 s `loop=overrun` can be retired in stages.

## Build

This is a catkin package. From the workspace root:

```bash
source /opt/ros/noetic/setup.bash
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

## Run alongside the existing Python stack

```bash
roslaunch constrained_local_replanner_cpp replanner_cpp.launch
```

It publishes its candidate avoidance path on `/planning/local_path_cpp` so you
can compare it to the Python node's `/planning/local_path` in rviz or rosbag
without changing the existing pipeline. The DWA controller is unaffected.

## Scope — what's ported vs. still in Python

### Ported (in this package)

- Drivable / occupancy grid view + occupied / unknown handling.
- Bresenham line-of-sight check (`hasLineOfSight`).
- `pathBlockedAhead` — the hot per-tick blocked-cells check.
- `gridPathMinDistanceToXY`, `gridPathEndpointDistanceToXY` — used by the
  release / endpoint guards on the Python side.
- 2-D voxel downsample + 4-connected cluster blob detection.
- Sidestep avoidance candidate generator (`buildSidestepAvoidance`) — the
  Python `_build_sidestep_avoidance_path` equivalent, minus the static-memory
  preference scoring.
- A minimal ROS node that subscribes to cloud / drivable grid / odom /
  global path and publishes a parallel local-path candidate.

### Not yet ported — still owned by the Python node

These rely on the existing replanner; do not turn off Python until they are
ported:

- Trigger debounce / confirm cycles / `avoid_pending` state machine.
- Static-obstacle memory and `locked_static` confirmation.
- Dynamic tracker integration (`tracked_objects` / blind-zone).
- Branch / A* search variant (`_build_branch_avoidance_path`).
- Path-mode publishing (`follow_local` / `follow_avoidance` / `hold`).
- Reuse-with-deviation and clear-hold release logic
  (`_hold_active_avoidance_until_endpoint`, `_reuse_active_avoidance_path`).
- Bypass diagnostics, post_stop_rotation feedback, weak-evidence trigger
  suppression, sparse-trigger debouncing.
- Visualization markers, explainability bridge, debug text publishing.
- Pitch-mode aware cloud filtering, memory persistence frames.
- Every launch parameter that wires into the above.

## Next steps

Once the parallel C++ candidate matches Python output for nominal-path and
sidestep cases (verify in rviz / rosbag for at least one outdoor run), the
next priorities are:

1. Port `_build_branch_avoidance_path` (A* with rejoin candidates) — biggest
   remaining CPU cost.
2. Port the trigger debounce + path-mode state machine so this node can
   actually own `/planning/local_path` and `/planning/path_mode`.
3. Port static-memory `locked_static` accumulation.
4. Replace the Python node with this one in `run.launch` once 1-3 are stable.

## Status

Foundational scaffold + core compute primitives. Tested by inspection only —
NOT yet validated against the existing Python node on a bag. Expect to need
parameter alignment and a side-by-side rviz session before promoting it.
