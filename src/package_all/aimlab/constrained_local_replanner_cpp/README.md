# constrained_local_replanner_cpp

C++ port-in-progress of `dynamic_window_approach/scripts/constrained_local_replanner.py`.

The node now publishes both `/planning/local_path` and `/planning/path_mode`
and contains a minimal state machine + sidestep + branch-search avoidance,
so it can be flipped from a shadow validator into the primary local
replanner. **Behavior parity with the Python node is NOT fully verified yet
— treat the primary mode as a risky test until the Python and C++ outputs
have been compared on at least one outdoor bag.**

## Build

```bash
source /opt/ros/noetic/setup.bash
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

## Run

### Shadow mode (default — safe, no behavior change)

```bash
roslaunch constrained_local_replanner_cpp replanner_cpp.launch
```

Publishes on `/planning/local_path_cpp` + `/planning/path_mode_cpp`. The
Python node still owns `/planning/local_path` and `/planning/path_mode`.

### Primary mode (replaces Python — disable Python first)

```bash
roslaunch constrained_local_replanner_cpp replanner_cpp.launch primary_mode:=true
```

You **must** disable the Python `constrained_local_replanner` node in your
outer launch before flipping the switch, otherwise both publishers fight
for the same topic and the DWA controller will see flickering modes.

In the project's `src/package_all/run.launch`, the simplest way is to
add `if="$(arg run_cpp_replanner)" / unless="$(arg run_cpp_replanner)"`
flags around the Python node block. (Not done in this commit — set up
manually when you are ready.)

## What's ported

- Drivable / occupancy grid view + occupied / unknown handling.
- Bresenham line-of-sight check.
- `pathBlockedAhead`, `gridPathMinDistanceToXY`, `gridPathEndpointDistanceToXY`.
- 2-D voxel downsample + 4-connected cluster blob detection.
- Sidestep avoidance candidate generator with drivable-aware rejection
  (the same guard we added to the Python node in commit `d6d300d`).
- 8-connected A* branch search with rejoin candidates and budget-bound
  expansion / time-budget.
- Avoidance state machine:
    - `FOLLOW_LOCAL` → `FOLLOW_AVOIDANCE` after `trigger_confirm_cycles`
      consecutive ticks of nominal-path blockage.
    - `FOLLOW_AVOIDANCE` → `FOLLOW_LOCAL` only when robot is within
      `keep_until_endpoint_distance_m` of the cached path's endpoint AND
      the `clear_detour_hold_s` has elapsed AND no locked-static blocker
      is within `locked_static_hold_radius_m` of the robot.
    - Any → `HOLD` when nominal is blocked and no candidate is available.
- locked-static memory: cluster centroids accumulate hits within
  `locked_static_hit_radius_m`; after `locked_static_persistence_hits`
  sightings the entry is "locked" and survives perception dropouts.
- Cached avoidance path re-validation against the current drivable+blocked
  mask every tick (the bug we hit in commit `d6d300d`).
- `/planning/path_mode` publisher with `follow_local` / `follow_avoidance`
  / `hold` strings matching the Python schema.

## What's NOT yet ported (still owned by Python if you stay in shadow mode)

- Tracked-object integration (`tracked_objects`, blind-zone turn conflicts).
- The full nominal-path-blocked decision pipeline (strong-evidence,
  grid_only_nominal_fallback, weak_grid_no_solution_fallback). The C++
  approximation considers a cell blocked if it is non-drivable OR has any
  obstacle overlay overlap.
- Sparse-trigger / weak-evidence trigger suppression.
- bypass diagnostics and post_stop_rotation feedback (those live in DWA,
  not the replanner — they continue to work via DWA's own logic).
- Visualization markers + explainability bridge.
- Reuse_active_avoidance_path with deviation gates beyond the simple
  re-validation. (Reusing the cached path when the candidate disappears
  is implemented; the deviation-based rebuild trigger from
  `_reuse_active_avoidance_path` is not.)
- The long tail of launch parameters that wire into the above.

## Validation checklist before flipping primary_mode

1. Run shadow mode for at least one outdoor bag.
2. In rviz, display `/planning/local_path` (Python) and
   `/planning/local_path_cpp` (C++) as two `Path` topics. They should
   trace similar geometry while the robot drives.
3. `rostopic echo /planning/path_mode_cpp` and compare to
   `/planning/path_mode` — same transitions in roughly the same windows.
4. If both look fine, set `primary_mode:=true` AND disable the Python
   constrained_local_replanner in run.launch.

## Status

State machine + sidestep + branch search are in. Latency improvement vs
Python should be visible only after switching to primary mode and
disabling the Python node — until then the C++ runs in parallel and the
controller still consumes Python output.
