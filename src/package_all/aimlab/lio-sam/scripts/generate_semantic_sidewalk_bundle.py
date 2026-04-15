#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace


REPO_ROOT = "/home/byeongjae/code/Modular_Approach_Autonomous_Driving-"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate semantic sidewalk outputs and sidewalk-only drivable state from a saved LIO-SAM bundle."
    )
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--map-pcd", default="")
    parser.add_argument("--trajectory-csv", default="")
    parser.add_argument("--source-bag", default="")
    parser.add_argument("--semantic-state-json", default="")
    parser.add_argument("--drivable-state-json", default="")
    parser.add_argument("--global-drivable-state-json", default="")
    parser.add_argument("--lidar-frame-stride", type=int, default=1)
    parser.add_argument("--point-stride", type=int, default=8)
    parser.add_argument("--max-lidar-frames", type=int, default=100000)
    parser.add_argument("--point-topic", default="/ouster/points")
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--imu-topic", default="/imu/data")
    parser.add_argument("--grid-resolution", type=float, default=0.20)
    parser.add_argument("--map-margin-m", type=float, default=8.0)
    parser.add_argument("--lidar-height-m", type=float, default=0.525)
    parser.add_argument("--min-points-per-cell", type=int, default=3)
    parser.add_argument("--keep-lowest-per-cell", type=int, default=5)
    parser.add_argument("--max-obstacle-height-m", type=float, default=0.35)
    parser.add_argument("--traj-seed-radius-m", type=float, default=0.90)
    parser.add_argument("--traj-seed-height-tol-m", type=float, default=0.16)
    parser.add_argument("--sidewalk-fill-max-distance-m", type=float, default=2.4)
    parser.add_argument("--sidewalk-fill-height-tol-m", type=float, default=0.12)
    parser.add_argument("--neighbor-height-tol-m", type=float, default=0.07)
    parser.add_argument("--curb-height-diff-m", type=float, default=0.08)
    parser.add_argument("--road-min-drop-from-sidewalk-m", type=float, default=0.04)
    parser.add_argument("--road-max-drop-from-sidewalk-m", type=float, default=0.30)
    parser.add_argument("--camera-height-m", type=float, default=0.445)
    parser.add_argument("--camera-offset-x-m", type=float, default=0.025)
    parser.add_argument("--camera-offset-y-m", type=float, default=0.0)
    parser.add_argument("--camera-roll-deg", type=float, default=0.0)
    parser.add_argument("--camera-pitch-deg", type=float, default=0.0)
    parser.add_argument("--camera-yaw-deg", type=float, default=0.0)
    parser.add_argument("--camera-hfov-deg", type=float, default=78.0)
    parser.add_argument("--preview-cell-size", type=int, default=4)
    return parser.parse_args()


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def build_map_args(cli):
    return SimpleNamespace(
        bundle_dir=cli.bundle_dir,
        map_pcd=cli.map_pcd,
        trajectory_csv=cli.trajectory_csv,
        source_bag=cli.source_bag,
        output_dir=cli.output_dir,
        grid_resolution=cli.grid_resolution,
        map_margin_m=cli.map_margin_m,
        lidar_height_m=cli.lidar_height_m,
        min_points_per_cell=cli.min_points_per_cell,
        keep_lowest_per_cell=cli.keep_lowest_per_cell,
        max_obstacle_height_m=cli.max_obstacle_height_m,
        traj_seed_radius_m=cli.traj_seed_radius_m,
        traj_seed_height_tol_m=cli.traj_seed_height_tol_m,
        sidewalk_fill_max_distance_m=cli.sidewalk_fill_max_distance_m,
        sidewalk_fill_height_tol_m=cli.sidewalk_fill_height_tol_m,
        neighbor_height_tol_m=cli.neighbor_height_tol_m,
        curb_height_diff_m=cli.curb_height_diff_m,
        road_min_drop_from_sidewalk_m=cli.road_min_drop_from_sidewalk_m,
        road_max_drop_from_sidewalk_m=cli.road_max_drop_from_sidewalk_m,
        preview_cell_size=cli.preview_cell_size,
    )


def semantic_label_for_key(key, classes):
    if key in classes["curb"]:
        return "curb"
    if key in classes["sidewalk"]:
        return "sidewalk"
    if key in classes["road"]:
        return "road"
    return "observed"


def sorted_cell_lists(classes):
    return {
        label: [[ix, iy] for ix, iy in sorted(keys)]
        for label, keys in classes.items()
        if label in ("sidewalk", "road", "curb")
    }


def write_editable_state(path, classes, cells, args, map_pcd, trajectory_csv):
    payload = {
        "meta": {
            "grid_resolution": float(args.grid_resolution),
            "map_pcd": map_pcd,
            "trajectory_csv": trajectory_csv,
            "source_bag": args.source_bag,
            "generated_by": "generate_semantic_sidewalk_bundle.py",
            "generated_at": float(time.time()),
        },
        "classes": sorted_cell_lists(classes),
        "observed_cells": [
            [ix, iy, float(cells[(ix, iy)]["ground_z"]), semantic_label_for_key((ix, iy), classes)]
            for ix, iy in sorted(cells.keys())
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def build_drivable_payload(classes, cells, resolution, trajectory_rows):
    sidewalk_keys = sorted(classes["sidewalk"])
    ground_lookup = {
        (ix, iy): float(cells[(ix, iy)]["ground_z"])
        for ix, iy in sidewalk_keys
        if (ix, iy) in cells
    }
    return build_drivable_payload_from_ground_lookup(sidewalk_keys, ground_lookup, resolution, trajectory_rows)


def build_drivable_payload_from_ground_lookup(sidewalk_keys, ground_lookup, resolution, trajectory_rows):
    cell_rows = []
    z_values = []
    for ix, iy in sidewalk_keys:
        z = float(ground_lookup[(ix, iy)])
        cell_rows.append([int(ix), int(iy), z])
        z_values.append(z)

    last_seed_xy = None
    if trajectory_rows:
        last_seed_xy = [float(trajectory_rows[0]["x"]), float(trajectory_rows[0]["y"])]
    elif sidewalk_keys:
        ix, iy = sidewalk_keys[0]
        last_seed_xy = [(ix + 0.5) * resolution, (iy + 0.5) * resolution]

    last_odom_z = float(sum(z_values) / float(len(z_values))) if z_values else 0.0
    return {
        "version": 1,
        "grid_resolution_m": float(resolution),
        "cells": cell_rows,
        "risk_cells": [],
        "last_seed_xy": last_seed_xy,
        "last_odom_z": last_odom_z,
        "saved_at": float(time.time()),
    }


def load_raw_state(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    classes = {
        label: {
            (int(row[0]), int(row[1]))
            for row in payload.get("classes", {}).get(label, [])
            if len(row) >= 2
        }
        for label in ("sidewalk", "road", "curb")
    }
    observed_ground = {}
    for row in payload.get("observed_cells", []):
        if len(row) < 3:
            continue
        observed_ground[(int(row[0]), int(row[1]))] = float(row[2])
    return payload, classes, observed_ground


def run_raw_bag_pipeline(cli, semantic_state_json, drivable_state_json, global_drivable_state_json):
    raw_script = os.path.join(os.path.dirname(__file__), "build_raw_semantic_bev_map.py")
    raw_output_dir = cli.output_dir
    ensure_dir(raw_output_dir)
    command = [
        sys.executable,
        raw_script,
        "--bag", cli.source_bag,
        "--bundle-dir", cli.bundle_dir,
        "--output-dir", raw_output_dir,
        "--point-topic", cli.point_topic,
        "--image-topic", cli.image_topic,
        "--imu-topic", cli.imu_topic,
        "--grid-resolution", str(cli.grid_resolution),
        "--lidar-frame-stride", str(cli.lidar_frame_stride),
        "--point-stride", str(cli.point_stride),
        "--max-lidar-frames", str(cli.max_lidar_frames),
        "--camera-height-m", str(cli.camera_height_m),
        "--camera-offset-x-m", str(cli.camera_offset_x_m),
        "--camera-offset-y-m", str(cli.camera_offset_y_m),
        "--camera-roll-deg", str(cli.camera_roll_deg),
        "--camera-pitch-deg", str(cli.camera_pitch_deg),
        "--camera-yaw-deg", str(cli.camera_yaw_deg),
        "--camera-hfov-deg", str(cli.camera_hfov_deg),
        "--lidar-height-m", str(cli.lidar_height_m),
        "--preview-cell-size", str(cli.preview_cell_size),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "raw semantic bev generation failed\nstdout:\n%s\nstderr:\n%s"
            % (result.stdout, result.stderr)
        )

    raw_state_json = os.path.join(raw_output_dir, "raw_semantic_bev_state.json")
    raw_manifest_json = os.path.join(raw_output_dir, "raw_semantic_bev_manifest.json")
    raw_override_template = os.path.join(raw_output_dir, "raw_semantic_bev_override.template.json")
    if not os.path.isfile(raw_state_json):
        raise FileNotFoundError("raw semantic state not found: %s" % raw_state_json)

    raw_state, classes, observed_ground = load_raw_state(raw_state_json)
    shutil.copy2(raw_state_json, semantic_state_json)
    if os.path.isfile(raw_override_template):
        shutil.copy2(
            raw_override_template,
            os.path.join(os.path.dirname(semantic_state_json), "semantic_sidewalk_override.template.json"),
        )

    trajectory_csv = os.path.join(cli.bundle_dir, "trajectory.csv")
    trajectory_rows = []
    if os.path.isfile(trajectory_csv):
        map_module_path = os.path.join(os.path.dirname(__file__), "build_sidewalk_semantic_map.py")
        map_mod = load_module(map_module_path, "build_sidewalk_semantic_map")
        trajectory_rows = map_mod.load_trajectory(trajectory_csv, cli.lidar_height_m)

    resolution = float(raw_state.get("meta", {}).get("grid_resolution", cli.grid_resolution))
    drivable_payload = build_drivable_payload_from_ground_lookup(
        sorted(classes["sidewalk"]),
        observed_ground,
        resolution,
        trajectory_rows,
    )
    write_json(drivable_state_json, drivable_payload)
    if global_drivable_state_json:
        write_json(global_drivable_state_json, drivable_payload)

    manifest = {}
    if os.path.isfile(raw_manifest_json):
        with open(raw_manifest_json, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    manifest.update(
        {
            "generation_mode": "raw_bag_accumulation",
            "semantic_state_json": semantic_state_json,
            "drivable_state_json": drivable_state_json,
            "global_drivable_state_json": global_drivable_state_json,
            "stdout": result.stdout,
        }
    )
    return manifest


def write_json(path, payload):
    out_dir = os.path.dirname(path)
    ensure_dir(out_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    cli = parse_args()
    cli.bundle_dir = os.path.abspath(os.path.expanduser(cli.bundle_dir))
    cli.source_bag = os.path.abspath(os.path.expanduser(cli.source_bag)) if cli.source_bag else ""
    cli.output_dir = os.path.abspath(
        os.path.expanduser(cli.output_dir or os.path.join(cli.bundle_dir, "semantic_sidewalk"))
    )
    semantic_state_json = os.path.abspath(
        os.path.expanduser(cli.semantic_state_json or os.path.join(cli.bundle_dir, "semantic_sidewalk_state.json"))
    )
    drivable_state_json = os.path.abspath(
        os.path.expanduser(cli.drivable_state_json or os.path.join(cli.bundle_dir, "lio_sam_drivable_area_state.json"))
    )
    global_drivable_state_json = os.path.abspath(os.path.expanduser(cli.global_drivable_state_json)) if cli.global_drivable_state_json else ""

    ensure_dir(cli.output_dir)

    source_bag_exists = bool(cli.source_bag) and os.path.isfile(os.path.expanduser(cli.source_bag))
    if source_bag_exists:
        manifest = run_raw_bag_pipeline(
            cli,
            semantic_state_json,
            drivable_state_json,
            global_drivable_state_json,
        )
        manifest_json = os.path.join(cli.output_dir, "semantic_sidewalk_manifest.json")
        write_json(manifest_json, manifest)
        print("semantic sidewalk bundle complete")
        print("  mode             : raw_bag_accumulation")
        print("  semantic_state   :", semantic_state_json)
        print("  drivable_state   :", drivable_state_json)
        if global_drivable_state_json:
            print("  global_drivable  :", global_drivable_state_json)
        print("  manifest         :", manifest_json)
        return

    map_module_path = os.path.join(os.path.dirname(__file__), "build_sidewalk_semantic_map.py")
    map_mod = load_module(map_module_path, "build_sidewalk_semantic_map")
    map_args = build_map_args(cli)
    map_pcd, trajectory_csv = map_mod.resolve_input_paths(map_args)
    trajectory_rows = map_mod.load_trajectory(trajectory_csv, cli.lidar_height_m)
    bounds_xy = map_mod.trajectory_bounds(trajectory_rows, cli.map_margin_m)
    cell_stats, total_points, kept_points = map_mod.accumulate_cells_from_pcd(
        map_pcd,
        bounds_xy,
        cli.grid_resolution,
        cli.keep_lowest_per_cell,
    )
    cells = map_mod.build_cell_map(cell_stats, cli.min_points_per_cell, cli.max_obstacle_height_m)
    classes = map_mod.classify_semantics(cells, trajectory_rows, cli.grid_resolution, map_args)

    sidewalk_pcd = os.path.join(cli.output_dir, "SidewalkMap.pcd")
    road_pcd = os.path.join(cli.output_dir, "RoadMap.pcd")
    curb_pcd = os.path.join(cli.output_dir, "CurbMap.pcd")
    preview_png = os.path.join(cli.output_dir, "sidewalk_semantic_preview.png")
    manifest_json = os.path.join(cli.output_dir, "sidewalk_semantic_manifest.json")

    map_mod.write_ascii_pcd(sidewalk_pcd, classes["sidewalk"], cells, cli.grid_resolution)
    map_mod.write_ascii_pcd(road_pcd, classes["road"], cells, cli.grid_resolution)
    map_mod.write_ascii_pcd(curb_pcd, classes["curb"], cells, cli.grid_resolution)
    map_mod.write_preview_png(
        preview_png,
        classes,
        cells,
        trajectory_rows,
        cli.grid_resolution,
        max(1, cli.preview_cell_size),
    )

    editable_payload = write_editable_state(semantic_state_json, classes, cells, cli, map_pcd, trajectory_csv)
    drivable_payload = build_drivable_payload(classes, cells, cli.grid_resolution, trajectory_rows)
    write_json(drivable_state_json, drivable_payload)
    if global_drivable_state_json:
        write_json(global_drivable_state_json, drivable_payload)

    manifest = {
        "source_bag": cli.source_bag,
        "map_pcd": map_pcd,
        "trajectory_csv": trajectory_csv,
        "output_dir": cli.output_dir,
        "semantic_state_json": semantic_state_json,
        "drivable_state_json": drivable_state_json,
        "global_drivable_state_json": global_drivable_state_json,
        "grid_resolution_m": cli.grid_resolution,
        "lidar_height_m": cli.lidar_height_m,
        "points_total": total_points,
        "points_used_in_bbox": kept_points,
        "cells_observed": len(cells),
        "cells_sidewalk": len(classes["sidewalk"]),
        "cells_road": len(classes["road"]),
        "cells_curb": len(classes["curb"]),
        "outputs": {
            "sidewalk_pcd": sidewalk_pcd,
            "road_pcd": road_pcd,
            "curb_pcd": curb_pcd,
            "preview_png": preview_png,
            "semantic_state_json": semantic_state_json,
            "drivable_state_json": drivable_state_json,
        },
    }
    write_json(manifest_json, manifest)

    print("semantic sidewalk bundle complete")
    print("  semantic_state   :", semantic_state_json)
    print("  drivable_state   :", drivable_state_json)
    if global_drivable_state_json:
        print("  global_drivable  :", global_drivable_state_json)
    print("  sidewalk_cells   :", len(classes["sidewalk"]))
    print("  preview          :", preview_png)


if __name__ == "__main__":
    main()
