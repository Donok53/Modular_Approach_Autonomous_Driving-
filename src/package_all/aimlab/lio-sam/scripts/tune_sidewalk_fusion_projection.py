#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib.util
import itertools
import json
import math
import os
import sys


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_range_list(text):
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("Empty search list: %r" % text)
    return values


def parse_args():
    repo_root = "/home/byeongjae/code/Modular_Approach_Autonomous_Driving-"
    parser = argparse.ArgumentParser(
        description="Grid-search camera projection parameters for sidewalk semantic fusion."
    )
    parser.add_argument(
        "--bundle-dir",
        default=os.path.join(repo_root, "src/package_all/monitoring_delivery/latest"),
    )
    parser.add_argument(
        "--bag",
        default="/home/byeongjae/bagfiles/1.made_map/camera_right.bag",
    )
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(repo_root, "generated/sidewalk_projection_tuning"),
    )
    parser.add_argument("--grid-resolution", type=float, default=0.20)
    parser.add_argument("--map-margin-m", type=float, default=8.0)
    parser.add_argument("--lidar-height-m", type=float, default=0.525)
    parser.add_argument("--camera-height-m", type=float, default=0.445)
    parser.add_argument("--camera-offset-y-m", type=float, default=0.0)
    parser.add_argument("--camera-roll-deg", type=float, default=0.0)
    parser.add_argument("--camera-yaw-deg", type=float, default=0.0)
    parser.add_argument("--pitch-values", default="-8,-6,-4,-2,0,2")
    parser.add_argument("--hfov-values", default="60,64,68,72,76")
    parser.add_argument("--offset-x-values", default="0.00,0.05,0.10")
    parser.add_argument("--sample-stride", type=int, default=240)
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--overlay-frames", type=int, default=4)
    parser.add_argument("--visible-distance-m", type=float, default=15.0)
    parser.add_argument("--visible-lateral-m", type=float, default=6.0)
    parser.add_argument("--min-seed-pixels", type=int, default=30)
    parser.add_argument("--feature-max-distance", type=float, default=0.22)
    parser.add_argument("--feature-margin", type=float, default=0.03)
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
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def prepare_base(fusion, args):
    trajectory_csv = os.path.join(args.bundle_dir, "trajectory.csv")
    map_pcd = os.path.join(args.bundle_dir, "GlobalMap.pcd")
    trajectory_rows = fusion.load_trajectory_with_attitude(trajectory_csv, args.lidar_height_m)
    bounds_xy = fusion.semantic_module.trajectory_bounds(trajectory_rows, args.map_margin_m)
    cell_stats, total_points, kept_points = fusion.semantic_module.accumulate_cells_from_pcd(
        map_pcd,
        bounds_xy,
        args.grid_resolution,
        keep_lowest_n=args.keep_lowest_per_cell,
    )
    cells = fusion.semantic_module.build_cell_map(
        cell_stats,
        min_points_per_cell=args.min_points_per_cell,
        max_obstacle_height_m=args.max_obstacle_height_m,
    )
    classes = fusion.semantic_module.classify_semantics(cells, trajectory_rows, args.grid_resolution, args)
    traversable_keys = set(classes["observed_traversable"])
    stamps = fusion.build_pose_index(trajectory_rows)
    return {
        "trajectory_rows": trajectory_rows,
        "cells": cells,
        "classes": classes,
        "traversable_keys": traversable_keys,
        "stamps": stamps,
        "map_pcd": map_pcd,
        "trajectory_csv": trajectory_csv,
        "points_total": total_points,
        "points_used_in_bbox": kept_points,
    }


def sample_frames(fusion, args, trajectory_rows, stamps):
    import rosbag

    bag = rosbag.Bag(args.bag)
    sampled = []
    for raw_index, (_, image_msg, _) in enumerate(bag.read_messages(topics=[args.image_topic])):
        if raw_index % max(1, args.sample_stride) != 0:
            continue
        if len(sampled) >= args.max_frames:
            break
        if getattr(image_msg, "encoding", "").lower() != "rgb8":
            continue
        ts = image_msg.header.stamp.to_sec()
        sampled.append(
            {
                "raw_index": raw_index,
                "timestamp": ts,
                "width": image_msg.width,
                "height": image_msg.height,
                "step": image_msg.step,
                "data": bytes(image_msg.data),
                "pose_row": fusion.nearest_pose(trajectory_rows, stamps, ts),
            }
        )
    bag.close()
    return sampled


def candidate_score(frame_results):
    used = [r for r in frame_results if r["frame_used"]]
    if not used:
        return -1e18
    used_frames = len(used)
    total_classified = sum(r["classified"] for r in used)
    total_road_seed = sum(r["road_seed_pixels"] for r in used)
    avg_proto = sum(r["prototype_distance"] for r in used) / float(used_frames)
    avg_curb = sum(r["curb_edge_mean"] for r in used) / float(used_frames)
    return (
        used_frames * 5000.0
        + total_classified * 1.0
        + total_road_seed * 6.0
        + avg_proto * 1200.0
        + avg_curb * 900.0
    )


def evaluate_candidate(fusion, base, frames, args, pitch_deg, hfov_deg, offset_x_m):
    body_to_optical = fusion.make_body_to_optical(
        fusion.deg2rad(args.camera_roll_deg),
        fusion.deg2rad(pitch_deg),
        fusion.deg2rad(args.camera_yaw_deg),
    )
    cam_offset_body = (
        offset_x_m,
        args.camera_offset_y_m,
        args.camera_height_m - args.lidar_height_m,
    )

    frame_results = []
    for frame in frames:
        intrinsics = fusion.compute_intrinsics(frame["width"], frame["height"], hfov_deg)
        collected = fusion.collect_visible_candidates(
            frame["data"],
            frame["width"],
            frame["height"],
            frame["step"],
            frame["pose_row"],
            base["cells"],
            base["traversable_keys"],
            base["classes"],
            args.grid_resolution,
            intrinsics,
            body_to_optical,
            cam_offset_body,
            args.visible_distance_m,
            args.visible_lateral_m,
        )
        evaluated = fusion.evaluate_visible_candidates(
            collected,
            args.feature_max_distance,
            args.feature_margin,
            args.min_seed_pixels,
        )
        frame_results.append(evaluated)

    used = [r for r in frame_results if r["frame_used"]]
    result = {
        "pitch_deg": pitch_deg,
        "hfov_deg": hfov_deg,
        "offset_x_m": offset_x_m,
        "used_frames": len(used),
        "total_frames": len(frame_results),
        "total_classified": sum(r["classified"] for r in used),
        "avg_road_seed_pixels": (
            sum(r["road_seed_pixels"] for r in used) / float(len(used))
            if used else 0.0
        ),
        "avg_sidewalk_seed_pixels": (
            sum(r["sidewalk_seed_pixels"] for r in used) / float(len(used))
            if used else 0.0
        ),
        "avg_prototype_distance": (
            sum(r["prototype_distance"] for r in used) / float(len(used))
            if used else 0.0
        ),
        "avg_curb_edge_mean": (
            sum(r["curb_edge_mean"] for r in used) / float(len(used))
            if used else 0.0
        ),
        "score": candidate_score(frame_results),
        "frame_results": frame_results,
    }
    return result


def render_best_overlays(fusion, base, frames, args, best):
    overlay_dir = os.path.join(args.output_dir, "best_overlays")
    ensure_dir(overlay_dir)
    body_to_optical = fusion.make_body_to_optical(
        fusion.deg2rad(args.camera_roll_deg),
        fusion.deg2rad(best["pitch_deg"]),
        fusion.deg2rad(args.camera_yaw_deg),
    )
    cam_offset_body = (
        best["offset_x_m"],
        args.camera_offset_y_m,
        args.camera_height_m - args.lidar_height_m,
    )
    overlay_paths = []
    rendered = 0
    for idx, frame in enumerate(frames):
        if rendered >= args.overlay_frames:
            break
        intrinsics = fusion.compute_intrinsics(frame["width"], frame["height"], best["hfov_deg"])
        collected = fusion.collect_visible_candidates(
            frame["data"],
            frame["width"],
            frame["height"],
            frame["step"],
            frame["pose_row"],
            base["cells"],
            base["traversable_keys"],
            base["classes"],
            args.grid_resolution,
            intrinsics,
            body_to_optical,
            cam_offset_body,
            args.visible_distance_m,
            args.visible_lateral_m,
        )
        evaluated = fusion.evaluate_visible_candidates(
            collected,
            args.feature_max_distance,
            args.feature_margin,
            args.min_seed_pixels,
        )
        if not evaluated["frame_used"]:
            continue
        overlay = bytearray(frame["data"])
        for key, u, v, label in evaluated["classified_candidates"]:
            if label == "sidewalk":
                fusion.draw_dot(overlay, frame["width"], frame["height"], frame["step"], u, v, (210, 230, 210))
            else:
                fusion.draw_dot(overlay, frame["width"], frame["height"], frame["step"], u, v, (70, 120, 200))
        for key, u, v, _ in evaluated["visible_candidates"]:
            if key in base["classes"]["curb"]:
                fusion.draw_dot(overlay, frame["width"], frame["height"], frame["step"], u, v, (220, 70, 60), radius=1)
        path = os.path.join(overlay_dir, "best_frame_%03d.png" % frame["raw_index"])
        fusion.semantic_module.write_png(path, frame["width"], frame["height"], bytes(overlay))
        overlay_paths.append(path)
        rendered += 1
    return overlay_paths


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    fusion = load_module(os.path.join(script_dir, "build_sidewalk_semantic_fusion.py"), "build_sidewalk_semantic_fusion")
    fusion.semantic_module = fusion.load_semantic_module()
    fusion.ensure_rosbag()

    base = prepare_base(fusion, args)
    frames = sample_frames(fusion, args, base["trajectory_rows"], base["stamps"])
    if not frames:
        raise RuntimeError("No usable sampled frames found in %s" % args.bag)

    pitch_values = parse_range_list(args.pitch_values)
    hfov_values = parse_range_list(args.hfov_values)
    offset_x_values = parse_range_list(args.offset_x_values)

    results = []
    for pitch_deg, hfov_deg, offset_x_m in itertools.product(pitch_values, hfov_values, offset_x_values):
        result = evaluate_candidate(fusion, base, frames, args, pitch_deg, hfov_deg, offset_x_m)
        results.append(result)

    results.sort(key=lambda r: r["score"], reverse=True)
    best = results[0]
    overlay_paths = render_best_overlays(fusion, base, frames, args, best)

    manifest = {
        "bag": args.bag,
        "bundle_dir": args.bundle_dir,
        "sampled_frames": len(frames),
        "search": {
            "pitch_values": pitch_values,
            "hfov_values": hfov_values,
            "offset_x_values": offset_x_values,
        },
        "best": {k: v for k, v in best.items() if k != "frame_results"},
        "top5": [
            {k: v for k, v in item.items() if k != "frame_results"}
            for item in results[:5]
        ],
        "camera_prior": {
            "camera_height_m": args.camera_height_m,
            "camera_offset_y_m": args.camera_offset_y_m,
            "camera_roll_deg": args.camera_roll_deg,
            "camera_yaw_deg": args.camera_yaw_deg,
            "lidar_height_m": args.lidar_height_m,
        },
        "overlay_paths": overlay_paths,
    }

    manifest_path = os.path.join(args.output_dir, "projection_tuning_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("projection tuning complete")
    print("  sampled_frames :", len(frames))
    print("  best_pitch_deg :", best["pitch_deg"])
    print("  best_hfov_deg  :", best["hfov_deg"])
    print("  best_offset_x  :", best["offset_x_m"])
    print("  best_score     :", round(best["score"], 3))
    print("  used_frames    :", best["used_frames"])
    print("  avg_road_seed  :", round(best["avg_road_seed_pixels"], 3))
    print("  avg_proto_dist :", round(best["avg_prototype_distance"], 4))
    print("  manifest       :", manifest_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
