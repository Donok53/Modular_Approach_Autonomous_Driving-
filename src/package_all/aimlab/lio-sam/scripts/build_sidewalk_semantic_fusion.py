#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import bisect
import csv
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict


def parse_args():
    repo_root = "/home/byeongjae/code/Modular_Approach_Autonomous_Driving-"
    parser = argparse.ArgumentParser(
        description="Lightweight camera-LiDAR fusion for sidewalk/road classification on a bag-derived map bundle."
    )
    parser.add_argument(
        "--bundle-dir",
        default=os.path.join(repo_root, "src/package_all/monitoring_delivery/latest"),
        help="Directory containing GlobalMap.pcd and trajectory.csv.",
    )
    parser.add_argument(
        "--bag",
        default="/home/byeongjae/bagfiles/1.made_map/camera_right.bag",
        help="Source rosbag containing /camera/color/image_raw.",
    )
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(repo_root, "generated/sidewalk_semantic_fusion"),
    )
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
    parser.add_argument("--sample-stride", type=int, default=240)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--visible-distance-m", type=float, default=15.0)
    parser.add_argument("--visible-lateral-m", type=float, default=6.0)
    parser.add_argument("--min-seed-pixels", type=int, default=30)
    parser.add_argument("--feature-max-distance", type=float, default=0.22)
    parser.add_argument("--feature-margin", type=float, default=0.03)
    parser.add_argument("--min-votes", type=int, default=3)
    parser.add_argument("--vote-margin", type=int, default=2)
    parser.add_argument("--overlay-frames", type=int, default=6)
    return parser.parse_args()


def load_semantic_module():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(script_dir, "build_sidewalk_semantic_map.py")
    spec = importlib.util.spec_from_file_location("build_sidewalk_semantic_map", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_rosbag():
    try:
        import rosbag  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "rosbag Python import failed: %s. "
            "Try installing PyYAML, pycryptodomex, python-gnupg, and rospkg into the active python."
            % exc
        )


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_trajectory_with_attitude(csv_path, lidar_height_m):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row["x"])
                y = float(row["y"])
                z = float(row["z"])
            except Exception:
                continue
            rows.append(
                {
                    "x": x,
                    "y": y,
                    "z": z,
                    "roll": float(row.get("roll", 0.0) or 0.0),
                    "pitch": float(row.get("pitch", 0.0) or 0.0),
                    "yaw": float(row.get("yaw", 0.0) or 0.0),
                    "ground_z": z - lidar_height_m,
                    "timestamp": float(row.get("timestamp", 0.0) or 0.0),
                }
            )
    if not rows:
        raise RuntimeError("No valid trajectory rows loaded from %s" % csv_path)
    return rows


def deg2rad(v):
    return v * math.pi / 180.0


def rot_x(a):
    ca = math.cos(a)
    sa = math.sin(a)
    return ((1.0, 0.0, 0.0), (0.0, ca, -sa), (0.0, sa, ca))


def rot_y(a):
    ca = math.cos(a)
    sa = math.sin(a)
    return ((ca, 0.0, sa), (0.0, 1.0, 0.0), (-sa, 0.0, ca))


def rot_z(a):
    ca = math.cos(a)
    sa = math.sin(a)
    return ((ca, -sa, 0.0), (sa, ca, 0.0), (0.0, 0.0, 1.0))


def matmul(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def transpose(m):
    return tuple(tuple(m[j][i] for j in range(3)) for i in range(3))


def matvec(m, v):
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def make_body_to_world(roll, pitch, yaw):
    return matmul(rot_z(yaw), matmul(rot_y(pitch), rot_x(roll)))


def make_body_to_optical(camera_roll, camera_pitch, camera_yaw):
    body_to_mount = matmul(rot_z(camera_yaw), matmul(rot_y(camera_pitch), rot_x(camera_roll)))
    optical_from_mount = (
        (0.0, -1.0, 0.0),
        (0.0, 0.0, -1.0),
        (1.0, 0.0, 0.0),
    )
    return matmul(optical_from_mount, body_to_mount)


def compute_intrinsics(width, height, hfov_deg):
    hfov = deg2rad(hfov_deg)
    fx = (0.5 * width) / math.tan(0.5 * hfov)
    fy = fx
    cx = 0.5 * (width - 1)
    cy = 0.5 * (height - 1)
    return fx, fy, cx, cy


def build_pose_index(trajectory_rows):
    stamps = [row["timestamp"] for row in trajectory_rows]
    return stamps


def nearest_pose(trajectory_rows, stamps, ts):
    idx = bisect.bisect_left(stamps, ts)
    if idx <= 0:
        return trajectory_rows[0]
    if idx >= len(stamps):
        return trajectory_rows[-1]
    prev_row = trajectory_rows[idx - 1]
    next_row = trajectory_rows[idx]
    if abs(prev_row["timestamp"] - ts) <= abs(next_row["timestamp"] - ts):
        return prev_row
    return next_row


def pixel_feature(image_bytes, width, step, u, v):
    idx = v * step + u * 3
    r = image_bytes[idx]
    g = image_bytes[idx + 1]
    b = image_bytes[idx + 2]
    denom = max(float(r + g + b), 1.0)
    luma = (0.299 * float(r) + 0.587 * float(g) + 0.114 * float(b)) / 255.0
    return (float(r) / denom, float(g) / denom, float(b) / denom, luma)


def feature_distance(a, b):
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))


def component_median(values):
    if not values:
        return None
    cols = list(zip(*values))
    med = []
    for col in cols:
        s = sorted(col)
        n = len(s)
        mid = n // 2
        if n % 2 == 1:
            med.append(float(s[mid]))
        else:
            med.append(0.5 * float(s[mid - 1] + s[mid]))
    return tuple(med)


def draw_dot(image_bytes, width, height, step, u, v, rgb, radius=2):
    for yy in range(max(0, v - radius), min(height, v + radius + 1)):
        for xx in range(max(0, u - radius), min(width, u + radius + 1)):
            idx = yy * step + xx * 3
            image_bytes[idx] = rgb[0]
            image_bytes[idx + 1] = rgb[1]
            image_bytes[idx + 2] = rgb[2]


def grayscale_luma(image_bytes, step, u, v):
    idx = v * step + u * 3
    r = image_bytes[idx]
    g = image_bytes[idx + 1]
    b = image_bytes[idx + 2]
    return 0.299 * float(r) + 0.587 * float(g) + 0.114 * float(b)


def gradient_magnitude(image_bytes, width, height, step, u, v):
    if u <= 0 or u >= width - 1 or v <= 0 or v >= height - 1:
        return 0.0
    gx = grayscale_luma(image_bytes, step, u + 1, v) - grayscale_luma(image_bytes, step, u - 1, v)
    gy = grayscale_luma(image_bytes, step, u, v + 1) - grayscale_luma(image_bytes, step, u, v - 1)
    return math.sqrt(gx * gx + gy * gy) / 255.0


def project_cell(key, cells, pose_row, world_to_body, body_to_optical, cam_offset_body, intrinsics, resolution):
    fx, fy, cx, cy = intrinsics
    xw, yw = semantic_module.key_to_center(key[0], key[1], resolution)
    zw = cells[key]["ground_z"]
    point_world = (xw, yw, zw)
    pose_xyz = (pose_row["x"], pose_row["y"], pose_row["z"])
    point_body = matvec(world_to_body, vec_sub(point_world, pose_xyz))
    forward = point_body[0]
    lateral = point_body[1]
    point_cam_mount = vec_sub(point_body, cam_offset_body)
    point_opt = matvec(body_to_optical, point_cam_mount)
    if point_opt[2] <= 0.2:
        return None
    u = int(round(fx * (point_opt[0] / point_opt[2]) + cx))
    v = int(round(fy * (point_opt[1] / point_opt[2]) + cy))
    return {
        "u": u,
        "v": v,
        "forward": forward,
        "lateral": lateral,
        "depth": point_opt[2],
    }


def collect_visible_candidates(
    image_data,
    width,
    height,
    step,
    pose_row,
    cells,
    traversable_keys,
    classes,
    resolution,
    intrinsics,
    body_to_optical,
    cam_offset_body,
    visible_distance_m,
    visible_lateral_m,
):
    body_to_world = make_body_to_world(pose_row["roll"], pose_row["pitch"], pose_row["yaw"])
    world_to_body = transpose(body_to_world)
    radius_cells = int(math.ceil(visible_distance_m / resolution))
    center_key = semantic_module.xy_to_key(pose_row["x"], pose_row["y"], resolution)

    sidewalk_feats = []
    road_feats = []
    curb_gradients = []
    visible_candidates = []

    for ix in range(center_key[0] - radius_cells, center_key[0] + radius_cells + 1):
        for iy in range(center_key[1] - radius_cells, center_key[1] + radius_cells + 1):
            key = (ix, iy)
            if key not in traversable_keys:
                continue
            proj = project_cell(
                key,
                cells,
                pose_row,
                world_to_body,
                body_to_optical,
                cam_offset_body,
                intrinsics,
                resolution,
            )
            if proj is None:
                continue
            if proj["forward"] < 0.5 or proj["forward"] > visible_distance_m:
                continue
            if abs(proj["lateral"]) > visible_lateral_m:
                continue
            if proj["u"] < 0 or proj["u"] >= width or proj["v"] < 0 or proj["v"] >= height:
                continue
            feat = pixel_feature(image_data, width, step, proj["u"], proj["v"])
            visible_candidates.append((key, proj["u"], proj["v"], feat))
            if key in classes["sidewalk"]:
                sidewalk_feats.append(feat)
            elif key in classes["road"]:
                road_feats.append(feat)
            elif key in classes["curb"]:
                curb_gradients.append(gradient_magnitude(image_data, width, height, step, proj["u"], proj["v"]))

    return {
        "visible_candidates": visible_candidates,
        "sidewalk_feats": sidewalk_feats,
        "road_feats": road_feats,
        "curb_gradients": curb_gradients,
    }


def evaluate_visible_candidates(result, feature_max_distance, feature_margin, min_seed_pixels):
    sidewalk_feats = result["sidewalk_feats"]
    road_feats = result["road_feats"]
    visible_candidates = result["visible_candidates"]
    curb_gradients = result["curb_gradients"]

    if len(sidewalk_feats) < min_seed_pixels or len(road_feats) < min_seed_pixels:
        return {
            "frame_used": False,
            "sidewalk_seed_pixels": len(sidewalk_feats),
            "road_seed_pixels": len(road_feats),
            "classified": 0,
            "prototype_distance": 0.0,
            "curb_edge_mean": 0.0,
            "sidewalk_proto": None,
            "road_proto": None,
            "classified_candidates": [],
            "visible_candidates": visible_candidates,
        }

    sidewalk_proto = component_median(sidewalk_feats)
    road_proto = component_median(road_feats)
    classified_candidates = []
    for key, u, v, feat in visible_candidates:
        d_sw = feature_distance(feat, sidewalk_proto)
        d_rd = feature_distance(feat, road_proto)
        if min(d_sw, d_rd) > feature_max_distance:
            continue
        if abs(d_sw - d_rd) < feature_margin:
            continue
        label = "sidewalk" if d_sw < d_rd else "road"
        classified_candidates.append((key, u, v, label))

    curb_edge_mean = 0.0
    if curb_gradients:
        curb_edge_mean = sum(curb_gradients) / float(len(curb_gradients))

    return {
        "frame_used": True,
        "sidewalk_seed_pixels": len(sidewalk_feats),
        "road_seed_pixels": len(road_feats),
        "classified": len(classified_candidates),
        "prototype_distance": feature_distance(sidewalk_proto, road_proto),
        "curb_edge_mean": curb_edge_mean,
        "sidewalk_proto": sidewalk_proto,
        "road_proto": road_proto,
        "classified_candidates": classified_candidates,
        "visible_candidates": visible_candidates,
    }


def classify_visible_cells(
    image_msg,
    pose_row,
    cells,
    traversable_keys,
    classes,
    resolution,
    intrinsics,
    body_to_optical,
    cam_offset_body,
    args,
    vote_counts,
    overlays,
    frame_index,
):
    width = image_msg.width
    height = image_msg.height
    step = image_msg.step
    image_bytes = image_msg.data

    collected = collect_visible_candidates(
        image_bytes,
        width,
        height,
        step,
        pose_row,
        cells,
        traversable_keys,
        classes,
        resolution,
        intrinsics,
        body_to_optical,
        cam_offset_body,
        args.visible_distance_m,
        args.visible_lateral_m,
    )
    evaluation = evaluate_visible_candidates(
        collected,
        args.feature_max_distance,
        args.feature_margin,
        args.min_seed_pixels,
    )
    if not evaluation["frame_used"]:
        return {
            "frame_used": False,
            "sidewalk_seed_pixels": evaluation["sidewalk_seed_pixels"],
            "road_seed_pixels": evaluation["road_seed_pixels"],
            "classified": 0,
            "prototype_distance": 0.0,
            "curb_edge_mean": 0.0,
        }

    overlay_image = None
    if frame_index < args.overlay_frames:
        overlay_image = bytearray(image_bytes)

    for key, u, v, label in evaluation["classified_candidates"]:
        if label == "sidewalk":
            vote_counts[key]["sidewalk"] += 1
            if overlay_image is not None:
                draw_dot(overlay_image, width, height, step, u, v, (210, 230, 210))
        else:
            vote_counts[key]["road"] += 1
            if overlay_image is not None:
                draw_dot(overlay_image, width, height, step, u, v, (70, 120, 200))

    if overlay_image is not None:
        for key, u, v, _ in evaluation["visible_candidates"]:
            if key in classes["curb"]:
                draw_dot(overlay_image, width, height, step, u, v, (220, 70, 60), radius=1)
        overlays.append(
            {
                "frame_index": frame_index,
                "timestamp": image_msg.header.stamp.to_sec(),
                "width": width,
                "height": height,
                "rgb": bytes(overlay_image),
            }
        )

    return {
        "frame_used": True,
        "sidewalk_seed_pixels": evaluation["sidewalk_seed_pixels"],
        "road_seed_pixels": evaluation["road_seed_pixels"],
        "classified": evaluation["classified"],
        "prototype_distance": evaluation["prototype_distance"],
        "curb_edge_mean": evaluation["curb_edge_mean"],
    }


def write_overlay_pngs(out_dir, overlays):
    ensure_dir(out_dir)
    paths = []
    for overlay in overlays:
        path = os.path.join(out_dir, "frame_%03d.png" % overlay["frame_index"])
        semantic_module.write_png(path, overlay["width"], overlay["height"], overlay["rgb"])
        paths.append(path)
    return paths


def fuse_classes(cells, base_classes, vote_counts, args):
    sidewalk = set()
    road = set()
    curb = set(base_classes["curb"])

    for key, info in cells.items():
        if info["is_obstacle"] or key in curb:
            continue
        votes = vote_counts.get(key, {})
        sv = int(votes.get("sidewalk", 0))
        rv = int(votes.get("road", 0))
        if sv >= args.min_votes and sv >= rv + args.vote_margin:
            sidewalk.add(key)
            continue
        if rv >= args.min_votes and rv >= sv + args.vote_margin:
            road.add(key)
            continue
        if key in base_classes["sidewalk"]:
            sidewalk.add(key)
        elif key in base_classes["road"]:
            road.add(key)

    return {
        "sidewalk": sidewalk,
        "road": road,
        "curb": curb,
        "observed_traversable": base_classes["observed_traversable"],
    }


def main():
    global semantic_module

    args = parse_args()
    ensure_dir(args.output_dir)
    ensure_rosbag()

    semantic_module = load_semantic_module()
    map_pcd = os.path.join(args.bundle_dir, "GlobalMap.pcd")
    trajectory_csv = os.path.join(args.bundle_dir, "trajectory.csv")
    trajectory_rows = load_trajectory_with_attitude(trajectory_csv, args.lidar_height_m)
    bounds_xy = semantic_module.trajectory_bounds(trajectory_rows, args.map_margin_m)
    cell_stats, total_points, kept_points = semantic_module.accumulate_cells_from_pcd(
        map_pcd,
        bounds_xy,
        args.grid_resolution,
        keep_lowest_n=args.keep_lowest_per_cell,
    )
    cells = semantic_module.build_cell_map(
        cell_stats,
        min_points_per_cell=args.min_points_per_cell,
        max_obstacle_height_m=args.max_obstacle_height_m,
    )
    base_classes = semantic_module.classify_semantics(cells, trajectory_rows, args.grid_resolution, args)

    traversable_keys = set(base_classes["observed_traversable"])
    stamps = build_pose_index(trajectory_rows)
    body_to_optical = make_body_to_optical(
        deg2rad(args.camera_roll_deg),
        deg2rad(args.camera_pitch_deg),
        deg2rad(args.camera_yaw_deg),
    )
    cam_offset_body = (
        args.camera_offset_x_m,
        args.camera_offset_y_m,
        args.camera_height_m - args.lidar_height_m,
    )

    import rosbag

    bag = rosbag.Bag(args.bag)
    vote_counts = defaultdict(lambda: {"sidewalk": 0, "road": 0})
    overlays = []
    frame_reports = []
    intrinsics = None
    sampled = 0

    for raw_index, (_, image_msg, _) in enumerate(bag.read_messages(topics=[args.image_topic])):
        if raw_index % max(1, args.sample_stride) != 0:
            continue
        if sampled >= args.max_frames:
            break
        if getattr(image_msg, "encoding", "").lower() != "rgb8":
            continue
        if intrinsics is None:
            intrinsics = compute_intrinsics(image_msg.width, image_msg.height, args.camera_hfov_deg)
        ts = image_msg.header.stamp.to_sec()
        pose_row = nearest_pose(trajectory_rows, stamps, ts)
        report = classify_visible_cells(
            image_msg,
            pose_row,
            cells,
            traversable_keys,
            base_classes,
            args.grid_resolution,
            intrinsics,
            body_to_optical,
            cam_offset_body,
            args,
            vote_counts,
            overlays,
            sampled,
        )
        report["frame_index"] = sampled
        report["raw_image_index"] = raw_index
        report["timestamp"] = ts
        frame_reports.append(report)
        sampled += 1

    bag.close()

    fused_classes = fuse_classes(cells, base_classes, vote_counts, args)

    sidewalk_pcd = os.path.join(args.output_dir, "SidewalkMap.fused.pcd")
    road_pcd = os.path.join(args.output_dir, "RoadMap.fused.pcd")
    curb_pcd = os.path.join(args.output_dir, "CurbMap.fused.pcd")
    preview_png = os.path.join(args.output_dir, "sidewalk_semantic_fused_preview.png")
    overlay_dir = os.path.join(args.output_dir, "frame_overlays")
    manifest_json = os.path.join(args.output_dir, "sidewalk_semantic_fusion_manifest.json")

    semantic_module.write_ascii_pcd(sidewalk_pcd, fused_classes["sidewalk"], cells, args.grid_resolution)
    semantic_module.write_ascii_pcd(road_pcd, fused_classes["road"], cells, args.grid_resolution)
    semantic_module.write_ascii_pcd(curb_pcd, fused_classes["curb"], cells, args.grid_resolution)
    semantic_module.write_preview_png(
        preview_png,
        fused_classes,
        cells,
        trajectory_rows,
        args.grid_resolution,
        4,
    )
    overlay_paths = write_overlay_pngs(overlay_dir, overlays)

    nonzero_votes = sum(1 for v in vote_counts.values() if v["sidewalk"] or v["road"])
    manifest = {
        "bag": args.bag,
        "bundle_dir": args.bundle_dir,
        "map_pcd": map_pcd,
        "trajectory_csv": trajectory_csv,
        "points_total": total_points,
        "points_used_in_bbox": kept_points,
        "cells_observed": len(cells),
        "base_cells_sidewalk": len(base_classes["sidewalk"]),
        "base_cells_road": len(base_classes["road"]),
        "base_cells_curb": len(base_classes["curb"]),
        "fused_cells_sidewalk": len(fused_classes["sidewalk"]),
        "fused_cells_road": len(fused_classes["road"]),
        "fused_cells_curb": len(fused_classes["curb"]),
        "cells_with_camera_votes": nonzero_votes,
        "sampled_frames": sampled,
        "frames_used": sum(1 for r in frame_reports if r["frame_used"]),
        "camera": {
            "height_m": args.camera_height_m,
            "offset_x_m": args.camera_offset_x_m,
            "offset_y_m": args.camera_offset_y_m,
            "roll_deg": args.camera_roll_deg,
            "pitch_deg": args.camera_pitch_deg,
            "yaw_deg": args.camera_yaw_deg,
            "hfov_deg": args.camera_hfov_deg,
            "intrinsics": {
                "fx": intrinsics[0] if intrinsics else None,
                "fy": intrinsics[1] if intrinsics else None,
                "cx": intrinsics[2] if intrinsics else None,
                "cy": intrinsics[3] if intrinsics else None,
            },
        },
        "params": {
            "sample_stride": args.sample_stride,
            "max_frames": args.max_frames,
            "visible_distance_m": args.visible_distance_m,
            "visible_lateral_m": args.visible_lateral_m,
            "min_seed_pixels": args.min_seed_pixels,
            "feature_max_distance": args.feature_max_distance,
            "feature_margin": args.feature_margin,
            "min_votes": args.min_votes,
            "vote_margin": args.vote_margin,
        },
        "frame_reports": frame_reports,
        "outputs": {
            "sidewalk_pcd": sidewalk_pcd,
            "road_pcd": road_pcd,
            "curb_pcd": curb_pcd,
            "preview_png": preview_png,
            "overlay_dir": overlay_dir,
            "overlay_frames": overlay_paths,
        },
    }
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("camera-lidar fusion complete")
    print("  sampled_frames       :", sampled)
    print("  frames_used          :", sum(1 for r in frame_reports if r["frame_used"]))
    print("  cells_with_cam_votes :", nonzero_votes)
    print("  fused_sidewalk       :", len(fused_classes["sidewalk"]))
    print("  fused_road           :", len(fused_classes["road"]))
    print("  fused_curb           :", len(fused_classes["curb"]))
    print("  preview              :", preview_png)
    print("  manifest             :", manifest_json)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
