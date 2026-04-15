#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import bisect
import csv
import importlib.util
import json
import math
import os
import struct
import sys
from collections import defaultdict, deque


SEMANTIC_MAP_MODULE = None


class LocalCellStats:
    __slots__ = ("count", "low_z", "max_z")

    def __init__(self):
        self.count = 0
        self.low_z = []
        self.max_z = None

    def add(self, z, keep_lowest_n):
        self.count += 1
        if self.max_z is None or z > self.max_z:
            self.max_z = z
        if len(self.low_z) < keep_lowest_n:
            bisect.insort(self.low_z, z)
            return
        if z >= self.low_z[-1]:
            return
        bisect.insort(self.low_z, z)
        del self.low_z[keep_lowest_n:]

    def ground_z(self):
        if not self.low_z:
            return None
        mid = len(self.low_z) // 2
        if len(self.low_z) % 2 == 1:
            return float(self.low_z[mid])
        return 0.5 * float(self.low_z[mid - 1] + self.low_z[mid])


def parse_args():
    repo_root = "/home/byeongjae/code/Modular_Approach_Autonomous_Driving-"
    parser = argparse.ArgumentParser(
        description="Build a raw bag semantic BEV map using lidar, camera, imu, and trajectory pose."
    )
    parser.add_argument("--bag", default="/home/byeongjae/bagfiles/1.made_map/camera_right.bag")
    parser.add_argument("--bundle-dir", default=os.path.join(repo_root, "src/package_all/monitoring_delivery/latest"))
    parser.add_argument("--point-topic", default="/ouster/points")
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--imu-topic", default="/imu/data")
    parser.add_argument("--output-dir", default=os.path.join(repo_root, "generated/raw_semantic_bev_map"))
    parser.add_argument("--grid-resolution", type=float, default=0.20)
    parser.add_argument("--lidar-frame-stride", type=int, default=5)
    parser.add_argument("--point-stride", type=int, default=8)
    parser.add_argument("--max-lidar-frames", type=int, default=800)
    parser.add_argument("--trajectory-time-padding-s", type=float, default=2.0)
    parser.add_argument("--keep-lowest-per-cell", type=int, default=5)
    parser.add_argument("--min-points-per-cell", type=int, default=3)
    parser.add_argument("--local-radius-m", type=float, default=10.0)
    parser.add_argument("--visible-distance-m", type=float, default=15.0)
    parser.add_argument("--visible-lateral-m", type=float, default=6.0)
    parser.add_argument("--seed-near-radius-m", type=float, default=1.6)
    parser.add_argument("--near-ground-percentile", type=float, default=20.0)
    parser.add_argument("--expected-ground-max-rise-m", type=float, default=0.14)
    parser.add_argument("--expected-ground-max-drop-m", type=float, default=0.22)
    parser.add_argument("--max-obstacle-height-m", type=float, default=0.35)
    parser.add_argument("--curb-height-diff-m", type=float, default=0.08)
    parser.add_argument("--same-level-height-tol-m", type=float, default=0.12)
    parser.add_argument("--neighbor-height-tol-m", type=float, default=0.08)
    parser.add_argument("--road-min-drop-m", type=float, default=0.04)
    parser.add_argument("--road-max-drop-m", type=float, default=0.30)
    parser.add_argument("--camera-height-m", type=float, default=0.445)
    parser.add_argument("--camera-offset-x-m", type=float, default=0.025)
    parser.add_argument("--camera-offset-y-m", type=float, default=0.0)
    parser.add_argument("--camera-roll-deg", type=float, default=0.0)
    parser.add_argument("--camera-pitch-deg", type=float, default=0.0)
    parser.add_argument("--camera-yaw-deg", type=float, default=0.0)
    parser.add_argument("--camera-hfov-deg", type=float, default=78.0)
    parser.add_argument("--lidar-height-m", type=float, default=0.525)
    parser.add_argument("--image-max-age-s", type=float, default=0.12)
    parser.add_argument("--imu-max-age-s", type=float, default=0.03)
    parser.add_argument("--min-seed-pixels", type=int, default=20)
    parser.add_argument("--feature-max-distance", type=float, default=0.22)
    parser.add_argument("--feature-margin", type=float, default=0.03)
    parser.add_argument("--prototype-bank-size", type=int, default=600)
    parser.add_argument("--score-sidewalk-local", type=float, default=2.0)
    parser.add_argument("--score-road-local", type=float, default=2.0)
    parser.add_argument("--score-curb-local", type=float, default=2.5)
    parser.add_argument("--score-obstacle-local", type=float, default=2.5)
    parser.add_argument("--score-camera", type=float, default=1.0)
    parser.add_argument("--preview-cell-size", type=int, default=4)
    parser.add_argument("--overlay-frames", type=int, default=6)
    parser.add_argument("--override-json", default="")
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_rosbag():
    try:
        import rosbag  # noqa: F401
    except Exception as exc:
        raise RuntimeError("rosbag import failed: %s" % exc)


def load_trajectory(csv_path, lidar_height_m):
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
        raise RuntimeError("No valid trajectory rows in %s" % csv_path)
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
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def transpose(m):
    return tuple(tuple(m[j][i] for j in range(3)) for i in range(3))


def matvec(m, v):
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def quat_to_rpy(qx, qy, qz, qw):
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def percentile(values, pct):
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    rank = max(0.0, min(100.0, pct)) / 100.0 * (len(s) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(s[lo])
    alpha = rank - lo
    return (1.0 - alpha) * float(s[lo]) + alpha * float(s[hi])


def xy_to_key(x, y, resolution):
    return int(math.floor(x / resolution)), int(math.floor(y / resolution))


def key_to_center(ix, iy, resolution):
    return (ix + 0.5) * resolution, (iy + 0.5) * resolution


def nearest_pose(rows, stamps, ts):
    idx = bisect.bisect_left(stamps, ts)
    if idx <= 0:
        return rows[0]
    if idx >= len(stamps):
        return rows[-1]
    prev_row = rows[idx - 1]
    next_row = rows[idx]
    if abs(prev_row["timestamp"] - ts) <= abs(next_row["timestamp"] - ts):
        return prev_row
    return next_row


def choose_nearest_sample(samples, ts, max_age_s):
    best = None
    best_age = None
    for sample in samples:
        age = abs(sample["timestamp"] - ts)
        if best_age is None or age < best_age:
            best = sample
            best_age = age
    if best is None or best_age is None or best_age > max_age_s:
        return None, None
    return best, best_age


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


def pixel_feature(image_bytes, step, u, v):
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
    out = []
    for col in cols:
        out.append(percentile(col, 50.0))
    return tuple(out)


def sample_feature_bank(feats, limit):
    if not feats:
        return []
    if len(feats) <= limit:
        return list(feats)
    step = max(1, len(feats) // limit)
    sampled = [feats[i] for i in range(0, len(feats), step)]
    return sampled[:limit]


def read_pointcloud_points(msg, point_stride):
    field_offsets = {f.name: f.offset for f in msg.fields}
    off_x = field_offsets["x"]
    off_y = field_offsets["y"]
    off_z = field_offsets["z"]
    step = msg.point_step
    data = memoryview(msg.data)
    total = msg.width * msg.height
    for i in range(0, total, max(1, point_stride)):
        base = i * step
        x = struct.unpack_from("<f", data, base + off_x)[0]
        y = struct.unpack_from("<f", data, base + off_y)[0]
        z = struct.unpack_from("<f", data, base + off_z)[0]
        yield x, y, z


def collect_local_cells(point_msg, imu_rpy, args):
    roll, pitch = imu_rpy
    body_to_level = matmul(rot_y(-pitch), rot_x(-roll))
    level_to_body = transpose(body_to_level)
    local_cells = {}
    near_ground_samples = []
    radius2 = args.local_radius_m * args.local_radius_m
    near_r2 = args.seed_near_radius_m * args.seed_near_radius_m
    for x, y, z in read_pointcloud_points(point_msg, args.point_stride):
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            continue
        p_level = matvec(body_to_level, (x, y, z))
        lx, ly, lz = p_level
        d2 = lx * lx + ly * ly
        if d2 > radius2:
            continue
        key = xy_to_key(lx, ly, args.grid_resolution)
        cell = local_cells.get(key)
        if cell is None:
            cell = LocalCellStats()
            local_cells[key] = cell
        cell.add(lz, args.keep_lowest_per_cell)
        if d2 <= near_r2:
            near_ground_samples.append(lz)
    near_ground = percentile(near_ground_samples, args.near_ground_percentile)
    expected_ground = -args.lidar_height_m
    if near_ground is None:
        near_ground = expected_ground
    if near_ground > expected_ground + args.expected_ground_max_rise_m:
        near_ground = expected_ground + args.expected_ground_max_rise_m
    elif near_ground < expected_ground - args.expected_ground_max_drop_m:
        near_ground = expected_ground - args.expected_ground_max_drop_m
    return local_cells, near_ground, level_to_body


def build_local_semantics(local_cells, ref_ground_z, args):
    cells = {}
    traversable = set()
    for key, stats in local_cells.items():
        if stats.count < args.min_points_per_cell:
            continue
        ground_z = stats.ground_z()
        if ground_z is None:
            continue
        max_z = stats.max_z if stats.max_z is not None else ground_z
        obstacle_h = max(0.0, float(max_z) - float(ground_z))
        is_obstacle = obstacle_h >= args.max_obstacle_height_m
        cells[key] = {"ground_z": float(ground_z), "obstacle_h": obstacle_h, "is_obstacle": is_obstacle}
        if not is_obstacle:
            traversable.add(key)

    curb_candidates = set()
    for key in traversable:
        z0 = cells[key]["ground_z"]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nk = (key[0] + dx, key[1] + dy)
            if nk not in traversable:
                continue
            if abs(z0 - cells[nk]["ground_z"]) >= args.curb_height_diff_m:
                curb_candidates.add(key)
                curb_candidates.add(nk)

    sidewalk_seed = set()
    for key in traversable:
        if abs(cells[key]["ground_z"] - ref_ground_z) <= args.same_level_height_tol_m:
            sidewalk_seed.add(key)

    sidewalk = set(sidewalk_seed)
    visited = set(sidewalk_seed)
    q = deque(sidewalk_seed)
    while q:
        key = q.popleft()
        z0 = cells[key]["ground_z"]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nk = (key[0] + dx, key[1] + dy)
            if nk in visited:
                continue
            visited.add(nk)
            if nk not in traversable or nk in curb_candidates:
                continue
            nz = cells[nk]["ground_z"]
            if abs(nz - ref_ground_z) > args.same_level_height_tol_m:
                continue
            if abs(nz - z0) > args.neighbor_height_tol_m:
                continue
            sidewalk.add(nk)
            q.append(nk)

    road_seed = set()
    for key in traversable:
        if key in sidewalk or key in curb_candidates:
            continue
        drop = ref_ground_z - cells[key]["ground_z"]
        if args.road_min_drop_m <= drop <= args.road_max_drop_m:
            road_seed.add(key)

    road = set(road_seed)
    rq = deque(road_seed)
    rvisited = set(road_seed)
    while rq:
        key = rq.popleft()
        z0 = cells[key]["ground_z"]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nk = (key[0] + dx, key[1] + dy)
            if nk in rvisited:
                continue
            rvisited.add(nk)
            if nk not in traversable or nk in sidewalk or nk in curb_candidates:
                continue
            nz = cells[nk]["ground_z"]
            if abs(nz - z0) > args.neighbor_height_tol_m:
                continue
            if nz >= ref_ground_z - args.road_min_drop_m:
                continue
            road.add(nk)
            rq.append(nk)

    curb = set()
    for key in curb_candidates:
        near_side = False
        near_road = False
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nk = (key[0] + dx, key[1] + dy)
            if nk in sidewalk:
                near_side = True
            if nk in road:
                near_road = True
            if near_side and near_road:
                curb.add(key)
                break

    return cells, {"sidewalk": sidewalk, "road": road, "curb": curb, "traversable": traversable}


def collect_camera_votes(
    local_classes,
    local_cells,
    ref_ground_z,
    image_msg,
    level_to_body,
    args,
    fallback_sidewalk_proto=None,
    fallback_road_proto=None,
):
    body_to_optical = make_body_to_optical(
        deg2rad(args.camera_roll_deg),
        deg2rad(args.camera_pitch_deg),
        deg2rad(args.camera_yaw_deg),
    )
    intrinsics = compute_intrinsics(image_msg.width, image_msg.height, args.camera_hfov_deg)
    fx, fy, cx, cy = intrinsics
    cam_offset = (
        args.camera_offset_x_m,
        args.camera_offset_y_m,
        args.camera_height_m - args.lidar_height_m,
    )
    image_bytes = image_msg.data
    step = image_msg.step
    width = image_msg.width
    height = image_msg.height
    sidewalk_feats = []
    road_feats = []
    visible = []
    for key in local_classes["traversable"]:
        lx, ly = key_to_center(key[0], key[1], args.grid_resolution)
        lz = local_cells[key]["ground_z"]
        point_body = matvec(level_to_body, (lx, ly, lz))
        px = point_body[0] - cam_offset[0]
        py = point_body[1] - cam_offset[1]
        pz = point_body[2] - cam_offset[2]
        point_opt = matvec(body_to_optical, (px, py, pz))
        if point_opt[2] <= 0.2:
            continue
        u = int(round(fx * (point_opt[0] / point_opt[2]) + cx))
        v = int(round(fy * (point_opt[1] / point_opt[2]) + cy))
        if u < 0 or u >= width or v < 0 or v >= height:
            continue
        if lx < 0.5 or lx > args.visible_distance_m or abs(ly) > args.visible_lateral_m:
            continue
        feat = pixel_feature(image_bytes, step, u, v)
        visible.append((key, u, v, feat))
        if key in local_classes["sidewalk"]:
            sidewalk_feats.append(feat)
        elif key in local_classes["road"]:
            road_feats.append(feat)
    sidewalk_proto = component_median(sidewalk_feats) if len(sidewalk_feats) >= args.min_seed_pixels else fallback_sidewalk_proto
    road_proto = component_median(road_feats) if len(road_feats) >= args.min_seed_pixels else fallback_road_proto
    if sidewalk_proto is None or road_proto is None:
        return {}, None, sidewalk_feats, road_feats
    votes = {}
    overlay = bytearray(image_bytes)
    for key, u, v, feat in visible:
        d_sw = feature_distance(feat, sidewalk_proto)
        d_rd = feature_distance(feat, road_proto)
        if min(d_sw, d_rd) > args.feature_max_distance:
            continue
        if abs(d_sw - d_rd) < args.feature_margin:
            continue
        if d_sw < d_rd:
            votes[key] = "sidewalk"
            color = (210, 230, 210)
        else:
            votes[key] = "road"
            color = (70, 120, 200)
        for yy in range(max(0, v - 1), min(height, v + 2)):
            for xx in range(max(0, u - 1), min(width, u + 2)):
                idx = yy * step + xx * 3
                overlay[idx] = color[0]
                overlay[idx + 1] = color[1]
                overlay[idx + 2] = color[2]
    return votes, {"width": width, "height": height, "rgb": bytes(overlay)}, sidewalk_feats, road_feats


def accumulate_world_scores(scores, local_cells, local_classes, camera_votes, pose_row, ref_ground_z, args):
    cy = math.cos(pose_row["yaw"])
    sy = math.sin(pose_row["yaw"])
    for key, cell in local_cells.items():
        lx, ly = key_to_center(key[0], key[1], args.grid_resolution)
        wx = pose_row["x"] + cy * lx - sy * ly
        wy = pose_row["y"] + sy * lx + cy * ly
        wkey = xy_to_key(wx, wy, args.grid_resolution)
        entry = scores[wkey]
        entry["observed"] += 1.0
        entry["ground_z_sum"] += pose_row["z"] + cell["ground_z"]
        entry["ground_z_count"] += 1.0
        if cell["is_obstacle"]:
            entry["obstacle"] += args.score_obstacle_local
            continue
        if key in local_classes["curb"]:
            entry["curb"] += args.score_curb_local
        if key in local_classes["sidewalk"]:
            entry["sidewalk"] += args.score_sidewalk_local
        elif key in local_classes["road"]:
            entry["road"] += args.score_road_local
        if key in camera_votes:
            entry[camera_votes[key]] += args.score_camera


def write_ascii_pcd(path, keys, scores, resolution):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write("WIDTH %d\n" % len(keys))
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write("POINTS %d\n" % len(keys))
        f.write("DATA ascii\n")
        for ix, iy in sorted(keys):
            x, y = key_to_center(ix, iy, resolution)
            entry = scores[(ix, iy)]
            z = entry["ground_z_sum"] / max(1.0, entry["ground_z_count"])
            f.write("%.4f %.4f %.4f\n" % (x, y, z))


def rgb_to_pcd_float(rgb):
    packed = (int(rgb[0]) << 16) | (int(rgb[1]) << 8) | int(rgb[2])
    return struct.unpack("f", struct.pack("I", packed))[0]


def semantic_color_for_key(key, classes):
    if key in classes["curb"]:
        return (220, 70, 60)
    if key in classes["sidewalk"]:
        return (210, 230, 210)
    if key in classes["road"]:
        return (70, 120, 200)
    return (70, 72, 78)


def semantic_label_for_key(key, classes):
    if key in classes["curb"]:
        return "curb"
    if key in classes["sidewalk"]:
        return "sidewalk"
    if key in classes["road"]:
        return "road"
    return "observed"


def write_rgb_semantic_pcd(path, keys, scores, resolution, classes):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z rgb\n")
        f.write("SIZE 4 4 4 4\n")
        f.write("TYPE F F F F\n")
        f.write("COUNT 1 1 1 1\n")
        f.write("WIDTH %d\n" % len(keys))
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write("POINTS %d\n" % len(keys))
        f.write("DATA ascii\n")
        for ix, iy in sorted(keys):
            x, y = key_to_center(ix, iy, resolution)
            entry = scores[(ix, iy)]
            z = entry["ground_z_sum"] / max(1.0, entry["ground_z_count"])
            rgb = rgb_to_pcd_float(semantic_color_for_key((ix, iy), classes))
            f.write("%.4f %.4f %.4f %.8e\n" % (x, y, z, rgb))


def write_png(path, width, height, rgb_data):
    SEMANTIC_MAP_MODULE.write_png(path, width, height, rgb_data)


def write_preview_png(path, classes, scores, resolution, cell_px):
    keys = set(scores.keys())
    if not keys:
        raise RuntimeError("No accumulated world cells to preview")
    min_ix = min(ix for ix, _ in keys)
    max_ix = max(ix for ix, _ in keys)
    min_iy = min(iy for _, iy in keys)
    max_iy = max(iy for _, iy in keys)
    width = (max_ix - min_ix + 1) * cell_px
    height = (max_iy - min_iy + 1) * cell_px
    bg = bytearray([24, 24, 28]) * (width * height)

    def paint(key, rgb):
        ix, iy = key
        px0 = (ix - min_ix) * cell_px
        py0 = (max_iy - iy) * cell_px
        for py in range(py0, py0 + cell_px):
            row = py * width
            for px in range(px0, px0 + cell_px):
                idx = (row + px) * 3
                bg[idx:idx + 3] = bytes(rgb)

    for key in keys:
        paint(key, (70, 72, 78))
    for key in classes["road"]:
        paint(key, (70, 120, 200))
    for key in classes["sidewalk"]:
        paint(key, (210, 230, 210))
    for key in classes["curb"]:
        paint(key, (220, 70, 60))
    write_png(path, width, height, bytes(bg))


def copy_classes(classes):
    return {label: set(keys) for label, keys in classes.items()}


def normalize_cell_key(cell):
    if not isinstance(cell, (list, tuple)) or len(cell) != 2:
        return None
    try:
        return int(cell[0]), int(cell[1])
    except Exception:
        return None


def rectangle_to_keys(rect, resolution):
    if not isinstance(rect, dict):
        return set()
    if all(k in rect for k in ("ix_min", "ix_max", "iy_min", "iy_max")):
        ix_min = int(rect["ix_min"])
        ix_max = int(rect["ix_max"])
        iy_min = int(rect["iy_min"])
        iy_max = int(rect["iy_max"])
    elif all(k in rect for k in ("x_min", "x_max", "y_min", "y_max")):
        ix_min = int(math.floor(float(rect["x_min"]) / resolution))
        ix_max = int(math.floor(float(rect["x_max"]) / resolution))
        iy_min = int(math.floor(float(rect["y_min"]) / resolution))
        iy_max = int(math.floor(float(rect["y_max"]) / resolution))
    else:
        return set()
    if ix_min > ix_max:
        ix_min, ix_max = ix_max, ix_min
    if iy_min > iy_max:
        iy_min, iy_max = iy_max, iy_min
    keys = set()
    for ix in range(ix_min, ix_max + 1):
        for iy in range(iy_min, iy_max + 1):
            keys.add((ix, iy))
    return keys


def clear_keys_from_classes(classes, keys):
    for label in ("sidewalk", "road", "curb"):
        classes[label].difference_update(keys)


def apply_override_json(classes, override_path, resolution):
    if not override_path or not os.path.isfile(override_path):
        return classes, None
    with open(override_path, "r", encoding="utf-8") as f:
        override = json.load(f)
    updated = copy_classes(classes)
    erase_keys = set()
    for cell in override.get("erase_cells", []):
        key = normalize_cell_key(cell)
        if key is not None:
            erase_keys.add(key)
    for rect in override.get("erase_rectangles", []):
        erase_keys.update(rectangle_to_keys(rect, resolution))
    clear_keys_from_classes(updated, erase_keys)

    paint_labels = override.get("paint", {})
    for label in ("sidewalk", "road", "curb"):
        payload = paint_labels.get(label, {})
        paint_keys = set()
        for cell in payload.get("cells", []):
            key = normalize_cell_key(cell)
            if key is not None:
                paint_keys.add(key)
        for rect in payload.get("rectangles", []):
            paint_keys.update(rectangle_to_keys(rect, resolution))
        clear_keys_from_classes(updated, paint_keys)
        updated[label].update(paint_keys)
    return updated, override


def finalize_classes(scores):
    sidewalk = set()
    road = set()
    curb = set()
    for key, entry in scores.items():
        if entry["observed"] < 1.0:
            continue
        if entry["obstacle"] >= max(entry["sidewalk"], entry["road"], entry["curb"]) and entry["obstacle"] >= 2.0:
            continue
        if entry["curb"] >= max(entry["sidewalk"], entry["road"]) and entry["curb"] >= 2.0:
            curb.add(key)
            continue
        if entry["sidewalk"] >= entry["road"] + 1.0 and entry["sidewalk"] >= 2.0:
            sidewalk.add(key)
        elif entry["road"] >= entry["sidewalk"] + 1.0 and entry["road"] >= 2.0:
            road.add(key)
    return {"sidewalk": sidewalk, "road": road, "curb": curb}


def sorted_cell_lists(classes):
    return {
        label: [[ix, iy] for ix, iy in sorted(keys)]
        for label, keys in classes.items()
    }


def write_editable_state(path, classes, scores, args, source_override_path):
    keys = set(scores.keys())
    if keys:
        min_ix = min(ix for ix, _ in keys)
        max_ix = max(ix for ix, _ in keys)
        min_iy = min(iy for _, iy in keys)
        max_iy = max(iy for _, iy in keys)
        x_min, y_min = key_to_center(min_ix, min_iy, args.grid_resolution)
        x_max, y_max = key_to_center(max_ix, max_iy, args.grid_resolution)
    else:
        min_ix = max_ix = min_iy = max_iy = 0
        x_min = x_max = y_min = y_max = 0.0
    payload = {
        "meta": {
            "bag": args.bag,
            "grid_resolution": args.grid_resolution,
            "labels": ["sidewalk", "road", "curb", "observed"],
            "bounds_index": {
                "ix_min": min_ix,
                "ix_max": max_ix,
                "iy_min": min_iy,
                "iy_max": max_iy,
            },
            "bounds_metric": {
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
            },
            "source_override_json": source_override_path or "",
        },
        "classes": sorted_cell_lists(classes),
        "observed_cells": [
            [
                ix,
                iy,
                float(scores[(ix, iy)]["ground_z_sum"] / max(1.0, scores[(ix, iy)]["ground_z_count"])),
                semantic_label_for_key((ix, iy), classes),
            ]
            for ix, iy in sorted(keys)
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_override_template(path):
    payload = {
        "description": "Edit semantic classes manually. Cells are [ix, iy]. Rectangles can use grid indices or metric bounds.",
        "paint": {
            "sidewalk": {
                "cells": [],
                "rectangles": [
                    {
                        "x_min": 0.0,
                        "x_max": 0.0,
                        "y_min": 0.0,
                        "y_max": 0.0,
                    }
                ],
            },
            "road": {
                "cells": [],
                "rectangles": [],
            },
            "curb": {
                "cells": [],
                "rectangles": [],
            },
        },
        "erase_cells": [],
        "erase_rectangles": [],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    global SEMANTIC_MAP_MODULE
    args = parse_args()
    ensure_dir(args.output_dir)
    ensure_rosbag()
    SEMANTIC_MAP_MODULE = load_module(
        "/home/byeongjae/code/Modular_Approach_Autonomous_Driving-/src/package_all/aimlab/lio-sam/scripts/build_sidewalk_semantic_map.py",
        "build_sidewalk_semantic_map_runtime",
    )

    bundle_trajectory = os.path.join(args.bundle_dir, "trajectory.csv")
    trajectory_rows = load_trajectory(bundle_trajectory, args.lidar_height_m)
    stamps = [row["timestamp"] for row in trajectory_rows]
    trajectory_start_ts = stamps[0] - max(0.0, args.trajectory_time_padding_s)
    trajectory_end_ts = stamps[-1] + max(0.0, args.trajectory_time_padding_s)

    import rosbag
    bag = rosbag.Bag(args.bag)
    recent_images = deque(maxlen=8)
    recent_imus = deque(maxlen=256)
    lidar_index = 0
    used_frames = 0
    scores = defaultdict(lambda: {
        "sidewalk": 0.0,
        "road": 0.0,
        "curb": 0.0,
        "obstacle": 0.0,
        "observed": 0.0,
        "ground_z_sum": 0.0,
        "ground_z_count": 0.0,
    })
    overlays = []
    frame_reports = []
    sidewalk_proto_bank = deque(maxlen=max(1, args.prototype_bank_size))
    road_proto_bank = deque(maxlen=max(1, args.prototype_bank_size))

    for topic, msg, bag_time in bag.read_messages(topics=[args.image_topic, args.imu_topic, args.point_topic]):
        bag_ts = bag_time.to_sec()
        if bag_ts < trajectory_start_ts:
            continue
        if bag_ts > trajectory_end_ts:
            break
        if topic == args.image_topic:
            recent_images.append({
                "timestamp": msg.header.stamp.to_sec(),
                "msg": msg,
            })
            continue
        if topic == args.imu_topic:
            recent_imus.append({
                "timestamp": msg.header.stamp.to_sec(),
                "rpy": quat_to_rpy(
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                    msg.orientation.w,
                ),
            })
            continue
        if topic != args.point_topic:
            continue
        lidar_index += 1
        if lidar_index % max(1, args.lidar_frame_stride) != 0:
            continue
        if used_frames >= args.max_lidar_frames:
            break
        ts = msg.header.stamp.to_sec()
        pose_row = nearest_pose(trajectory_rows, stamps, ts)
        imu_sample, imu_age = choose_nearest_sample(recent_imus, ts, args.imu_max_age_s)
        if imu_sample is None:
            imu_roll = pose_row["roll"]
            imu_pitch = pose_row["pitch"]
            imu_ok = False
            imu_age = None
        else:
            imu_roll, imu_pitch, _imu_yaw = imu_sample["rpy"]
            imu_ok = True

        local_cells, ref_ground_z, level_to_body = collect_local_cells(msg, (imu_roll, imu_pitch), args)
        cells, local_classes = build_local_semantics(local_cells, ref_ground_z, args)
        camera_votes = {}
        overlay = None
        image_sample, image_age = choose_nearest_sample(recent_images, ts, args.image_max_age_s)
        image_ok = image_sample is not None
        seed_sidewalk_count = 0
        seed_road_count = 0
        if image_ok:
            fallback_sidewalk_proto = component_median(sidewalk_proto_bank) if len(sidewalk_proto_bank) >= args.min_seed_pixels else None
            fallback_road_proto = component_median(road_proto_bank) if len(road_proto_bank) >= args.min_seed_pixels else None
            camera_votes, overlay, sidewalk_feats, road_feats = collect_camera_votes(
                local_classes,
                cells,
                ref_ground_z,
                image_sample["msg"],
                level_to_body,
                args,
                fallback_sidewalk_proto=fallback_sidewalk_proto,
                fallback_road_proto=fallback_road_proto,
            )
            seed_sidewalk_count = len(sidewalk_feats)
            seed_road_count = len(road_feats)
            for feat in sample_feature_bank(sidewalk_feats, 24):
                sidewalk_proto_bank.append(feat)
            for feat in sample_feature_bank(road_feats, 24):
                road_proto_bank.append(feat)
            if overlay is not None and len(overlays) < args.overlay_frames:
                overlay_path = os.path.join(args.output_dir, "frame_overlay_%03d.png" % used_frames)
                write_png(overlay_path, overlay["width"], overlay["height"], overlay["rgb"])
                overlays.append(overlay_path)

        accumulate_world_scores(scores, cells, local_classes, camera_votes, pose_row, ref_ground_z, args)
        frame_reports.append(
            {
                "frame_index": used_frames,
                "lidar_timestamp": ts,
                "imu_ok": imu_ok,
                "imu_age_s": imu_age,
                "image_ok": image_ok,
                "image_age_s": image_age,
                "local_cells": len(cells),
                "local_sidewalk": len(local_classes["sidewalk"]),
                "local_road": len(local_classes["road"]),
                "local_curb": len(local_classes["curb"]),
                "camera_seed_sidewalk": seed_sidewalk_count,
                "camera_seed_road": seed_road_count,
                "camera_votes": len(camera_votes),
                "ref_ground_z": ref_ground_z,
            }
        )
        used_frames += 1

    bag.close()

    final_classes = finalize_classes(scores)
    sidewalk_pcd = os.path.join(args.output_dir, "SidewalkMap.raw_bev.pcd")
    road_pcd = os.path.join(args.output_dir, "RoadMap.raw_bev.pcd")
    curb_pcd = os.path.join(args.output_dir, "CurbMap.raw_bev.pcd")
    observed_pcd = os.path.join(args.output_dir, "ObservedMap.raw_bev.pcd")
    semantic_rgb_pcd = os.path.join(args.output_dir, "SemanticObservedMap.raw_bev.pcd")
    preview_png = os.path.join(args.output_dir, "raw_semantic_bev_preview.png")
    manifest_json = os.path.join(args.output_dir, "raw_semantic_bev_manifest.json")
    editable_state_json = os.path.join(args.output_dir, "raw_semantic_bev_state.json")
    override_template_json = os.path.join(args.output_dir, "raw_semantic_bev_override.template.json")

    final_classes, applied_override = apply_override_json(final_classes, args.override_json, args.grid_resolution)

    write_ascii_pcd(sidewalk_pcd, final_classes["sidewalk"], scores, args.grid_resolution)
    write_ascii_pcd(road_pcd, final_classes["road"], scores, args.grid_resolution)
    write_ascii_pcd(curb_pcd, final_classes["curb"], scores, args.grid_resolution)
    write_ascii_pcd(observed_pcd, set(scores.keys()), scores, args.grid_resolution)
    write_rgb_semantic_pcd(semantic_rgb_pcd, set(scores.keys()), scores, args.grid_resolution, final_classes)
    write_preview_png(preview_png, final_classes, scores, args.grid_resolution, args.preview_cell_size)
    write_editable_state(editable_state_json, final_classes, scores, args, args.override_json)
    write_override_template(override_template_json)

    manifest = {
        "bag": args.bag,
        "trajectory_csv": bundle_trajectory,
        "used_lidar_frames": used_frames,
        "cells_total": len(scores),
        "cells_sidewalk": len(final_classes["sidewalk"]),
        "cells_road": len(final_classes["road"]),
        "cells_curb": len(final_classes["curb"]),
        "params": {
            "grid_resolution": args.grid_resolution,
            "lidar_frame_stride": args.lidar_frame_stride,
            "point_stride": args.point_stride,
            "lidar_height_m": args.lidar_height_m,
            "camera_height_m": args.camera_height_m,
            "camera_offset_x_m": args.camera_offset_x_m,
            "camera_hfov_deg": args.camera_hfov_deg,
            "override_json": args.override_json,
        },
        "frame_reports": frame_reports,
        "outputs": {
            "sidewalk_pcd": sidewalk_pcd,
            "road_pcd": road_pcd,
            "curb_pcd": curb_pcd,
            "observed_pcd": observed_pcd,
            "semantic_rgb_pcd": semantic_rgb_pcd,
            "preview_png": preview_png,
            "editable_state_json": editable_state_json,
            "override_template_json": override_template_json,
            "overlay_frames": overlays,
        },
        "override_applied": applied_override is not None,
    }
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("raw semantic bev complete")
    print("  used_lidar_frames :", used_frames)
    print("  cells_total       :", len(scores))
    print("  cells_sidewalk    :", len(final_classes["sidewalk"]))
    print("  cells_road        :", len(final_classes["road"]))
    print("  cells_curb        :", len(final_classes["curb"]))
    print("  observed_pcd      :", observed_pcd)
    print("  semantic_rgb_pcd  :", semantic_rgb_pcd)
    print("  preview           :", preview_png)
    print("  editable_state    :", editable_state_json)
    print("  override_template :", override_template_json)
    print("  manifest          :", manifest_json)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
