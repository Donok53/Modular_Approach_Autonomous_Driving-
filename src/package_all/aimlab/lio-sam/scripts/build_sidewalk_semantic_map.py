#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import bisect
import binascii
import csv
import json
import math
import os
import struct
from collections import deque
import zlib


class CellStats:
    __slots__ = ("count", "max_z", "low_z")

    def __init__(self):
        self.count = 0
        self.max_z = None
        self.low_z = []

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
    default_bundle = os.path.join(repo_root, "src/package_all/monitoring_delivery/latest")
    parser = argparse.ArgumentParser(
        description="Classify sidewalk / road / curb candidates from a bag-derived LIO-SAM map bundle."
    )
    parser.add_argument("--bundle-dir", default=default_bundle)
    parser.add_argument("--map-pcd", default="")
    parser.add_argument("--trajectory-csv", default="")
    parser.add_argument("--source-bag", default="")
    parser.add_argument("--output-dir", default=os.path.join(repo_root, "generated/sidewalk_semantic_map"))
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
    parser.add_argument("--preview-cell-size", type=int, default=4)
    return parser.parse_args()


def resolve_input_paths(args):
    map_pcd = args.map_pcd or os.path.join(args.bundle_dir, "GlobalMap.pcd")
    trajectory_csv = args.trajectory_csv or os.path.join(args.bundle_dir, "trajectory.csv")
    if not os.path.isfile(map_pcd):
        raise FileNotFoundError("GlobalMap.pcd not found: %s" % map_pcd)
    if not os.path.isfile(trajectory_csv):
        raise FileNotFoundError("trajectory.csv not found: %s" % trajectory_csv)
    return map_pcd, trajectory_csv


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
            yaw = float(row.get("yaw", 0.0) or 0.0)
            rows.append(
                {
                    "x": x,
                    "y": y,
                    "z": z,
                    "yaw": yaw,
                    "ground_z": z - lidar_height_m,
                    "timestamp": float(row.get("timestamp", 0.0) or 0.0),
                }
            )
    if not rows:
        raise RuntimeError("No valid trajectory rows loaded from %s" % csv_path)
    return rows


def trajectory_bounds(rows, margin_m):
    xs = [r["x"] for r in rows]
    ys = [r["y"] for r in rows]
    return (
        min(xs) - margin_m,
        min(ys) - margin_m,
        max(xs) + margin_m,
        max(ys) + margin_m,
    )


def xy_to_key(x, y, resolution):
    return int(math.floor(x / resolution)), int(math.floor(y / resolution))


def key_to_center(ix, iy, resolution):
    return (ix + 0.5) * resolution, (iy + 0.5) * resolution


def read_pcd_header(fp):
    header_lines = []
    while True:
        line = fp.readline()
        if not line:
            raise RuntimeError("Unexpected EOF while reading PCD header")
        text = line.decode("ascii", errors="ignore").strip()
        header_lines.append(text)
        if text.startswith("DATA "):
            break
    header = {}
    for line in header_lines:
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        key = parts[0].upper()
        vals = parts[1:]
        header[key] = vals
    if header.get("DATA", [""])[0].lower() != "binary":
        raise RuntimeError("Only binary PCD is supported")
    fields = header.get("FIELDS", [])
    sizes = [int(v) for v in header.get("SIZE", [])]
    counts = [int(v) for v in header.get("COUNT", [])]
    if not counts:
        counts = [1] * len(fields)
    point_step = 0
    offsets = {}
    for name, size, count in zip(fields, sizes, counts):
        offsets[name] = point_step
        point_step += size * count
    for name in ("x", "y", "z"):
        if name not in offsets:
            raise RuntimeError("PCD missing field '%s'" % name)
    points = int(header.get("POINTS", ["0"])[0])
    return {
        "offsets": offsets,
        "point_step": point_step,
        "points": points,
    }


def accumulate_cells_from_pcd(map_pcd, bounds_xy, resolution, keep_lowest_n):
    min_x, min_y, max_x, max_y = bounds_xy
    cell_stats = {}
    total_points = 0
    kept_points = 0
    with open(map_pcd, "rb") as fp:
        meta = read_pcd_header(fp)
        point_step = meta["point_step"]
        off_x = meta["offsets"]["x"]
        off_y = meta["offsets"]["y"]
        off_z = meta["offsets"]["z"]
        chunk_bytes = point_step * 16384
        while True:
            blob = fp.read(chunk_bytes)
            if not blob:
                break
            usable = len(blob) - (len(blob) % point_step)
            if usable <= 0:
                continue
            blob = memoryview(blob[:usable])
            total_points += usable // point_step
            for idx in range(0, usable, point_step):
                x = struct.unpack_from("<f", blob, idx + off_x)[0]
                y = struct.unpack_from("<f", blob, idx + off_y)[0]
                z = struct.unpack_from("<f", blob, idx + off_z)[0]
                if x < min_x or x > max_x or y < min_y or y > max_y:
                    continue
                key = xy_to_key(x, y, resolution)
                stats = cell_stats.get(key)
                if stats is None:
                    stats = CellStats()
                    cell_stats[key] = stats
                stats.add(z, keep_lowest_n)
                kept_points += 1
    return cell_stats, total_points, kept_points


def build_cell_map(cell_stats, min_points_per_cell, max_obstacle_height_m):
    cells = {}
    for key, stats in cell_stats.items():
        if stats.count < min_points_per_cell:
            continue
        ground_z = stats.ground_z()
        if ground_z is None:
            continue
        max_z = stats.max_z if stats.max_z is not None else ground_z
        obstacle_h = max(0.0, float(max_z) - float(ground_z))
        cells[key] = {
            "ground_z": float(ground_z),
            "obstacle_h": obstacle_h,
            "is_obstacle": obstacle_h >= max_obstacle_height_m,
            "support": int(stats.count),
        }
    return cells


def nearest_trajectory_point(x, y, trajectory_rows, cache, resolution):
    cache_key = xy_to_key(x, y, resolution)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    best = None
    best_d2 = float("inf")
    for row in trajectory_rows:
        dx = x - row["x"]
        dy = y - row["y"]
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best = row
    cache[cache_key] = (best, math.sqrt(best_d2))
    return cache[cache_key]


def classify_semantics(cells, trajectory_rows, resolution, args):
    traj_cache = {}
    sidewalk_seed = set()
    curb_candidates = set()
    observed_traversable = set()

    for key, info in cells.items():
        if not info["is_obstacle"]:
            observed_traversable.add(key)

    for key, info in cells.items():
        x, y = key_to_center(key[0], key[1], resolution)
        traj_row, dist = nearest_trajectory_point(x, y, trajectory_rows, traj_cache, resolution)
        if dist <= args.traj_seed_radius_m and (not info["is_obstacle"]):
            if abs(info["ground_z"] - traj_row["ground_z"]) <= args.traj_seed_height_tol_m:
                sidewalk_seed.add(key)

    neighbor4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for key, info in cells.items():
        if info["is_obstacle"]:
            continue
        z0 = info["ground_z"]
        for dx, dy in neighbor4:
            nk = (key[0] + dx, key[1] + dy)
            ninfo = cells.get(nk)
            if ninfo is None or ninfo["is_obstacle"]:
                continue
            if abs(z0 - ninfo["ground_z"]) >= args.curb_height_diff_m:
                curb_candidates.add(key)
                curb_candidates.add(nk)

    sidewalk = set(sidewalk_seed)
    q = deque(sidewalk_seed)
    visited = set(sidewalk_seed)
    while q:
        key = q.popleft()
        info = cells[key]
        x, y = key_to_center(key[0], key[1], resolution)
        traj_row, dist_here = nearest_trajectory_point(x, y, trajectory_rows, traj_cache, resolution)
        base_ground = traj_row["ground_z"]
        for dx, dy in (
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ):
            nk = (key[0] + dx, key[1] + dy)
            if nk in visited:
                continue
            visited.add(nk)
            ninfo = cells.get(nk)
            if ninfo is None or ninfo["is_obstacle"]:
                continue
            nx, ny = key_to_center(nk[0], nk[1], resolution)
            _traj_row, ndist = nearest_trajectory_point(nx, ny, trajectory_rows, traj_cache, resolution)
            if ndist > args.sidewalk_fill_max_distance_m:
                continue
            if abs(ninfo["ground_z"] - base_ground) > args.sidewalk_fill_height_tol_m:
                continue
            if abs(ninfo["ground_z"] - info["ground_z"]) > args.neighbor_height_tol_m:
                continue
            sidewalk.add(nk)
            q.append(nk)

    road_seed = set()
    for key, info in cells.items():
        if key in sidewalk or info["is_obstacle"]:
            continue
        x, y = key_to_center(key[0], key[1], resolution)
        traj_row, dist = nearest_trajectory_point(x, y, trajectory_rows, traj_cache, resolution)
        if dist > (args.sidewalk_fill_max_distance_m + 2.0):
            continue
        drop = traj_row["ground_z"] - info["ground_z"]
        if args.road_min_drop_from_sidewalk_m <= drop <= args.road_max_drop_from_sidewalk_m:
            road_seed.add(key)

    road = set(road_seed)
    rq = deque(road_seed)
    road_visited = set(road_seed)
    while rq:
        key = rq.popleft()
        info = cells[key]
        x, y = key_to_center(key[0], key[1], resolution)
        traj_row, dist_here = nearest_trajectory_point(x, y, trajectory_rows, traj_cache, resolution)
        base_ground = info["ground_z"]
        for dx, dy in (
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ):
            nk = (key[0] + dx, key[1] + dy)
            if nk in road_visited:
                continue
            road_visited.add(nk)
            ninfo = cells.get(nk)
            if ninfo is None or ninfo["is_obstacle"] or nk in sidewalk:
                continue
            nx, ny = key_to_center(nk[0], nk[1], resolution)
            _traj_row, ndist = nearest_trajectory_point(nx, ny, trajectory_rows, traj_cache, resolution)
            if ndist > (args.sidewalk_fill_max_distance_m + 3.0):
                continue
            if abs(ninfo["ground_z"] - base_ground) > args.neighbor_height_tol_m:
                continue
            if ninfo["ground_z"] >= traj_row["ground_z"] - args.road_min_drop_from_sidewalk_m:
                continue
            road.add(nk)
            rq.append(nk)

    curb = set()
    for key in curb_candidates:
        near_sidewalk = False
        near_road = False
        for dx, dy in (
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ):
            nk = (key[0] + dx, key[1] + dy)
            if nk in sidewalk:
                near_sidewalk = True
            if nk in road:
                near_road = True
            if near_sidewalk and near_road:
                curb.add(key)
                break

    return {
        "sidewalk": sidewalk,
        "road": road,
        "curb": curb,
        "observed_traversable": observed_traversable,
    }


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_ascii_pcd(path, keys, cells, resolution):
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
            z = cells[(ix, iy)]["ground_z"]
            f.write("%.4f %.4f %.4f\n" % (x, y, z))


def write_png(path, width, height, rgb_data):
    def chunk(tag, payload):
        return (
            struct.pack("!I", len(payload))
            + tag
            + payload
            + struct.pack("!I", binascii.crc32(tag + payload) & 0xFFFFFFFF)
        )

    raw = bytearray()
    row_bytes = width * 3
    for y in range(height):
        raw.append(0)
        start = y * row_bytes
        raw.extend(rgb_data[start:start + row_bytes])

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(bytes(raw), 9)))
        f.write(chunk(b"IEND", b""))


def write_preview_png(path, classes, cells, trajectory_rows, resolution, cell_px):
    keys = set(cells.keys())
    if not keys:
        raise RuntimeError("No cells to render")
    min_ix = min(ix for ix, _ in keys)
    max_ix = max(ix for ix, _ in keys)
    min_iy = min(iy for _, iy in keys)
    max_iy = max(iy for _, iy in keys)
    width = (max_ix - min_ix + 1) * cell_px
    height = (max_iy - min_iy + 1) * cell_px

    bg = bytearray([24, 24, 28]) * (width * height)

    def paint_cell(key, rgb):
        ix, iy = key
        px0 = (ix - min_ix) * cell_px
        py0 = (max_iy - iy) * cell_px
        for py in range(py0, py0 + cell_px):
            row = py * width
            for px in range(px0, px0 + cell_px):
                idx = (row + px) * 3
                bg[idx: idx + 3] = bytes(rgb)

    for key in keys:
        paint_cell(key, (70, 72, 78))
    for key in classes["road"]:
        paint_cell(key, (70, 120, 200))
    for key in classes["sidewalk"]:
        paint_cell(key, (210, 230, 210))
    for key in classes["curb"]:
        paint_cell(key, (220, 70, 60))

    for row in trajectory_rows:
        key = xy_to_key(row["x"], row["y"], resolution)
        if key in keys:
            paint_cell(key, (250, 220, 40))

    write_png(path, width, height, bytes(bg))


def main():
    args = parse_args()
    map_pcd, trajectory_csv = resolve_input_paths(args)
    ensure_dir(args.output_dir)

    trajectory_rows = load_trajectory(trajectory_csv, args.lidar_height_m)
    bounds_xy = trajectory_bounds(trajectory_rows, args.map_margin_m)
    cell_stats, total_points, kept_points = accumulate_cells_from_pcd(
        map_pcd,
        bounds_xy,
        args.grid_resolution,
        args.keep_lowest_per_cell,
    )
    cells = build_cell_map(cell_stats, args.min_points_per_cell, args.max_obstacle_height_m)
    classes = classify_semantics(cells, trajectory_rows, args.grid_resolution, args)

    sidewalk_pcd = os.path.join(args.output_dir, "SidewalkMap.pcd")
    road_pcd = os.path.join(args.output_dir, "RoadMap.pcd")
    curb_pcd = os.path.join(args.output_dir, "CurbMap.pcd")
    preview_png = os.path.join(args.output_dir, "sidewalk_semantic_preview.png")
    manifest_json = os.path.join(args.output_dir, "sidewalk_semantic_manifest.json")

    write_ascii_pcd(sidewalk_pcd, classes["sidewalk"], cells, args.grid_resolution)
    write_ascii_pcd(road_pcd, classes["road"], cells, args.grid_resolution)
    write_ascii_pcd(curb_pcd, classes["curb"], cells, args.grid_resolution)
    write_preview_png(
        preview_png,
        classes,
        cells,
        trajectory_rows,
        args.grid_resolution,
        max(1, args.preview_cell_size),
    )

    manifest = {
        "source_bag": args.source_bag,
        "map_pcd": map_pcd,
        "trajectory_csv": trajectory_csv,
        "output_dir": args.output_dir,
        "grid_resolution_m": args.grid_resolution,
        "lidar_height_m": args.lidar_height_m,
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
        },
        "params": {
            "traj_seed_radius_m": args.traj_seed_radius_m,
            "traj_seed_height_tolerance_m": args.traj_seed_height_tol_m,
            "sidewalk_fill_max_distance_m": args.sidewalk_fill_max_distance_m,
            "sidewalk_fill_height_tolerance_m": args.sidewalk_fill_height_tol_m,
            "neighbor_height_tolerance_m": args.neighbor_height_tol_m,
            "curb_height_diff_m": args.curb_height_diff_m,
            "road_min_drop_from_sidewalk_m": args.road_min_drop_from_sidewalk_m,
            "road_max_drop_from_sidewalk_m": args.road_max_drop_from_sidewalk_m,
            "max_obstacle_height_m": args.max_obstacle_height_m,
        },
    }
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("semantic classification complete")
    print("  cells_observed :", len(cells))
    print("  cells_sidewalk :", len(classes["sidewalk"]))
    print("  cells_road     :", len(classes["road"]))
    print("  cells_curb     :", len(classes["curb"]))
    print("  preview        :", preview_png)
    print("  manifest       :", manifest_json)


if __name__ == "__main__":
    main()
