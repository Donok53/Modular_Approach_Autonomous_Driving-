#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import shutil
import struct
import time

import rospy


class LioSamMapSync:
    def __init__(self):
        self.enabled = bool(rospy.get_param("~enable", True))
        self.source_dir = os.path.expanduser(rospy.get_param("~source_dir", "~/Downloads/test"))
        self.destination_dir = os.path.expanduser(
            rospy.get_param("~destination_dir", "~/code/Modular_Approach_Autonomous_Driving-/src/package_all/aimlab/lio-localizer/map/test")
        )
        self.sync_astar_ref = bool(rospy.get_param("~sync_astar_ref", True))
        self.source_ref_file = rospy.get_param("~source_ref_file", "map_reference_coordinate.csv")
        self.astar_ref_destination_file = os.path.expanduser(
            rospy.get_param(
                "~astar_ref_destination_file",
                "~/code/Modular_Approach_Autonomous_Driving-/src/package_all/aimlab/astar_map/map/map_reference_coordinate.csv",
            )
        )
        self.wait_timeout_s = max(0.0, float(rospy.get_param("~wait_timeout_s", 120.0)))
        self.poll_period_s = max(0.05, float(rospy.get_param("~poll_period_s", 0.5)))
        self.stable_checks = max(1, int(rospy.get_param("~stable_checks", 3)))
        self.delete_extra = bool(rospy.get_param("~delete_extra", True))
        self.copy_on_start = bool(rospy.get_param("~copy_on_start", False))
        self.required_files = rospy.get_param(
            "~required_files",
            ["GlobalMap.pcd", "CornerMap.pcd", "SurfMap.pcd", "map_reference_coordinate.csv"],
        )
        self.extra_files = rospy.get_param("~extra_files", [])
        if isinstance(self.extra_files, str):
            self.extra_files = [self.extra_files]
        self.write_manifest = bool(rospy.get_param("~write_manifest", False))
        self.manifest_file = rospy.get_param("~manifest_file", "asset_manifest.json")
        self.export_drivable_area_pcd = bool(rospy.get_param("~export_drivable_area_pcd", False))
        self.drivable_area_state_file = os.path.expanduser(
            rospy.get_param("~drivable_area_state_file", "~/.ros/lio_sam_drivable_area_state.json")
        )
        self.drivable_area_pcd_output = rospy.get_param("~drivable_area_pcd_output", "DrivableAreaMap.pcd")
        self.drivable_area_risk_pcd_output = rospy.get_param(
            "~drivable_area_risk_pcd_output", "DrivableAreaRiskMap.pcd"
        )
        self.export_global_map_2d_pcd = bool(rospy.get_param("~export_global_map_2d_pcd", False))
        self.global_map_input_file = rospy.get_param("~global_map_input_file", "GlobalMap.pcd")
        self.global_map_2d_output = rospy.get_param("~global_map_2d_output", "GlobalMap2D.pcd")
        self.global_map_2d_z_value = float(rospy.get_param("~global_map_2d_z_value", 0.0))
        self.write_asset_descriptions = bool(rospy.get_param("~write_asset_descriptions", False))
        self.asset_readme_file = rospy.get_param("~asset_readme_file", "README_static_assets.txt")

        rospy.loginfo(
            "lio_sam_map_sync started | enable=%s src=%s dst=%s extra=%d export_drivable_pcd=%s",
            str(self.enabled),
            self.source_dir,
            self.destination_dir,
            len(self.extra_files),
            str(self.export_drivable_area_pcd),
        )

        if self.enabled and self.copy_on_start:
            self.sync_once("startup")
        rospy.on_shutdown(self.on_shutdown)

    def _file_signature(self):
        # Signature over required files to detect write completion (stable mtime+size).
        sig = []
        for rel in self.required_files:
            path = os.path.join(self.source_dir, rel)
            if not os.path.isfile(path):
                return None
            st = os.stat(path)
            if st.st_size <= 0:
                return None
            sig.append((rel, st.st_mtime_ns, st.st_size))
        return tuple(sig)

    def _wait_until_stable(self):
        if self.wait_timeout_s <= 0.0:
            return True
        deadline = time.time() + self.wait_timeout_s
        last_sig = None
        stable = 0
        saw_source = False
        while time.time() < deadline:
            if os.path.isdir(self.source_dir):
                saw_source = True
                sig = self._file_signature()
                if sig is not None:
                    if sig == last_sig:
                        stable += 1
                    else:
                        last_sig = sig
                        stable = 1
                    if stable >= self.stable_checks:
                        return True
            time.sleep(self.poll_period_s)
        if not saw_source:
            rospy.logwarn("lio_sam_map_sync: source directory not found: %s", self.source_dir)
            return False
        rospy.logwarn("lio_sam_map_sync: timeout waiting map files to stabilize, trying best-effort copy")
        return True

    @staticmethod
    def _copy_tree(src, dst):
        copied = 0
        for root, dirs, files in os.walk(src):
            rel = os.path.relpath(root, src)
            dst_root = dst if rel == "." else os.path.join(dst, rel)
            os.makedirs(dst_root, exist_ok=True)
            for d in dirs:
                os.makedirs(os.path.join(dst_root, d), exist_ok=True)
            for f in files:
                s = os.path.join(root, f)
                t = os.path.join(dst_root, f)
                shutil.copy2(s, t)
                copied += 1
        return copied

    @staticmethod
    def _delete_extras(src, dst):
        removed = 0
        for root, dirs, files in os.walk(dst, topdown=False):
            rel = os.path.relpath(root, dst)
            src_root = src if rel == "." else os.path.join(src, rel)
            for f in files:
                dst_f = os.path.join(root, f)
                src_f = os.path.join(src_root, f)
                if not os.path.exists(src_f):
                    os.remove(dst_f)
                    removed += 1
            for d in dirs:
                dst_d = os.path.join(root, d)
                src_d = os.path.join(src_root, d)
                if not os.path.exists(src_d):
                    try:
                        os.rmdir(dst_d)
                        removed += 1
                    except OSError:
                        pass
        return removed

    @staticmethod
    def _parse_extra_mapping(entry):
        value = str(entry).strip()
        if not value:
            return None, None
        if "::" in value:
            src, rel = value.split("::", 1)
        else:
            src = value
            rel = os.path.basename(value)
        src = os.path.expanduser(src.strip())
        rel = rel.strip().lstrip("/\\")
        if not rel:
            rel = os.path.basename(src)
        return src, rel

    def _copy_extra_files(self):
        copied = 0
        for entry in self.extra_files:
            src, rel = self._parse_extra_mapping(entry)
            if not src:
                continue
            if not os.path.isfile(src):
                rospy.logwarn("lio_sam_map_sync: extra file missing, skipped: %s", src)
                continue
            dst = os.path.join(self.destination_dir, rel)
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
        return copied

    @staticmethod
    def _write_ascii_pcd(path, points):
        with open(path, "w", encoding="utf-8") as f:
            f.write("# .PCD v0.7 - Point Cloud Data file format\n")
            f.write("VERSION 0.7\n")
            f.write("FIELDS x y z intensity\n")
            f.write("SIZE 4 4 4 4\n")
            f.write("TYPE F F F F\n")
            f.write("COUNT 1 1 1 1\n")
            f.write("WIDTH %d\n" % len(points))
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write("POINTS %d\n" % len(points))
            f.write("DATA ascii\n")
            for x, y, z, intensity in points:
                f.write("%.6f %.6f %.6f %.6f\n" % (x, y, z, intensity))

    @staticmethod
    def _cell_center(ix, iy, resolution):
        return (float(ix) + 0.5) * resolution, (float(iy) + 0.5) * resolution

    @staticmethod
    def _pcd_type_to_struct(type_code, size):
        table = {
            ("F", 4): "f",
            ("F", 8): "d",
            ("I", 1): "b",
            ("I", 2): "h",
            ("I", 4): "i",
            ("I", 8): "q",
            ("U", 1): "B",
            ("U", 2): "H",
            ("U", 4): "I",
            ("U", 8): "Q",
        }
        return table.get((type_code, size))

    @classmethod
    def _read_pcd_header(cls, path):
        header = {}
        with open(path, "rb") as f:
            while True:
                line = f.readline()
                if not line:
                    raise ValueError("pcd header ended unexpectedly: %s" % path)
                text = line.decode("ascii", errors="ignore").strip()
                if not text or text.startswith("#"):
                    continue
                parts = text.split()
                key = parts[0].upper()
                values = parts[1:]
                header[key] = values
                if key == "DATA":
                    header["DATA_START"] = f.tell()
                    break

        fields = header.get("FIELDS", [])
        size = [int(v) for v in header.get("SIZE", [])]
        types = header.get("TYPE", [])
        counts = [int(v) for v in header.get("COUNT", ["1"] * len(fields))]
        points = int(header.get("POINTS", [header.get("WIDTH", ["0"])[0]])[0])
        data_mode = header.get("DATA", [""])[0].lower()

        if not (len(fields) == len(size) == len(types) == len(counts)):
            raise ValueError("pcd header field metadata mismatch: %s" % path)

        offset = 0
        layout = []
        for name, item_size, item_type, item_count in zip(fields, size, types, counts):
            struct_code = cls._pcd_type_to_struct(item_type, item_size)
            if struct_code is None:
                raise ValueError("unsupported pcd field type %s/%s in %s" % (item_type, item_size, path))
            layout.append(
                {
                    "name": name,
                    "size": item_size,
                    "type": item_type,
                    "count": item_count,
                    "offset": offset,
                    "struct_code": struct_code,
                }
            )
            offset += item_size * item_count

        header["POINT_COUNT"] = points
        header["DATA_MODE"] = data_mode
        header["POINT_STEP"] = offset
        header["LAYOUT"] = layout
        return header

    @classmethod
    def _read_pcd_points(cls, path):
        header = cls._read_pcd_header(path)
        fields = {item["name"]: item for item in header["LAYOUT"]}
        if "x" not in fields or "y" not in fields or "z" not in fields:
            raise ValueError("pcd missing x/y/z fields: %s" % path)

        points = []
        data_mode = header["DATA_MODE"]
        point_count = header["POINT_COUNT"]
        point_step = header["POINT_STEP"]
        field_names = header.get("FIELDS", [])

        if data_mode == "ascii":
            with open(path, "r", encoding="ascii", errors="ignore") as f:
                started = False
                field_index = {name: idx for idx, name in enumerate(field_names)}
                for line in f:
                    if not started:
                        if line.strip().lower().startswith("data"):
                            started = True
                        continue
                    row = line.strip().split()
                    if not row:
                        continue
                    x = float(row[field_index["x"]])
                    y = float(row[field_index["y"]])
                    z = float(row[field_index["z"]])
                    intensity = float(row[field_index["intensity"]]) if "intensity" in field_index else 0.0
                    points.append((x, y, z, intensity))
            return points

        if data_mode != "binary":
            raise ValueError("unsupported pcd data mode '%s' in %s" % (data_mode, path))

        def unpack_scalar(blob, spec):
            fmt = "<" + spec["struct_code"]
            return struct.unpack_from(fmt, blob, spec["offset"])[0]

        with open(path, "rb") as f:
            f.seek(header["DATA_START"])
            for _ in range(point_count):
                blob = f.read(point_step)
                if len(blob) < point_step:
                    break
                x = float(unpack_scalar(blob, fields["x"]))
                y = float(unpack_scalar(blob, fields["y"]))
                z = float(unpack_scalar(blob, fields["z"]))
                intensity = float(unpack_scalar(blob, fields["intensity"])) if "intensity" in fields else 0.0
                points.append((x, y, z, intensity))
        return points

    def _export_global_map_2d(self):
        generated = []
        if not self.export_global_map_2d_pcd:
            return generated

        src_path = os.path.join(self.destination_dir, self.global_map_input_file)
        if not os.path.isfile(src_path):
            rospy.logwarn("lio_sam_map_sync: global map file missing, skip 2d export: %s", src_path)
            return generated

        try:
            points = self._read_pcd_points(src_path)
        except Exception as e:
            rospy.logwarn("lio_sam_map_sync: failed to build 2d global map pcd: %s", str(e))
            return generated

        flattened = [(x, y, self.global_map_2d_z_value, intensity) for x, y, _z, intensity in points]
        if not flattened:
            rospy.logwarn("lio_sam_map_sync: global map had no points for 2d export")
            return generated

        out_path = os.path.join(self.destination_dir, self.global_map_2d_output)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        self._write_ascii_pcd(out_path, flattened)
        generated.append(os.path.relpath(out_path, self.destination_dir))
        return generated

    def _export_drivable_area_pcds(self):
        generated = []
        if not self.export_drivable_area_pcd:
            return generated
        if not os.path.isfile(self.drivable_area_state_file):
            rospy.logwarn(
                "lio_sam_map_sync: drivable-area state json missing, skip pcd export: %s",
                self.drivable_area_state_file,
            )
            return generated

        try:
            with open(self.drivable_area_state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            rospy.logwarn("lio_sam_map_sync: failed to read drivable-area state json: %s", str(e))
            return generated

        resolution = float(payload.get("grid_resolution_m", 0.20))
        default_z = float(payload.get("last_odom_z", 0.0))
        cells = payload.get("cells", [])
        risk_cells = payload.get("risk_cells", [])

        drivable_points = []
        for row in cells:
            if len(row) < 2:
                continue
            ix = int(row[0])
            iy = int(row[1])
            z = float(row[2]) if len(row) >= 3 else default_z
            x, y = self._cell_center(ix, iy, resolution)
            drivable_points.append((x, y, z, 100.0))

        if drivable_points:
            drivable_path = os.path.join(self.destination_dir, self.drivable_area_pcd_output)
            os.makedirs(os.path.dirname(drivable_path) or ".", exist_ok=True)
            self._write_ascii_pcd(drivable_path, drivable_points)
            generated.append(os.path.relpath(drivable_path, self.destination_dir))
        else:
            rospy.logwarn("lio_sam_map_sync: drivable-area state json has no cells to export")

        risk_points = []
        for row in risk_cells:
            if len(row) < 2:
                continue
            ix = int(row[0])
            iy = int(row[1])
            x, y = self._cell_center(ix, iy, resolution)
            risk_points.append((x, y, default_z, 50.0))

        risk_path = os.path.join(self.destination_dir, self.drivable_area_risk_pcd_output)
        if risk_points:
            os.makedirs(os.path.dirname(risk_path) or ".", exist_ok=True)
            self._write_ascii_pcd(risk_path, risk_points)
            generated.append(os.path.relpath(risk_path, self.destination_dir))
        elif os.path.exists(risk_path):
            os.remove(risk_path)

        return generated

    def _write_manifest(self):
        if not self.write_manifest:
            return
        manifest_path = os.path.join(self.destination_dir, self.manifest_file)
        files = []
        for root, _, names in os.walk(self.destination_dir):
            for name in sorted(names):
                abs_path = os.path.join(root, name)
                rel_path = os.path.relpath(abs_path, self.destination_dir)
                if rel_path == self.manifest_file:
                    continue
                st = os.stat(abs_path)
                files.append(
                    {
                        "path": rel_path.replace("\\", "/"),
                        "size_bytes": int(st.st_size),
                        "mtime_ns": int(st.st_mtime_ns),
                    }
                )
        payload = {
            "generated_at": float(time.time()),
            "source_dir": self.source_dir,
            "destination_dir": self.destination_dir,
            "extra_files": [str(v) for v in self.extra_files],
            "drivable_area_state_file": self.drivable_area_state_file if self.export_drivable_area_pcd else "",
            "files": files,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)

    def _write_asset_descriptions(self):
        if not self.write_asset_descriptions:
            return
        readme_path = os.path.join(self.destination_dir, self.asset_readme_file)
        existing_assets = []
        for name in [
            "GlobalMap.pcd",
            "GlobalMap2D.pcd",
            "CornerMap.pcd",
            "SurfMap.pcd",
            "DrivableAreaMap.pcd",
            "DrivableAreaRiskMap.pcd",
            "auto_trajectory.osm",
            "map_reference_coordinate.csv",
            "lio_sam_drivable_area_state.json",
            "asset_manifest.json",
        ]:
            if os.path.exists(os.path.join(self.destination_dir, name)):
                existing_assets.append(name)

        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("정적 자산 설명\n")
            f.write("\n")
            f.write("이 폴더는 웹 관제 서비스로 바로 전달하기 위한 정적 자산 번들입니다.\n")
            f.write("실시간 위치 기준 토픽은 /lio_localizer/odometry/optimization 입니다.\n")
            f.write("\n")
            f.write("파일 설명\n")
            f.write("\n")
            if "GlobalMap.pcd" in existing_assets:
                f.write("- GlobalMap.pcd: 전체 환경을 나타내는 대표 3D 포인트클라우드 맵입니다. 웹에서 3D 배경 맵으로 쓰기 좋습니다.\n")
            if "GlobalMap2D.pcd" in existing_assets:
                f.write("- GlobalMap2D.pcd: GlobalMap.pcd를 z=0 평면으로 펼쳐 만든 2D 배경용 포인트클라우드입니다. 웹 관제에서 평면 배경 맵처럼 쓰기 좋습니다.\n")
            if "CornerMap.pcd" in existing_assets:
                f.write("- CornerMap.pcd: 코너/엣지 특징점만 따로 모아둔 맵입니다. 주로 LOAM 방식 정합과 디버깅에 사용합니다.\n")
            if "SurfMap.pcd" in existing_assets:
                f.write("- SurfMap.pcd: 평면/표면 특징점만 따로 모아둔 맵입니다. 정합 품질 확인과 디버깅에 사용합니다.\n")
            if "DrivableAreaMap.pcd" in existing_assets:
                f.write("- DrivableAreaMap.pcd: 주행 가능 영역을 셀 중심점 형태의 포인트클라우드로 만든 파일입니다. 웹 관제에서 2D/2.5D 배경 레이어로 쓰기 좋습니다.\n")
            if "DrivableAreaRiskMap.pcd" in existing_assets:
                f.write("- DrivableAreaRiskMap.pcd: 위험 셀 또는 제한 셀을 모아둔 포인트클라우드입니다. 위험 영역 표시용 오버레이로 사용합니다.\n")
            if "auto_trajectory.osm" in existing_assets:
                f.write("- auto_trajectory.osm: 주행 궤적으로부터 만든 경로 그래프 파일입니다. A* 경로 계획이나 경로 네트워크 시각화에 사용합니다.\n")
            if "map_reference_coordinate.csv" in existing_assets:
                f.write("- map_reference_coordinate.csv: 맵 기준 좌표 메타데이터입니다. GNSS 없이 쓰는 경우 0 값일 수 있으며, 주 위치 기준으로 직접 쓰면 안 됩니다.\n")
            if "lio_sam_drivable_area_state.json" in existing_assets:
                f.write("- lio_sam_drivable_area_state.json: 주행 가능 영역의 원본 셀 상태 파일입니다. 나중에 다른 포맷으로 다시 변환할 때 기준 데이터로 사용할 수 있습니다.\n")
            if "asset_manifest.json" in existing_assets:
                f.write("- asset_manifest.json: 번들 안에 들어 있는 파일 목록과 크기 정보를 정리한 관리용 파일입니다.\n")
            f.write("\n")
            f.write("권장 사용 방식\n")
            f.write("- 3D 관제 화면: GlobalMap.pcd + localization pose\n")
            f.write("- 2D 관제 화면: GlobalMap2D.pcd + localization pose\n")
            f.write("- 2D/2.5D 관제 화면: DrivableAreaMap.pcd + path/osm + localization pose\n")
            f.write("- 설명 가능한 주행 화면: 위 자산들에 path, tracked objects, explainability topic을 함께 겹쳐서 사용\n")

    def sync_once(self, reason):
        if not self.enabled:
            return
        if not os.path.isdir(self.source_dir):
            rospy.logwarn("lio_sam_map_sync skipped (%s): source does not exist: %s", reason, self.source_dir)
            return
        os.makedirs(self.destination_dir, exist_ok=True)

        copied = self._copy_tree(self.source_dir, self.destination_dir)
        removed = 0
        if self.delete_extra:
            removed = self._delete_extras(self.source_dir, self.destination_dir)
        copied += self._copy_extra_files()
        generated = []
        generated.extend(self._export_global_map_2d())
        generated.extend(self._export_drivable_area_pcds())
        self._write_manifest()
        self._write_asset_descriptions()

        astar_ref_synced = False
        if self.sync_astar_ref:
            src_ref_path = os.path.join(self.source_dir, self.source_ref_file)
            dst_ref_path = self.astar_ref_destination_file
            if os.path.isfile(src_ref_path):
                os.makedirs(os.path.dirname(dst_ref_path) or ".", exist_ok=True)
                shutil.copy2(src_ref_path, dst_ref_path)
                astar_ref_synced = True
            else:
                rospy.logwarn(
                    "lio_sam_map_sync (%s): source ref csv missing: %s",
                    reason,
                    src_ref_path,
                )

        rospy.loginfo(
            "lio_sam_map_sync completed (%s): copied=%d removed=%d generated=%d ref_synced=%s src=%s dst=%s",
            reason,
            copied,
            removed,
            len(generated),
            str(astar_ref_synced),
            self.source_dir,
            self.destination_dir,
        )

    def on_shutdown(self):
        if not self.enabled:
            return
        try:
            self._wait_until_stable()
            self.sync_once("shutdown")
        except Exception as e:
            rospy.logwarn("lio_sam_map_sync failed on shutdown: %s", str(e))


if __name__ == "__main__":
    rospy.init_node("lio_sam_map_sync", anonymous=False)
    node = LioSamMapSync()
    rospy.spin()
