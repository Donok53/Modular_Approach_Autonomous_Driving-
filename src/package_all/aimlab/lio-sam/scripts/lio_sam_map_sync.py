#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
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

        rospy.loginfo(
            "lio_sam_map_sync started | enable=%s src=%s dst=%s",
            str(self.enabled),
            self.source_dir,
            self.destination_dir,
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
            "lio_sam_map_sync completed (%s): copied=%d removed=%d ref_synced=%s src=%s dst=%s",
            reason,
            copied,
            removed,
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
