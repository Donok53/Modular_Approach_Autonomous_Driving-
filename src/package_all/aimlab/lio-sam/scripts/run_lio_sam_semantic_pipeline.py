#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from typing import Optional, Tuple


REPO_ROOT = "/home/byeongjae/code/Modular_Approach_Autonomous_Driving-"
WORKSPACE_SETUP = os.path.join(REPO_ROOT, "devel/setup.bash")
MAP_TEST_DIR = os.path.join(REPO_ROOT, "src/package_all/aimlab/lio-localizer/map/test")
DEFAULT_BAG = "/home/byeongjae/bagfiles/1.made_map/camera_right.bag"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run lio_sam mapping, replay a bag, stop cleanly, and wait for semantic sidewalk post-processing."
    )
    parser.add_argument("--bag", default=DEFAULT_BAG)
    parser.add_argument("--play-rate", type=float, default=1.0)
    parser.add_argument("--start-delay", type=float, default=1.0)
    parser.add_argument("--bag-duration", type=float, default=0.0)
    parser.add_argument("--roslaunch-warmup-s", type=float, default=5.0)
    parser.add_argument("--topics", nargs="+", default=["/ouster/points", "/imu/data"])
    parser.add_argument("--wait-semantic-timeout-s", type=float, default=1800.0)
    parser.add_argument("--display", default=os.environ.get("DISPLAY", ":1"))
    parser.add_argument("--semantic-log-file", default=os.path.join(MAP_TEST_DIR, "semantic_sidewalk_generation.log"))
    parser.add_argument("--semantic-lidar-frame-stride", type=int, default=1)
    parser.add_argument("--semantic-point-stride", type=int, default=8)
    parser.add_argument("--semantic-max-lidar-frames", type=int, default=100000)
    parser.add_argument("--pipeline-master-port", type=int, default=11361)
    parser.add_argument("--restart-run-launch", action="store_true")
    parser.add_argument("--run-launch-command", default="")
    return parser.parse_args()


def bash_env_prefix(display, ros_master_uri="", ros_hostname=""):
    parts = [
        "source /opt/ros/noetic/setup.bash",
        "source %s" % shlex.quote(WORKSPACE_SETUP),
    ]
    if ros_master_uri:
        parts.insert(0, "export ROS_MASTER_URI=%s" % shlex.quote(ros_master_uri))
    if ros_hostname:
        parts.insert(0, "export ROS_HOSTNAME=%s" % shlex.quote(ros_hostname))
    if display:
        parts.insert(0, "export DISPLAY=%s" % shlex.quote(display))
    return " && ".join(parts)


def launch_process(command, display="", ros_master_uri="", ros_hostname="", stdout=None, stderr=None):
    shell_cmd = "%s && %s" % (bash_env_prefix(display, ros_master_uri=ros_master_uri, ros_hostname=ros_hostname), command)
    return subprocess.Popen(
        ["/bin/bash", "-lc", shell_cmd],
        stdout=stdout,
        stderr=stderr,
        preexec_fn=os.setsid,
        text=True,
    )


def terminate_process_group(proc, label):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=20.0)
    except subprocess.TimeoutExpired:
        print("[warn] %s did not exit after SIGINT, escalating to SIGTERM" % label, flush=True)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            print("[warn] %s did not exit after SIGTERM, escalating to SIGKILL" % label, flush=True)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                return
            proc.wait(timeout=5.0)


def process_matches(patterns):
    if not patterns:
        return False
    try:
        proc = subprocess.run(["pgrep", "-af", "python3"], capture_output=True, text=True, check=False)
    except Exception:
        return False
    if proc.returncode not in (0, 1):
        return False
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    for line in lines:
        if all(pattern in line for pattern in patterns):
            return True
    return False


def wait_for_semantic_completion(args):
    bundle_dir = MAP_TEST_DIR
    bundle_pattern = "generate_semantic_sidewalk_bundle.py"
    raw_pattern = "build_raw_semantic_bev_map.py"
    bag_pattern = args.bag
    deadline = time.time() + max(0.0, args.wait_semantic_timeout_s)
    saw_generator = False
    while time.time() < deadline:
        has_bundle = process_matches([bundle_pattern, bundle_dir])
        has_raw = process_matches([raw_pattern, bag_pattern])
        if has_bundle or has_raw:
            saw_generator = True
        if saw_generator and not has_bundle and not has_raw:
            return True
        if not saw_generator:
            if os.path.isfile(args.semantic_log_file):
                try:
                    with open(args.semantic_log_file, "r", encoding="utf-8") as f:
                        text = f.read()
                    if "launch:" in text:
                        saw_generator = True
                except Exception:
                    pass
        time.sleep(2.0)
    return False


def tail_file(path, max_lines=40):
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    return "".join(lines[-max_lines:])


def detect_running_run_launch() -> Tuple[Optional[int], str]:
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid,args"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None, ""

    candidates = []
    for line in proc.stdout.splitlines():
        if "/src/package_all/run.launch" not in line:
            continue
        if "roslaunch" not in line:
            continue
        line = line.strip()
        if not line:
            continue
        try:
            pid_str, cmd = line.split(None, 1)
            pid = int(pid_str)
        except Exception:
            continue
        candidates.append((pid, cmd))

    # Prefer the actual roslaunch Python process if present.
    for pid, cmd in candidates:
        marker = "/opt/ros/noetic/bin/roslaunch "
        if marker in cmd:
            roslaunch_part = cmd.split(marker, 1)[1]
            return pid, "roslaunch %s" % roslaunch_part

    if candidates:
        pid, cmd = candidates[0]
        marker = "roslaunch "
        if marker in cmd:
            return pid, cmd.split(marker, 1)[1] and ("roslaunch " + cmd.split(marker, 1)[1])
        return pid, cmd
    return None, ""


def terminate_pid(pid: Optional[int], label: str):
    if not pid:
        return
    try:
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError:
        return
    deadline = time.time() + 20.0
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.5)
    print("[warn] %s pid=%d did not exit after SIGINT, sending SIGTERM" % (label, pid), flush=True)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def main():
    args = parse_args()
    args.bag = os.path.abspath(os.path.expanduser(args.bag))
    if not os.path.isfile(args.bag):
        print("[error] bag not found: %s" % args.bag, file=sys.stderr)
        return 2

    pipeline_master_uri = "http://127.0.0.1:%d" % int(args.pipeline_master_port)
    pipeline_hostname = "127.0.0.1"

    roslaunch_cmd = (
        "roslaunch lio_sam run-robot.launch "
        "semantic_source_bag:=%s "
        "semantic_lidar_frame_stride:=%d "
        "semantic_point_stride:=%d "
        "semantic_max_lidar_frames:=%d"
        % (
            shlex.quote(args.bag),
            max(1, int(args.semantic_lidar_frame_stride)),
            max(1, int(args.semantic_point_stride)),
            max(1, int(args.semantic_max_lidar_frames)),
        )
    )
    rosbag_cmd = (
        "rosbag play --clock -r %s -d %s %s %s --topics %s"
        % (
            args.play_rate,
            args.start_delay,
            ("-u %s" % args.bag_duration) if args.bag_duration > 0.0 else "",
            shlex.quote(args.bag),
            " ".join(shlex.quote(topic) for topic in args.topics),
        )
    )

    roslaunch_proc = None
    rosbag_proc = None
    restarted_run_launch_proc = None
    existing_run_launch_pid = None
    existing_run_launch_cmd = args.run_launch_command.strip()
    try:
        if args.restart_run_launch and not existing_run_launch_cmd:
            existing_run_launch_pid, existing_run_launch_cmd = detect_running_run_launch()
            if existing_run_launch_cmd:
                print("[info] detected running run.launch pid=%s" % str(existing_run_launch_pid), flush=True)
                print("[info] will restart with: %s" % existing_run_launch_cmd, flush=True)
            else:
                print("[warn] --restart-run-launch requested but no running run.launch was detected", flush=True)

        print("[info] starting lio_sam run-robot.launch", flush=True)
        roslaunch_proc = launch_process(
            roslaunch_cmd,
            display=args.display,
            ros_master_uri=pipeline_master_uri,
            ros_hostname=pipeline_hostname,
        )
        time.sleep(max(0.0, args.roslaunch_warmup_s))

        print("[info] replaying bag: %s" % args.bag, flush=True)
        rosbag_proc = launch_process(
            rosbag_cmd,
            display=args.display,
            ros_master_uri=pipeline_master_uri,
            ros_hostname=pipeline_hostname,
        )
        rosbag_rc = rosbag_proc.wait()
        if rosbag_rc != 0:
            print("[warn] rosbag play exited with code %d" % rosbag_rc, flush=True)

        print("[info] stopping lio_sam launch cleanly (SIGINT)", flush=True)
        terminate_process_group(roslaunch_proc, "roslaunch")
        roslaunch_proc = None

        print("[info] waiting for semantic sidewalk post-processing to finish", flush=True)
        finished = wait_for_semantic_completion(args)
        if not finished:
            print("[warn] semantic sidewalk post-processing did not finish within timeout", flush=True)

        if args.restart_run_launch and existing_run_launch_cmd:
            print("[info] restarting run.launch on the default ROS master", flush=True)
            terminate_pid(existing_run_launch_pid, "existing run.launch")
            restarted_run_launch_proc = launch_process(
                existing_run_launch_cmd,
                display=args.display,
                ros_master_uri=os.environ.get("ROS_MASTER_URI", ""),
                ros_hostname=os.environ.get("ROS_HOSTNAME", ""),
            )
            time.sleep(5.0)
            if restarted_run_launch_proc.poll() is not None:
                print("[warn] restarted run.launch exited early with code %s" % str(restarted_run_launch_proc.returncode), flush=True)

        semantic_state = os.path.join(MAP_TEST_DIR, "semantic_sidewalk_state.json")
        drivable_state = os.path.join(MAP_TEST_DIR, "lio_sam_drivable_area_state.json")
        global_map = os.path.join(MAP_TEST_DIR, "GlobalMap.pcd")
        global_map2d = os.path.join(MAP_TEST_DIR, "GlobalMap2D.pcd")
        ros_drivable = os.path.expanduser("~/.ros/lio_sam_drivable_area_state.json")

        print("[info] final outputs", flush=True)
        for path in (global_map, global_map2d, semantic_state, drivable_state, ros_drivable):
            exists = os.path.isfile(path)
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path))) if exists else "-"
            print("  %s | exists=%s | mtime=%s" % (path, str(exists), mtime), flush=True)

        log_tail = tail_file(args.semantic_log_file)
        if log_tail:
            print("[info] semantic log tail", flush=True)
            print(log_tail, flush=True)
        return 0
    except KeyboardInterrupt:
        print("[warn] interrupted by user, stopping child processes", flush=True)
        return 130
    finally:
        terminate_process_group(rosbag_proc, "rosbag")
        terminate_process_group(roslaunch_proc, "roslaunch")


if __name__ == "__main__":
    sys.exit(main())
