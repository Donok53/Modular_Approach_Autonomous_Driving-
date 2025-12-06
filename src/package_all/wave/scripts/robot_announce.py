#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import requests
import rospy

def main():
    rospy.init_node("robot_announce")

    base_url        = rospy.get_param("~server_base_url", "http://127.0.0.1:44000")
    endpoint        = rospy.get_param("~announce_endpoint", "/app/robot/announce")
    robot_id        = str(rospy.get_param("~robot_id", "1"))
    robot_name      = rospy.get_param("~robot_name", "LIO 로봇")
    secret          = rospy.get_param("~robot_reg_secret", "")
    period_sec      = float(rospy.get_param("~period_sec", 10.0))
    timeout_sec     = float(rospy.get_param("~timeout_sec", 3.0))

    # ✅ 서버가 대시보드에 표시할 수 있도록 공개 WS/CAM URL도 함께 보냄
    public_ws_url   = rospy.get_param("~public_ws_url",  "")
    public_cam_url  = rospy.get_param("~public_cam_url", "")

    url = base_url.rstrip("/") + endpoint

    rospy.loginfo("robot_announce: url=%s, id=%s, name=%s", url, robot_id, robot_name)

    session = requests.Session()

    while not rospy.is_shutdown():
        payload = {
            "id": robot_id,
            "name": robot_name,
        }
        if secret:
            payload["secret"] = secret

        # 서버 구현에 따라 키 이름이 다를 수 있어 동의어까지 같이 전송
        if public_ws_url:
            payload["public_ws_url"] = public_ws_url
            payload["ws_url"] = public_ws_url
        if public_cam_url:
            payload["public_cam_url"] = public_cam_url
            payload["cam_url"] = public_cam_url

        try:
            resp = session.post(url, json=payload, timeout=timeout_sec)
            text = resp.text.strip()
            rospy.loginfo("Announce OK (code %s): %s", resp.status_code, text[:200])
        except requests.RequestException as e:
            rospy.logwarn("Announce failed: %s", e)

        # 주기 대기
        for _ in range(int(max(1, period_sec * 10))):
            if rospy.is_shutdown():
                break
            time.sleep(0.1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
