#!/usr/bin/env bash
# 서버 API 프록시(-L) + ROS Bridge/Camera 역터널(-R)을 열고,
# 공개 URL을 /tmp/robotdash_net.env 에 기록해 announce가 읽게 함.

# 사용법(launch에서 넘겨줌):
# ssh_tunnel.sh <server_user> <server_host> <ssh_port> <web_video_port> <server_api_port> <api_local_port> "<robot_name>"

set -euo pipefail

SERVER_USER="${1:?server_user}"
SERVER_HOST="${2:?server_host}"
SSH_PORT="${3:?ssh_port}"
WEB_VIDEO_LOCAL_PORT="${4:?web_video_port}"   # 보통 8080
SERVER_API_PORT="${5:?server_api_port}"       # 보통 4000
API_LOCAL_PORT="${6:?api_local_port}"         # 보통 44000
ROBOT_NAME="${7:-robot}"

# 역터널 외부 포트(단일 로봇이면 고정해도 무방)
WS_REMOTE_PORT="${WS_REMOTE_PORT:-49090}"
CAM_REMOTE_PORT="${CAM_REMOTE_PORT:-58080}"

IDENTITY_OPT=""
if [[ -n "${SSH_IDENTITY:-}" ]]; then
  IDENTITY_OPT="-i ${SSH_IDENTITY}"
fi

echo "[ssh_tunnel] opening:"
echo "  -L 127.0.0.1:${API_LOCAL_PORT} -> ${SERVER_HOST}:127.0.0.1:${SERVER_API_PORT}"
echo "  -R 0.0.0.0:${WS_REMOTE_PORT}  -> localhost:9090 (rosbridge)"
echo "  -R 0.0.0.0:${CAM_REMOTE_PORT} -> localhost:${WEB_VIDEO_LOCAL_PORT} (web_video)"

# Announce가 읽을 수 있게 공개 URL을 기록
NET_ENV_FILE="/tmp/robotdash_net.env"
{
  echo "export PUBLIC_WS_URL=ws://${SERVER_HOST}:${WS_REMOTE_PORT}"
  echo "export PUBLIC_CAM_URL=http://${SERVER_HOST}:${CAM_REMOTE_PORT}"
} > "${NET_ENV_FILE}"

# SSH 터널
exec ssh -N -T \
  -o ExitOnForwardFailure=yes \
  -L "127.0.0.1:${API_LOCAL_PORT}:127.0.0.1:${SERVER_API_PORT}" \
  -R "0.0.0.0:${WS_REMOTE_PORT}:localhost:9090" \
  -R "0.0.0.0:${CAM_REMOTE_PORT}:localhost:${WEB_VIDEO_LOCAL_PORT}" \
  ${IDENTITY_OPT} -p "${SSH_PORT}" "${SERVER_USER}@${SERVER_HOST}"
