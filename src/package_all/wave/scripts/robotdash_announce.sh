#!/usr/bin/env bash
# 로봇 네트워크 정보(ROS Bridge / Camera URL)를 Announce에 포함해 서버로 보냄

# 사용법(launch에서 넘겨줌):
# robotdash_announce.sh <server_host> <server_api_port> "<robot_name>" <gps_out_topic> <api_local_port>

set -euo pipefail

SERVER_HOST="${1:?server_host}"
SERVER_API_PORT="${2:?server_api_port}"
ROBOT_NAME="${3:?robot_name}"
GPS_TOPIC="${4:-/fix_web}"           # 사용 안하더라도 인자 형식 유지
API_LOCAL_PORT="${5:?api_local_port}"

BASE_URL="http://127.0.0.1:${API_LOCAL_PORT}"
ANN_URL="${BASE_URL}/app/robot/announce"

ROBOT_ID="${ROBOT_ID:-1}"
SECRET="${ROBOT_REG_SECRET:-}"

# ssh_tunnel.sh 가 기록해 둔 포트/URL 파일 (없어도 동작)
NET_ENV_FILE="/tmp/robotdash_net.env"
PUBLIC_WS_URL=""
PUBLIC_CAM_URL=""

if [[ -f "${NET_ENV_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${NET_ENV_FILE}"
fi

# 파일이 없거나 값이 비어있으면 합리적인 기본값으로 세팅(고정 역터널 포트)
PUBLIC_WS_URL="${PUBLIC_WS_URL:-ws://${SERVER_HOST}:49090}"
PUBLIC_CAM_URL="${PUBLIC_CAM_URL:-http://${SERVER_HOST}:58080}"

echo "[robot_announce] url=${ANN_URL}, id=${ROBOT_ID}, name=${ROBOT_NAME}"
echo "[robot_announce] WS=${PUBLIC_WS_URL} / CAM=${PUBLIC_CAM_URL}"

while true; do
  PAYLOAD=$(jq -n \
    --arg id "${ROBOT_ID}" \
    --arg name "${ROBOT_NAME}" \
    --arg ws "${PUBLIC_WS_URL}" \
    --arg cam "${PUBLIC_CAM_URL}" \
    --arg secret "${SECRET}" '
    {
      id: $id,
      name: $name,
      public_ws_url: $ws,
      public_cam_url: $cam
    } + ( $secret|length>0 ? {secret:$secret} : {} )
  ')

  HTTP_CODE=$(curl -sS -m 3 -o /tmp/robot_announce.out -w "%{http_code}" \
    -H 'Content-Type: application/json' \
    -d "${PAYLOAD}" \
    "${ANN_URL}" || echo "000")

  if [[ "${HTTP_CODE}" =~ ^2..$ ]]; then
    echo "[robot_announce] Announce OK (code ${HTTP_CODE}): $(cat /tmp/robot_announce.out)"
  else
    echo "[robot_announce] Announce FAIL (code ${HTTP_CODE})"
  fi

  sleep 10
done
