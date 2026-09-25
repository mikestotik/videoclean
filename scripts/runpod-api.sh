#!/usr/bin/env bash
# RunPod REST helpers for GitHub Actions (list / create / terminate / wait SSH).
set -euo pipefail

API_BASE="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
POD_NAME="${RUNPOD_POD_NAME:-videoclean-dev}"
IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404}"
GPU_TYPE="${RUNPOD_GPU_TYPE:-NVIDIA GeForce RTX 4090}"
DATA_CENTER="${RUNPOD_DATA_CENTER:-EU-RO-1}"
CONTAINER_DISK_GB="${RUNPOD_CONTAINER_DISK_GB:-40}"

need() {
  local v="$1"
  if [[ -z "${!v:-}" ]]; then
    echo "missing env: ${v}" >&2
    exit 1
  fi
}

api() {
  need RUNPOD_API_KEY
  local method="$1"
  local path="$2"
  shift 2
  curl -fsS -X "${method}" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    "${API_BASE}${path}" \
    "$@"
}

list_pods_json() {
  api GET /pods
}

find_running_pod() {
  list_pods_json | jq -c --arg name "${POD_NAME}" '
    [.[] | select(.name == $name and .desiredStatus == "RUNNING")] | .[0] // empty
  '
}

find_any_pod() {
  list_pods_json | jq -c --arg name "${POD_NAME}" '
    [.[] | select(.name == $name and .desiredStatus != "TERMINATED")] | .[0] // empty
  '
}

ssh_target_from_pod() {
  local pod_json="$1"
  local ip port
  ip="$(jq -r '.publicIp // empty' <<<"${pod_json}")"
  port="$(jq -r '.portMappings["22"] // .portMappings."22" // empty' <<<"${pod_json}")"
  if [[ -z "${ip}" || -z "${port}" || "${ip}" == "null" || "${port}" == "null" ]]; then
    echo "pod has no publicIp/port 22 mapping yet" >&2
    return 1
  fi
  echo "${ip}:${port}"
}

wait_ssh() {
  local pod_id="$1"
  local tries="${2:-36}"
  local i pod_json target ip port
  for i in $(seq 1 "${tries}"); do
    pod_json="$(api GET "/pods/${pod_id}")"
    if target="$(ssh_target_from_pod "${pod_json}")"; then
      ip="${target%%:*}"
      port="${target##*:}"
      if ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
        -i "${RUNPOD_SSH_KEY_PATH}" -p "${port}" "root@${ip}" "true" 2>/dev/null; then
        echo "${pod_json}"
        return 0
      fi
    fi
    echo "waiting for SSH (${i}/${tries})..."
    sleep 5
  done
  echo "SSH did not become ready for pod ${pod_id}" >&2
  return 1
}

create_pod() {
  need RUNPOD_API_KEY
  need RUNPOD_NETWORK_VOLUME_ID

  # Inline bootstrap: empty volume has no scripts yet. Clone public repo, then start.
  local start_cmd='set -euo pipefail
mkdir -p /workspace
if [ ! -d /workspace/videoclean/.git ]; then
  git clone --branch "${VIDEOCLEAN_REF:-main}" \
    "${VIDEOCLEAN_REPO_URL:-https://github.com/mikestotik/videoclean.git}" \
    /workspace/videoclean
fi
exec bash /workspace/videoclean/scripts/runpod-start.sh
'

  local body
  body="$(jq -n \
    --arg name "${POD_NAME}" \
    --arg image "${IMAGE}" \
    --arg gpu "${GPU_TYPE}" \
    --arg vol "${RUNPOD_NETWORK_VOLUME_ID}" \
    --arg dc "${DATA_CENTER}" \
    --arg start "${start_cmd}" \
    --argjson disk "${CONTAINER_DISK_GB}" \
    '{
      name: $name,
      imageName: $image,
      gpuTypeIds: [$gpu],
      gpuCount: 1,
      cloudType: "SECURE",
      dataCenterIds: [$dc],
      networkVolumeId: $vol,
      volumeMountPath: "/workspace",
      containerDiskInGb: $disk,
      ports: ["7860/http", "22/tcp"],
      env: {
        VIDEOCLEAN_AUTH: "off",
        VIDEOCLEAN_PORT: "7860",
        SAM2_BUILD_CUDA: "0"
      },
      dockerStartCmd: ["bash", "-lc", $start]
    }')"

  echo "creating pod name=${POD_NAME} gpu=${GPU_TYPE} dc=${DATA_CENTER}" >&2
  api POST /pods -d "${body}"
}

terminate_pod() {
  local pod_id="$1"
  echo "terminating pod ${pod_id}" >&2
  curl -fsS -o /dev/null -X DELETE \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    "${API_BASE}/pods/${pod_id}"
}

cmd="${1:-}"
case "${cmd}" in
  find-running)
    find_running_pod
    ;;
  find-any)
    find_any_pod
    ;;
  create)
    create_pod
    ;;
  wait-ssh)
    need RUNPOD_SSH_KEY_PATH
    wait_ssh "$2"
    ;;
  terminate)
    terminate_pod "$2"
    ;;
  ssh-target)
    ssh_target_from_pod "$(cat)"
    ;;
  *)
    echo "usage: $0 {find-running|find-any|create|wait-ssh <id>|terminate <id>|ssh-target}" >&2
    exit 2
    ;;
esac
