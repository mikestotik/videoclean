#!/usr/bin/env bash
# Deploy videoclean to a RunPod GPU pod: create if absent, update (restart) if present.
# Idempotent by pod name. Uses the v1 REST API (rest.runpod.io).
set -euo pipefail

API_URL="${RUNPOD_API_URL:-https://rest.runpod.io}"
POD_NAME="${RUNPOD_POD_NAME:-videoclean-test}"
IMAGE="${RUNPOD_IMAGE:-ghcr.io/mikestotik/videoclean:runpod}"
GPU="${RUNPOD_GPU:-NVIDIA RTX 2000 Ada Generation}"
CLOUD="${RUNPOD_CLOUD:-COMMUNITY}"
CONTAINER_DISK_GB="${RUNPOD_CONTAINER_DISK_GB:-40}"
VOLUME_GB="${RUNPOD_VOLUME_GB:-50}"
NETWORK_VOLUME_ID="${RUNPOD_NETWORK_VOLUME_ID:-}"
PORTS="${RUNPOD_PORTS:-7860/http}"
DATA_CENTERS="${RUNPOD_DATA_CENTERS:-}"
UI_USER="${VIDEOCLEAN_UI_USER:-admin}"
UI_PASSWORD="${VIDEOCLEAN_UI_PASSWORD:-}"

DRY_RUN=0
UPDATE_ONLY=0
if [[ "${RUNPOD_UPDATE_ONLY:-0}" == "1" ]]; then
  UPDATE_ONLY=1
fi

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --dry-run            print the plan and exit (no API calls, no key needed)
  --update-only        never create a pod; fail if the pod name is not found
  --pod NAME           pod name (default: videoclean-test)
  --image IMAGE        image tag (default: ghcr.io/mikestotik/videoclean:runpod)
  --gpu "TYPE"         GPU id (default: NVIDIA RTX 2000 Ada Generation)
  --cloud SECURE|COMMUNITY
  --container-disk GB  container disk in GB (default: 40)
  --volume GB          pod volume in GB (default: 50); create-only, ignored with --volume-id
  --volume-id ID       existing network volume id (survives pod terminate);
                       its datacenter is used for the pod; create-only
  --dc "DC1,DC2"       data center ids for creation (default: RunPod picks;
                       ignored with --volume-id)
  --user USER          UI login (default: admin)
  --ports LIST         comma-separated ports (default: 7860/http)

Password and key are taken from env (preferred over CLI, which is visible
in the process list): VIDEOCLEAN_UI_PASSWORD, RUNPOD_API_KEY.
EOF
}

need_val() {
  if [[ $# -lt 2 ]]; then
    echo "missing value for $1" >&2
    usage >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --update-only) UPDATE_ONLY=1; shift ;;
    --pod) need_val "$@"; POD_NAME="$2"; shift 2 ;;
    --image) need_val "$@"; IMAGE="$2"; shift 2 ;;
    --gpu) need_val "$@"; GPU="$2"; shift 2 ;;
    --cloud) need_val "$@"; CLOUD="$2"; shift 2 ;;
    --container-disk) need_val "$@"; CONTAINER_DISK_GB="$2"; shift 2 ;;
    --volume) need_val "$@"; VOLUME_GB="$2"; shift 2 ;;
    --volume-id) need_val "$@"; NETWORK_VOLUME_ID="$2"; shift 2 ;;
    --dc) need_val "$@"; DATA_CENTERS="$2"; shift 2 ;;
    --user) need_val "$@"; UI_USER="$2"; shift 2 ;;
    --ports) need_val "$@"; PORTS="$2"; shift 2 ;;
    --password) echo "--password is removed: pass VIDEOCLEAN_UI_PASSWORD via env (CLI args are visible in ps)" >&2; exit 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

command -v jq >/dev/null 2>&1 || { echo "jq is required (brew install jq)" >&2; exit 1; }

if [[ "$CLOUD" != "SECURE" && "$CLOUD" != "COMMUNITY" ]]; then
  echo "cloud must be SECURE or COMMUNITY" >&2
  exit 1
fi

for n in "$CONTAINER_DISK_GB" "$VOLUME_GB"; do
  if ! [[ "$n" =~ ^[1-9][0-9]*$ ]]; then
    echo "disk sizes must be positive integers (got: $n)" >&2
    exit 1
  fi
done

port_list="$(printf '%s' "$PORTS" | tr ',' '\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | grep -v '^$' || true)"
if ! grep -qx '7860/http' <<<"$port_list"; then
  echo "ports must include the UI port 7860/http (got: $PORTS)" >&2
  exit 1
fi

if [[ $DRY_RUN -eq 0 && -z "$UI_PASSWORD" ]]; then
  echo "VIDEOCLEAN_UI_PASSWORD is required (serve refuses to start without it)" >&2
  exit 1
fi

# Hide secrets that may appear inside echoed API response bodies.
redact() {
  printf '%s' "${1:-}" | sed -E 's/("(VIDEOCLEAN_UI_PASSWORD|VIDEOCLEAN_API_TOKEN)"[[:space:]]*:[[:space:]]*")[^"]*"/\1<redacted>"/g'
}

# REST v1 expects env as an object {"KEY":"value"} (openapi: type object),
# not the GraphQL-style [{key,value}].
build_env_json() {
  local user="$1" password="$2"
  [[ -z "$user" ]] && user="admin"
  jq -c -n \
    --arg u "$user" \
    --arg p "$password" \
    '{VIDEOCLEAN_AUTH:"on",
      VIDEOCLEAN_UI_USER:$u,
      VIDEOCLEAN_UI_PASSWORD:$p,
      VIDEOCLEAN_PORT:"7860",
      VIDEOCLEAN_DATA_DIR:"/workspace/.videoclean",
      HF_HOME:"/workspace/.cache/huggingface",
      SAM2_BUILD_CUDA:"0"}'
}

# REST v1 expects ports as an array of "port/protocol".
build_ports_json() {
  jq -c -n --arg p "$PORTS" \
    '[$p | split(",")[] | gsub("^\\s+|\\s+$";"") | select(length > 0)]'
}

ENV_JSON="$(build_env_json "$UI_USER" "$UI_PASSWORD")"
PORTS_JSON="$(build_ports_json)"
UI_PORT="$(printf '%s' "$port_list" | head -n 1 | cut -d/ -f1)"

echo "RunPod deploy plan"
echo "  pod: $POD_NAME"
echo "  image: $IMAGE"
echo "  gpu: $GPU ($CLOUD)"
echo "  disk: ${CONTAINER_DISK_GB}GB"
if [[ -n "$NETWORK_VOLUME_ID" ]]; then
  echo "  network volume: $NETWORK_VOLUME_ID (create only; datacenter derived from volume)"
else
  echo "  pod volume: ${VOLUME_GB}GB (create only; survives restart/stop, NOT pod terminate)"
fi
echo "  ports: $PORTS_JSON"
echo "  datacenters: ${DATA_CENTERS:-auto}"
echo "  ui user: $UI_USER"

if [[ $DRY_RUN -eq 1 ]]; then
  echo "  DRY-RUN: no API calls made"
  exit 0
fi

if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
  echo "RUNPOD_API_KEY is not set" >&2
  exit 1
fi

api() {
  # api METHOD URL [DATA] — fails on HTTP >= 400 (body captured on failure)
  local method="$1" url="$2" data="${3:-}"
  local args=(-sS --fail-with-body --max-time 60 --retry 2 -X "$method" -H "Authorization: Bearer $RUNPOD_API_KEY")
  if [[ -n "$data" ]]; then
    args+=(-H "Content-Type: application/json" --data "$data")
  fi
  curl "${args[@]}" "$url"
}

fail() {
  local label="$1" body="${2:-}"
  echo "RunPod API error ($label): $(redact "$body")" >&2
  exit 1
}

require_id() {
  local label="$1" body="$2"
  local id
  id="$(printf '%s' "$body" | jq -r '.id // empty' 2>/dev/null || true)"
  if [[ -z "$id" ]]; then
    fail "$label (no id in response)" "$body"
  fi
  printf '%s' "$id"
}

# --- network volume (optional): resolve id + its datacenter ---
# REST v1 routes are case-sensitive: /networkvolumes (all lowercase).
VOLUME_DC=""
if [[ -n "$NETWORK_VOLUME_ID" ]]; then
  vol="$(api GET "$API_URL/v1/networkvolumes/$NETWORK_VOLUME_ID")" || fail "get network volume" "$vol"
  VOLUME_DC="$(printf '%s' "$vol" | jq -r '.dataCenterId // empty')"
  if [[ -z "$VOLUME_DC" ]]; then
    fail "get network volume (no dataCenterId)" "$vol"
  fi
fi

pods="$(api GET "$API_URL/v1/pods")" || fail "list pods" "$pods"
matches="$(printf '%s' "$pods" | jq -c --arg name "$POD_NAME" '[.[] | select(.name == $name)]')"
match_count="$(printf '%s' "$matches" | jq 'length')"
if [[ "$match_count" -gt 1 ]]; then
  echo "RunPod API error: $match_count pods named '$POD_NAME' — delete or rename duplicates, then retry" >&2
  exit 1
fi
pod="$(printf '%s' "$matches" | jq -c 'first // empty')"
pod_id="$(printf '%s' "$pod" | jq -r '.id // empty')"

if [[ -n "$pod_id" ]]; then
  echo "Pod '$POD_NAME' found ($pod_id) — updating image/env/ports (reset: wipes container disk and in-flight jobs)"
  if [[ -n "$NETWORK_VOLUME_ID" ]]; then
    current_vol="$(printf '%s' "$pod" | jq -r '.networkVolumeId // empty')"
    if [[ "$current_vol" != "$NETWORK_VOLUME_ID" ]]; then
      echo "pod '$POD_NAME' has network volume '${current_vol:-none}' but --volume-id is '$NETWORK_VOLUME_ID'" >&2
      echo "RunPod cannot attach or replace a network volume after creation — terminate the pod and redeploy (or drop --volume-id)" >&2
      exit 1
    fi
  fi
  # PATCH replaces env wholesale: merge current pod env so manual vars survive.
  merged_env="$(printf '%s' "$pod" | jq -c --argjson ours "$ENV_JSON" '((.env // {}) * $ours)')"
  update_body="$(jq -c -n \
    --arg img "$IMAGE" \
    --argjson ports "$PORTS_JSON" \
    --argjson env "$merged_env" \
    '{imageName:$img, ports:$ports, env:$env}')"
  resp="$(api PATCH "$API_URL/v1/pods/$pod_id" "$update_body")" || fail "update pod" "$resp"
  require_id "update pod" "$resp" >/dev/null
  echo "$resp" | jq -r '"updated: id=" + .id + " desiredStatus=" + (.desiredStatus // "n/a")'
else
  if [[ $UPDATE_ONLY -eq 1 ]]; then
    echo "pod '$POD_NAME' not found and --update-only is set — refusing to create a new pod" >&2
    exit 1
  fi
  echo "Pod '$POD_NAME' not found — creating"
  create_json="$(jq -c -n \
    --arg name "$POD_NAME" \
    --arg img "$IMAGE" \
    --arg gpu "$GPU" \
    --arg cloud "$CLOUD" \
    --arg dc "$DATA_CENTERS" \
    --arg vol_id "$NETWORK_VOLUME_ID" \
    --arg vol_dc "$VOLUME_DC" \
    --argjson env "$ENV_JSON" \
    --argjson ports "$PORTS_JSON" \
    --argjson cd "$CONTAINER_DISK_GB" \
    --argjson vol "$VOLUME_GB" \
    '{name:$name,
      imageName:$img,
      cloudType:$cloud,
      gpuTypeIds:[$gpu],
      gpuCount:1,
      containerDiskInGb:$cd,
      volumeMountPath:"/workspace",
      env:$env,
      ports:$ports}
     + (if ($vol_id | length) > 0
        then {networkVolumeId:$vol_id, dataCenterIds:[$vol_dc]}
        else {volumeInGb:$vol} end)
     + (if (($vol_id | length) == 0) and (($dc | length) > 0)
        then {dataCenterIds: ($dc | split(",") | map(gsub("^\\s+|\\s+$"; "")))}
        else {} end)')"
  resp="$(api POST "$API_URL/v1/pods" "$create_json")" || fail "create pod" "$resp"
  pod_id="$(require_id "create pod" "$resp")"
  echo "$resp" | jq -r '"created: id=" + .id + " desiredStatus=" + (.desiredStatus // "n/a")'
fi

echo
echo "Proxy URL: https://${pod_id}-${UI_PORT}.proxy.runpod.net"
