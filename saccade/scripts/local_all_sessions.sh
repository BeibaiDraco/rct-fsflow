#!/usr/bin/env bash
set -euo pipefail

# make local package imports work
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ---- knobs you’ll tweak locally ----
ROOT=RCT_02
ORIENT=vertical
TAG=sacc_v1
T0=-0.40
T1=0.20
BIN_MS=5
TRAINC="-0.30:-0.10"
TRAINS="-0.10:-0.03"
LAGS_MS=30
PERMS=200   # smaller locally

# ---- discover all sessions ----
if [[ ! -d "${ROOT}" ]]; then
  echo "[ERROR] ${ROOT} not found"; exit 1
fi
SESSIONS=$(ls -1d "${ROOT}"/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9] 2>/dev/null | xargs -n1 basename | sort)

# ---- filter to sessions having >=2 areas ----
VALID_AREAS_REGEX='^(MFEF|MLIP|MSC|SFEF|SLIP|SSC)$'
FILTERED=()
for sid in ${SESSIONS}; do
  adir="${ROOT}/${sid}/areas"
  if [[ -d "${adir}" ]]; then
    n=$(ls -1 "${adir}" 2>/dev/null | egrep -E "${VALID_AREAS_REGEX}" | wc -l | awk '{print $1}')
  else
    n=0
  fi
  if [[ "${n}" -ge 2 ]]; then
    FILTERED+=("${sid}")
  else
    echo "[skip] ${sid}: only ${n} area(s) under ${adir}"
  fi
done

TOTAL=${#FILTERED[@]}
echo "[info] ${TOTAL} sessions meet the >=2 areas criterion"

# ---- run sequentially ----
i=0
for sid in "${FILTERED[@]}"; do
  echo "[local] running ${sid}  ($((i+1))/${TOTAL})"
  python scripts/41_run_all_sessions_sacc.py \
    --root "${ROOT}" \
    --sid "${sid}" \
    --orientation "${ORIENT}" \
    --t0 "${T0}" --t1 "${T1}" --bin_ms "${BIN_MS}" \
    --trainC="${TRAINC}" --trainS="${TRAINS}" \
    --lags_ms "${LAGS_MS}" --perms "${PERMS}" \
    --tag "${TAG}" \
    --out_root results_sacc
  i=$((i+1))
done
