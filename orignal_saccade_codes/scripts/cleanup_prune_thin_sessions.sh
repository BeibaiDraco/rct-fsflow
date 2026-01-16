#!/usr/bin/env bash
set -euo pipefail

# config
ROOT="${1:-RCT_02}"          # path to RCT_02
OUT="${2:-results_sacc}"     # path to results_sacc
MIN="${3:-2}"                # minimum areas to keep
MODE="${4:-dry}"             # 'dry' or 'delete'

VALID_AREAS_REGEX='^(MFEF|MLIP|MSC|SFEF|SLIP|SSC)$'

if [[ ! -d "${ROOT}" ]]; then
  echo "[ERROR] root not found: ${ROOT}" >&2; exit 1
fi
if [[ ! -d "${OUT}" ]]; then
  echo "[ERROR] out_root not found: ${OUT}" >&2; exit 1
fi

echo "[info] scanning ${OUT} against ${ROOT}, min areas = ${MIN}, mode = ${MODE}"

COUNT=0
REM=0

# iterate sessions that have results_sacc/<sid>
for p in "${OUT}"/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]; do
  [[ -e "$p" ]] || continue
  sid="$(basename "$p")"
  COUNT=$((COUNT+1))

  adir="${ROOT}/${sid}/areas"
  if [[ -d "${adir}" ]]; then
    n=$(ls -1 "${adir}" 2>/dev/null | egrep -E "${VALID_AREAS_REGEX}" | wc -l | awk '{print $1}')
  else
    n=0
  fi

  if [[ "${n}" -lt "${MIN}" ]]; then
    if [[ "${MODE}" == "delete" ]]; then
      echo "[rm] ${OUT}/${sid}  (areas=${n})"
      rm -rf "${OUT}/${sid}"
      REM=$((REM+1))
    else
      echo "[would remove] ${OUT}/${sid}  (areas=${n})"
    fi
  fi
done

if [[ "${MODE}" == "delete" ]]; then
  echo "[done] scanned=${COUNT}, removed=${REM}"
else
  echo "[dry-run] scanned=${COUNT}. To delete, run:"
  echo "  bash scripts/cleanup_prune_thin_sessions.sh ${ROOT} ${OUT} ${MIN} delete"
fi
