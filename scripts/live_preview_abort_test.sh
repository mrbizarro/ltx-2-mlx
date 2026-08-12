#!/bin/zsh
# End-to-end early-abort test: start a REAL render, wait for the first thumbnail, drop the
# ABORT sentinel, then assert the whole contract:
#
#   * the process exits 75 (not 0, not 1)
#   * the sentinel is consumed, so it cannot kill the next render
#   * status.json says {"status": "aborted", "aborted": true} and names the stage
#   * no mp4 and no wav are left behind — the abort happens during denoise, before encode
#
#   scripts/live_preview_abort_test.sh <out-dir> [extra ltx-2-mlx args...]
#
# Same environment as live_preview_ab.sh. Run under the GPU lock.
set -u

OUT=${1:?usage: live_preview_abort_test.sh <out-dir> [args...]}
shift || true
rm -rf "$OUT"; mkdir -p "$OUT"
LIVE="$OUT/live"

WT=${LTX_WT:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${LTX_PY:?set LTX_PY}
MODELS=${LTX_MODELS:?set LTX_MODELS}
TAE=${LTX_TAE:?set LTX_TAE}
PROMPT="$(cat ${LTX_PROMPT_FILE:?set LTX_PROMPT_FILE})"
PYTHONPATH="$WT/packages/ltx-core-mlx/src:$WT/packages/ltx-pipelines-mlx/src"
export PYTHONPATH

$PY -m ltx_pipelines_mlx.cli \
  generate --prompt "$PROMPT" -o "$OUT/aborted.mp4" \
  --model "$MODELS/ltx-2.5-mlx-q8" --gemma "$MODELS/gemma4-12b-ltx25-q4" \
  --live-preview tae --live-preview-tae "$TAE" --live-preview-dir "$LIVE" \
  "$@" > "$OUT/abort.log" 2>&1 &
pid=$!

waited=0
while [[ ! -f "$LIVE/preview_latest.png" ]]; do
  if ! kill -0 $pid 2>/dev/null; then
    echo "[abort-test] render exited before publishing a preview; see $OUT/abort.log" >&2
    exit 1
  fi
  sleep 2; waited=$((waited + 2))
  if (( waited > 900 )); then echo "[abort-test] timed out waiting for a preview" >&2; kill $pid; exit 1; fi
done
echo "[abort-test] preview seen at $(date +%T) after ${waited}s; dropping ABORT"
: > "$LIVE/ABORT"

wait $pid
rc=$?
echo "[abort-test] exit code: $rc (expected 75)"

fail=0
[[ $rc == 75 ]] || { echo "[abort-test] WRONG EXIT CODE"; fail=1 }
if [[ -f "$LIVE/ABORT" ]]; then echo "[abort-test] sentinel NOT consumed"; fail=1
else echo "[abort-test] sentinel consumed: OK"; fi
if [[ -f "$OUT/aborted.mp4" ]]; then echo "[abort-test] mp4 left behind"; fail=1
else echo "[abort-test] no mp4 left behind: OK"; fi
if ls "$OUT"/*.wav >/dev/null 2>&1; then echo "[abort-test] wav left behind"; fail=1
else echo "[abort-test] no wav left behind: OK"; fi

$PY - "$LIVE/status.json" <<'EOF'
import json, sys
status = json.load(open(sys.argv[1]))
print("[abort-test] status:", json.dumps({k: status.get(k) for k in
      ("schema", "status", "aborted", "aborted_at_stage", "forward", "total_forwards", "stage")}))
assert status["status"] == "aborted" and status["aborted"] is True, "status.json does not say aborted"
EOF
[[ $? == 0 ]] || fail=1

exit $fail
