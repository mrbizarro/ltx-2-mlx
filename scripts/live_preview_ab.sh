#!/bin/zsh
# Prove the live preview is LOSSLESS: render the same seed+prompt+args twice, once with
# --live-preview off and once with tae, and hash the elementary streams.
#
# Hashing the container is not enough on its own (mux metadata can carry a timestamp), so
# this hashes the *video elementary stream* and the *decoded audio* as well. All three must
# match. If they do not, the change is not lossless and the report must say so with both
# hashes rather than shipping it as "exact".
#
#   scripts/live_preview_ab.sh <out-dir> [extra ltx-2-mlx args...]
#
# Expects: $LTX_PY (python), $LTX_MODELS (pack dir), $LTX_TAE (taeltx2_3.safetensors),
#          $LTX_PROMPT_FILE. Run it under the GPU lock; it does not take one itself.
set -eu

OUT=${1:?usage: live_preview_ab.sh <out-dir> [args...]}
shift || true
mkdir -p "$OUT"

WT=${LTX_WT:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${LTX_PY:?set LTX_PY to the python that has mlx installed}
MODELS=${LTX_MODELS:?set LTX_MODELS to the model pack directory}
TAE=${LTX_TAE:?set LTX_TAE to taeltx2_3.safetensors}
PROMPT="$(cat ${LTX_PROMPT_FILE:?set LTX_PROMPT_FILE})"
PYTHONPATH="$WT/packages/ltx-core-mlx/src:$WT/packages/ltx-pipelines-mlx/src"
export PYTHONPATH

hash_streams() {
  local mp4=$1 tag=$2
  ffmpeg -v error -i "$mp4" -map 0:v -c copy -f rawvideo - \
    | shasum -a 256 | awk -v t="$tag" '{print t" video "$1}'
  ffmpeg -v error -i "$mp4" -map 0:a -f wav -acodec pcm_s16le - \
    | shasum -a 256 | awk -v t="$tag" '{print t" audio "$1}'
  shasum -a 256 "$mp4" | awk -v t="$tag" '{print t" file  "$1}'
}

for arm in off on; do
  extra=()
  if [[ $arm == on ]]; then
    extra=(--live-preview tae --live-preview-tae "$TAE" --live-preview-dir "$OUT/live_on")
  fi
  echo "=== arm: $arm ==="
  /usr/bin/time -l $PY -m ltx_pipelines_mlx.cli \
    generate --prompt "$PROMPT" -o "$OUT/lp_$arm.mp4" \
    --model "$MODELS/ltx-2.5-mlx-q8" --gemma "$MODELS/gemma4-12b-ltx25-q4" \
    "${extra[@]}" "$@" 2>&1 | tee "$OUT/lp_$arm.log"
done

echo "=== hashes ==="
hash_streams "$OUT/lp_off.mp4" off | tee "$OUT/hashes.txt"
hash_streams "$OUT/lp_on.mp4" on | tee -a "$OUT/hashes.txt"

if diff <(grep '^off' "$OUT/hashes.txt" | sed 's/^off //') \
        <(grep '^on'  "$OUT/hashes.txt" | sed 's/^on //') >/dev/null; then
  echo "LOSSLESS: video, audio and container hashes all match."
else
  echo "NOT LOSSLESS: hashes differ. Report both, do not ship this as exact." >&2
  exit 2
fi
