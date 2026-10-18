#!/usr/bin/env bash
# Run the entire long-video chain without requiring a seeded player gallery.
set -eo pipefail

CV_ROOT="${BASKET_AUTODL_ROOT:-/root/autodl-tmp}"
CV_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CV_INPUT="${1:-/root/autodl-fs/data/long_video_test/nba-hd/source.mp4}"
CV_OUTPUT="${2:-$CV_ROOT/output/matches/long-e2e-$(date +%Y%m%d-%H%M%S)}"
CV_MATCH_ID="${3:-$(basename "$CV_OUTPUT")}"
CV_PY="$CV_ROOT/env/control/bin/python"
CV_ENV="$CV_ROOT/tool/manifests/basket-autodl.env"

if (( $# > 3 )); then
    echo 'Usage: bash tools/run_long_match_exploratory.sh [input.mp4] [output-dir] [match-id]'
    exit 2
fi
for CV_FILE in "$CV_INPUT" "$CV_ENV"; do
    if [[ ! -f "$CV_FILE" ]]; then
        printf 'Required file not found: %s\n' "$CV_FILE"
        exit 1
    fi
done
if [[ ! -x "$CV_PY" ]]; then
    printf 'Control Python not found: %s\n' "$CV_PY"
    exit 1
fi
mkdir -p "$CV_OUTPUT" "$CV_ROOT/tool/manifests"
CV_OUTPUT="$(cd "$CV_OUTPUT" && pwd)"
CV_INPUT="$(cd "$(dirname "$CV_INPUT")" && pwd)/$(basename "$CV_INPUT")"
if [[ -f "$CV_OUTPUT/run.pid" ]]; then
    CV_PREVIOUS_PID="$(cat "$CV_OUTPUT/run.pid")"
    if [[ "$CV_PREVIOUS_PID" =~ ^[1-9][0-9]*$ ]] && kill -0 "$CV_PREVIOUS_PID" 2>/dev/null; then
        printf 'This run already has a live process: %s\n' "$CV_PREVIOUS_PID"
        exit 1
    fi
fi

export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$CV_ROOT/tool/cache/torch"
cd "$CV_REPO"
nohup "$CV_PY" -u -m cli.process_match \
    --env-file "$CV_ENV" --input "$CV_INPUT" \
    --output-dir "$CV_OUTPUT" --match-id "$CV_MATCH_ID" \
    --stage full --skip-bootstrap --allow-unreviewed --allow-shared-gpu \
    --qwen-min-free-gib 20 --video-tokens 4096 \
    --coarse-method states --window-s 20 --overlap-s 6 \
    --coarse-frames 16 --review-frames 16 --min-free-gb 6 \
    >> "$CV_OUTPUT/run.log" 2>&1 < /dev/null &
CV_PID=$!
printf '%s\n' "$CV_PID" > "$CV_OUTPUT/run.pid"
printf '%s\n' "$CV_OUTPUT" > "$CV_ROOT/tool/manifests/latest-long-match-run.txt"
printf 'Started PID: %s\nInput: %s\nOutput: %s\nLog: %s/run.log\n' \
    "$CV_PID" "$CV_INPUT" "$CV_OUTPUT" "$CV_OUTPUT"
