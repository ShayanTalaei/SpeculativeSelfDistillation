#!/usr/bin/env bash
# Installation check: run every config for a few optimizer steps with two
# evaluations. The single-GPU configs (on-policy, SFT, FKL) run two at a time,
# one per GPU; the SSD configs run afterwards on both GPUs. Configs whose output
# already holds completion-state.json are skipped, so the script can be rerun.
#   bash scripts/smoke.sh [runs/smoke] [extra run-script args, e.g. --no-wandb]
# Needs: bash setup.sh, scripts/prepare_data.py all, and the judge on GPU 1
# (JUDGE_GPU=1 bash scripts/serve_judge.sh) for the memo tasks.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
OUT="${1:-runs/smoke}"; shift || true
EXTRA=("$@")
STEPS="${SMOKE_STEPS:-20}" EVAL_EVERY="${SMOKE_EVAL_EVERY:-10}"
GPUS=(${SMOKE_GPUS:-0 1})
mkdir -p "$OUT"
rm -rf "$OUT/.claimed"; mkdir -p "$OUT/.claimed"

run_one() {  # <config> <gpus>
    local config="$1" gpus="$2" name method
    name="$(basename "$config" .yaml)"; method="${name%%_*}"
    if [[ -f "$OUT/$name/completion-state.json" ]]; then echo "    $name already done"; return 0; fi
    # Both lanes walk the same list; mkdir is atomic, so each config runs once,
    # on whichever GPU is free first.
    mkdir "$OUT/.claimed/$name" 2>/dev/null || return 0
    rm -rf "$OUT/$name"
    echo ">>> $name on GPU $gpus  $(date -u +%H:%M:%S)"
    if .venv/bin/python "scripts/run_$method.py" "$config" --output "$OUT/$name" --gpus "$gpus" \
        --set "stop_after_steps=$STEPS" --set "eval_every_n_steps=$EVAL_EVERY" "${EXTRA[@]}" \
        > "$OUT/$name.launcher.log" 2>&1; then
        echo "    ok: $name, $(grep -ac 'eval\] mean=' "$OUT/$name/train.log") evaluations  $(date -u +%H:%M:%S)"
    else
        echo "    FAILED: $name (exit $?); see $OUT/$name/train.log"; return 1
    fi
}
lane() { local gpu="$1" status=0; shift; for c in "$@"; do run_one "$c" "$gpu" || status=1; done; return $status; }

single=(configs/onpolicy_*.yaml configs/sft_*.yaml configs/fkl_*.yaml)
lane "${GPUS[0]}" "${single[@]}" & pid_a=$!
sleep 5; lane "${GPUS[1]}" "${single[@]}" & pid_b=$!
status=0
wait "$pid_a" || status=1
wait "$pid_b" || status=1
for c in configs/ssd_*.yaml; do run_one "$c" "${GPUS[0]},${GPUS[1]}" || status=1; done
echo "smoke finished with status $status  $(date -u +%H:%M:%S)"
exit $status
