#!/usr/bin/env bash
# LLM-as-judge server for the Company Memo and Wikipedia evaluations.
# Serves the pinned Qwen3-4B-Instruct-2507 snapshot with vLLM on one GPU, using a
# small memory slice so it can share the GPU with a trainer (the released runs
# put it on the SSD teacher GPU).
#
#   JUDGE_GPU=1 bash scripts/serve_judge.sh        # defaults: GPU 1, port 18001, 15% of GPU memory
#
# Configs point at http://127.0.0.1:18001/v1; the launcher checks it is up.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
MODEL="$(sed -n 's/^model: //p' configs/versions.yaml)"
REVISION="$(sed -n 's/^model_revision: //p' configs/versions.yaml)"
HF_HOME="${HF_HOME:-$REPO/scratch/huggingface}"
SNAPSHOT="$HF_HOME/hub/models--${MODEL//\//--}/snapshots/$REVISION"
[[ -f "$SNAPSHOT/config.json" ]] || { echo "model snapshot missing at $SNAPSHOT; run bash setup.sh" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${JUDGE_GPU:-1}" HF_HOME HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT="$REPO/scratch/vllm-cache"
export PATH="$REPO/.venv/bin:$PATH"   # vLLM's compile step shells out to the venv's ninja
exec .venv/bin/vllm serve "$SNAPSHOT" --served-model-name "$MODEL" \
    --port "${JUDGE_PORT:-18001}" --gpu-memory-utilization "${JUDGE_MEMORY_FRACTION:-0.15}" \
    --max-model-len 4096 --max-num-seqs 64 --enforce-eager
