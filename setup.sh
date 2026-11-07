#!/usr/bin/env bash
# Create the pinned Python environment and download the pinned student/teacher model.
#
#   bash setup.sh                                # venv, caches and scratch under /workspace or ./.venv_home
#   VENV_HOME=/big/disk/ssd_env bash setup.sh    # put them somewhere else
#
# Afterwards: .venv/bin/python scripts/prepare_data.py all
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [[ -z "${VENV_HOME:-}" ]]; then
    if [[ -d /workspace ]]; then VENV_HOME=/workspace/ssd_env; else VENV_HOME="$REPO_ROOT/.venv_home"; fi
fi
VENV_DIR="$VENV_HOME/venv"
SCRATCH_ROOT="${SCRATCH_ROOT:-$VENV_HOME/scratch}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$VENV_HOME/uv-cache}"
mkdir -p "$VENV_HOME" "$SCRATCH_ROOT" "$UV_CACHE_DIR"
echo ">>> venv:    $VENV_DIR"
echo ">>> scratch: $SCRATCH_ROOT  (datasets, model, vLLM cache, W&B files)"

# ── Python environment (uv) ───────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
    echo ">>> installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
[[ -x "$VENV_DIR/bin/python" ]] || uv venv --python 3.12 "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
uv pip sync requirements.txt
uv pip check

# Repo-local symlinks that every script expects.
[[ -e .venv && ! -L .venv ]] || ln -sfn "$VENV_DIR" .venv
[[ -e scratch && ! -L scratch ]] || ln -sfn "$SCRATCH_ROOT" scratch

# ── Pinned versions ──────────────────────────────────────────────────
python - <<'PY'
import sys
from importlib import metadata
import torch, yaml
pinned = yaml.safe_load(open("configs/versions.yaml"))
rows = {"python": f"{sys.version_info.major}.{sys.version_info.minor}", "cuda": torch.version.cuda}
rows.update({p: metadata.version(p) for p in ("torch", "vllm", "transformers", "trl", "accelerate", "peft")})
bad = [k for k, v in rows.items() if str(v) != str(pinned[k])]
for k, v in rows.items():
    print(f"  {k:13s} {v}" + ("" if k not in bad else f"   (released runs: {pinned[k]})"))
print("  GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none visible")
PY

# ── Model snapshot ───────────────────────────────────────────────────
HF_HOME="${HF_HOME:-$SCRATCH_ROOT/huggingface}" HF_XET_HIGH_PERFORMANCE=1 python - <<'PY'
import yaml
from huggingface_hub import snapshot_download
pinned = yaml.safe_load(open("configs/versions.yaml"))
print("  model:", snapshot_download(pinned["model"], revision=pinned["model_revision"]))
PY

# ── Imports the trainers need ────────────────────────────────────────
PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT" python -c "
from paired_vllm.lockstep_engine import LockstepConfig
from ssd.distil_trainer import SSDDistilTrainer
from distil_trainer import DistilTrainer
print('  trainer imports OK')"

cat <<EOT

>>> Setup complete.
    Data:   .venv/bin/python scripts/prepare_data.py all
    W&B:    export WANDB_API_KEY=...   (or pass --no-wandb to the run scripts)
    Train:  .venv/bin/python scripts/run_ssd.py configs/ssd_scienceqa.yaml --output runs/ssd-scienceqa-s42
EOT
