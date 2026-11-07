#!/usr/bin/env python3
"""Shared launcher behind run_onpolicy.py, run_ssd.py, run_sft.py and run_fkl.py.

    .venv/bin/python scripts/run_<method>.py configs/<method>_<task>.yaml \
        --output runs/<name> [--seed 42] [--gpus 0,1] [--no-wandb] [--dry-run]

The YAML config carries every hyperparameter; the command line only chooses
the seed, the output directory and the GPUs (``--set key=value`` overrides a
single value, e.g. for a smoke test). The launcher resolves paths,
checks the pinned versions (configs/versions.yaml), writes the resolved config
and versions.json into the output directory and runs the trainer.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
import uuid
from importlib import metadata
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
TRAINERS = {
    "onpolicy": "src/train.py",
    "ssd": "src/train.py",
    "sft": "src/train_off_policy.py",
    "fkl": "src/train_off_policy.py",
}
PINNED_PACKAGES = ("torch", "vllm", "transformers", "trl", "accelerate", "peft")
JUDGE_SCORER = "evals.online_eval:make_llm_judge_scorer"


def load_yaml(path):
    with open(path) as stream:
        return yaml.safe_load(stream)


def installed_versions():
    versions = {"python": f"{sys.version_info.major}.{sys.version_info.minor}"}
    for package in PINNED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    try:
        import torch

        versions["cuda"] = torch.version.cuda
    except Exception:  # noqa: BLE001 -- torch is absent in --dry-run environments
        versions["cuda"] = None
    return versions


def check_versions(pinned, installed):
    mismatches = {
        key: (pinned[key], installed.get(key))
        for key in ("python", "cuda", *PINNED_PACKAGES)
        if str(pinned[key]) != str(installed.get(key))
    }
    for key, (want, have) in mismatches.items():
        print(f"WARNING: {key} is {have}, the released runs used {want}", file=sys.stderr)
    return mismatches


def model_snapshot(pinned, hf_home):
    name = pinned["model"].replace("/", "--")
    snapshot = hf_home / "hub" / f"models--{name}" / "snapshots" / pinned["model_revision"]
    if not (snapshot / "config.json").exists():
        raise FileNotFoundError(
            f"model snapshot missing at {snapshot}; run `bash setup.sh` to download it"
        )
    return snapshot


def judge_available(url, model):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url.rstrip("/") + "/models", timeout=10) as response:
            served = [m["id"] for m in json.load(response)["data"]]
    except Exception as error:  # noqa: BLE001 -- any failure means no judge
        raise RuntimeError(
            f"judge server not reachable at {url} ({error}); start it with scripts/serve_judge.sh"
        ) from None
    if model not in served:
        raise RuntimeError(f"judge at {url} serves {served}, expected {model}")


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001 -- exported trees have no .git
        return None


def resolve(config, method, args, pinned, hf_home):
    if config.get("method") != method:
        raise ValueError(f"{args.config} is a {config.get('method')!r} config, not {method!r}")
    task = config["task"]
    train = dict(config["train"])
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        train[key] = yaml.safe_load(value)
    train["dataset"] = str(ROOT / config["dataset"])
    train["eval_data"] = str(ROOT / config["eval_data"])
    train["seed"] = args.seed
    train["output_dir"] = str(args.output)
    train["run_name"] = f"{task}_{method}_s{args.seed}"
    train["wandb_project"] = config.get("wandb_project", "speculative-self-distillation")
    train["wandb_group"] = task
    if args.no_wandb:
        train["report_to"] = "none"
    if not args.dry_run:
        train["model_name"] = str(model_snapshot(pinned, hf_home))
    else:
        train["model_name"] = pinned["model"]
    if method == "ssd":
        train["paired_python"] = str(PYTHON)
    return train


def runtime_env(gpus, hf_home, run_group):
    env = os.environ.copy()
    (ROOT / "scratch/wandb").mkdir(parents=True, exist_ok=True)
    env.update(
        CUDA_VISIBLE_DEVICES=gpus,
        PYTHONPATH=f"{ROOT / 'src'}{os.pathsep}{ROOT}",
        PATH=f"{PYTHON.parent}{os.pathsep}{env.get('PATH', '')}",
        HF_HOME=str(hf_home),
        HF_HUB_OFFLINE="1",
        VLLM_CACHE_ROOT=str(ROOT / "scratch/vllm-cache"),
        VLLM_PLUGINS="",
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(29570 + int(gpus.split(",")[0]) if gpus.split(",")[0].isdigit() else 29570),
        WANDB_RUN_GROUP=run_group,
        WANDB_RUN_ID=uuid.uuid4().hex,
        WANDB_RESUME="never",
        WANDB_DIR=str(ROOT / "scratch/wandb"),
    )
    env.setdefault("WANDB_INIT_TIMEOUT", "180")
    env.setdefault("OMP_NUM_THREADS", "4")
    # Each trainer owns its W&B service process.
    env.pop("WANDB_SERVICE", None)
    env.pop("_WANDB_SERVICE", None)
    return env


def main(method):
    parser = argparse.ArgumentParser(description=f"Train {method} from a YAML config")
    parser.add_argument("config", type=Path, help="configs/<method>_<task>.yaml")
    parser.add_argument("--output", type=Path, required=True, help="fresh run directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", default=None, help="CUDA_VISIBLE_DEVICES (default from the config)")
    parser.add_argument("--no-wandb", action="store_true", help="report_to=none")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override one value of the config's train block (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="print the trainer command and exit")
    args = parser.parse_args()

    config = load_yaml(args.config)
    pinned = load_yaml(ROOT / "configs/versions.yaml")
    hf_home = Path(os.environ.get("HF_HOME", ROOT / "scratch/huggingface"))
    args.output = args.output.resolve()
    train = resolve(config, method, args, pinned, hf_home)
    gpus = args.gpus or str(config["gpus"])
    expected = 2 if method == "ssd" else 1
    if len(gpus.split(",")) != expected:
        parser.error(f"{method} uses {expected} GPU(s); got --gpus {gpus}")

    command = [str(PYTHON), str(ROOT / TRAINERS[method]), *[f"{k}={v}" for k, v in train.items()]]
    if args.dry_run:
        print(json.dumps(train, indent=2))
        print(" ".join(command), file=sys.stderr)
        return

    if not PYTHON.exists():
        raise FileNotFoundError(f"{PYTHON} missing; run `bash setup.sh`")
    if not Path(train["dataset"]).exists() or not Path(train["eval_data"]).exists():
        raise FileNotFoundError(
            f"data for {config['task']} missing; run `scripts/prepare_data.py {config['task']}`"
        )
    if train.get("eval_scoring") == JUDGE_SCORER:
        kwargs = train["eval_scoring_kwargs"]
        judge_available(kwargs["judge_url"], kwargs["judge_model"])
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"{args.output} is not empty; results are never overwritten")
    args.output.mkdir(parents=True, exist_ok=True)

    installed = installed_versions()
    mismatches = check_versions(pinned, installed)
    config_text = Path(args.config).read_text()
    (args.output / "config.yaml").write_text(config_text)
    (args.output / "resolved-config.json").write_text(json.dumps(train, indent=2) + "\n")
    (args.output / "versions.json").write_text(json.dumps({
        "pinned": pinned,
        "installed": installed,
        "matches_pinned": not mismatches,
        "git_commit": git_commit(),
        "config": str(args.config),
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "gpus": gpus,
        "command": command,
        "started_at_unix": time.time(),
    }, indent=2) + "\n")

    env = runtime_env(gpus, hf_home, train["wandb_group"])
    print(f"{method} {config['task']} seed {args.seed} -> {args.output}", flush=True)
    with (args.output / "train.log").open("w") as log:
        code = subprocess.call(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f"finished with exit code {code}; log: {args.output / 'train.log'}", flush=True)
    raise SystemExit(code)
