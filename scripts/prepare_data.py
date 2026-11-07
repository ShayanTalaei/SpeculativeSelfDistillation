#!/usr/bin/env python3
"""Build the training sets and evaluation files for the three released tasks.

    .venv/bin/python scripts/prepare_data.py company_memo|wikipedia|scienceqa|all

Writes scratch/data/builds/<task> (training dataset, loaded by the trainers)
and scratch/data/<task>/eval.json (held-out questions for online evaluation).
Company Memo and Wikipedia ship in data/; ScienceQA is fetched at the pinned
revision of the SDFT authors' repository (configs/versions.yaml).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
TASKS = {
    # task: (source directory, expected train rows, expected test rows)
    "company_memo": (ROOT / "data/company_memo", 1133, 100),
    "wikipedia": (ROOT / "data/wikipedia", 1129, 100),
    "scienceqa": (None, 2674, 507),
}


def build(dataset, output, **kwargs):
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{ROOT}"}
    overrides = [f"{k}={v}" for k, v in kwargs.items()]
    subprocess.run(
        [str(PYTHON), str(ROOT / "src/data/build.py"), f"dataset={dataset}", f"output={output}", *overrides],
        cwd=ROOT, env=env, check=True,
    )


def fetch_scienceqa(pinned):
    repo = ROOT / "third_party/sdft"
    if not (repo / ".git").exists():
        repo.parent.mkdir(exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", pinned["scienceqa_repo"], str(repo)], check=True)
    revision = pinned["scienceqa_revision"]
    paths = subprocess.check_output(
        ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", revision, "data/science_data"], text=True,
    ).split()
    for name in paths:
        dest = repo / name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(subprocess.check_output(["git", "-C", str(repo), "show", f"{revision}:{name}"]))
    return repo / "data/science_data"


def prepare(task):
    source, train_rows, test_rows = TASKS[task]
    build_dir = ROOT / "scratch/data/builds" / task
    eval_dir = ROOT / "scratch/data" / task
    eval_path = eval_dir / "eval.json"
    pinned = yaml.safe_load((ROOT / "configs/versions.yaml").read_text())

    if task == "scienceqa":
        fetch_scienceqa(pinned)
        from datasets import load_from_disk  # the venv interpreter runs this script

        if not build_dir.exists():
            build("scienceqa", build_dir, split="train")
        test = load_from_disk(str(ROOT / "third_party/sdft/data/science_data/eval_data"))
        rows = [
            {"question": r["prompt"][1]["content"], "answer": r["answer"], "system_prompt": r["prompt"][0]["content"]}
            for r in test
        ]
    else:
        qa = json.loads(next(source.glob("*_questions_answers_split_fixed.json")).read_text())
        if not build_dir.exists():
            build("qa_pairs", build_dir, path=source, split="train")
        rows = [{"question": q, "answer": a} for q, a in zip(qa["test"]["question"], qa["test"]["answer"])]

    if len(rows) != test_rows:
        raise ValueError(f"{task}: expected {test_rows} evaluation rows, found {len(rows)}")
    eval_dir.mkdir(parents=True, exist_ok=True)
    if not eval_path.exists():
        eval_path.write_text(json.dumps(rows, indent=2) + "\n")
    metadata = json.loads((build_dir / "metadata.json").read_text())
    if metadata["num_rows"] != train_rows:
        raise ValueError(f"{task}: expected {train_rows} training rows, built {metadata['num_rows']}")
    print(f"{task}: {train_rows} training rows -> {build_dir}; {test_rows} evaluation rows -> {eval_path}")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in (*TASKS, "all"):
        raise SystemExit(__doc__)
    for task in TASKS if sys.argv[1] == "all" else [sys.argv[1]]:
        prepare(task)


if __name__ == "__main__":
    main()
