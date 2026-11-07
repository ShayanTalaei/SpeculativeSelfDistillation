"""Per-run cost/performance log that survives job chaining.

The paper compares methods on **accuracy vs supervised tokens**, not vs
optimizer steps, because on-policy rollouts are longer than off-policy ones and
so a step means different amounts of work per method. Wall clock matters too,
for the practical question of what a method costs to run.

Both are awkward to recover after the fact when a run spans several Slurm
allocations:

* ``_cum_num_tokens`` in the trainer is initialised with
  ``getattr(self, "_cum_num_tokens", 0.0)``, so it restarts at 0 every time a
  job resumes. Concatenating the logs of a chained run would show the token
  axis sawtooth back to zero at each restart.
* Wall clock is per-process and simply vanishes when the job is killed.

``RunMetricsCallback`` writes one JSON line per logging step to
``run_metrics.jsonl`` in the output dir, carrying **cumulative** figures that
are stitched across jobs by reading back what previous jobs already wrote. It
also records ``n_gpu`` so wall clock can be converted to GPU-hours, which is
the only fair way to compare runs that used different allocations.
"""
from __future__ import annotations

import json
import os
import time

from transformers import TrainerCallback


class RunMetricsCallback(TrainerCallback):
    """Append cumulative tokens / wall clock / metrics, resumable across jobs."""

    FILENAME = "run_metrics.jsonl"

    def __init__(self, output_dir: str, method: str, model: str, n_gpu: int):
        self.path = os.path.join(output_dir, self.FILENAME)
        self.method = method
        self.model = model
        self.n_gpu = n_gpu
        self.job_id = os.environ.get("SLURM_JOB_ID", "local")
        self._t0 = time.time()
        self._callback_started = self._t0
        lifecycle_path = os.path.join(output_dir, "run-lifecycle.json")
        if os.path.isfile(lifecycle_path):
            with open(lifecycle_path) as stream:
                lifecycle = json.load(stream)
            # Only an active launch owns this clock; an old completed launch
            # must not inflate a directly resumed trainer's elapsed time.
            started = lifecycle.get("launcher_started_at_unix")
            if "finished_at_unix" not in lifecycle and isinstance(started, (int, float)) and 0 < started <= self._t0:
                self._t0 = started
        # Offsets from everything previous jobs in this chain already did.
        self._tok_offset, self._wall_offset = self._read_offsets()
        if self._tok_offset or self._wall_offset:
            print(
                f"[metrics] resuming cost log: "
                f"{self._tok_offset/1e6:.2f}M supervised tokens and "
                f"{self._wall_offset/3600:.2f} h already recorded",
                flush=True,
            )

    def _read_offsets(self) -> tuple[float, float]:
        if not os.path.isfile(self.path):
            return 0.0, 0.0
        tok = wall = 0.0
        with open(self.path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a torn last line from a killed job
                tok = max(tok, r.get("cum_tokens") or 0.0)
                wall = max(wall, r.get("cum_wall_s") or 0.0)
        return tok, wall

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not state.is_world_process_zero:
            return control
        if "loss" not in logs:  # eval-only or summary logs
            return control

        # The trainer's num_tokens is cumulative WITHIN this job; add what
        # earlier jobs in the chain contributed.
        job_tokens = float(logs.get("num_tokens") or 0.0)
        elapsed = time.time() - self._t0

        rec = {
            "method": self.method,
            "model": self.model,
            "n_gpu": self.n_gpu,
            "job_id": self.job_id,
            "step": state.global_step,
            "epoch": logs.get("epoch"),
            "cum_tokens": self._tok_offset + job_tokens,
            "cum_wall_s": self._wall_offset + elapsed,
            "cum_gpu_hours": (self._wall_offset + elapsed) * self.n_gpu / 3600.0,
            "job_wall_s": elapsed,
            "job_started_at_unix": self._t0,
            "callback_started_at_unix": self._callback_started,
            "loss": logs.get("loss"),
            "grad_norm": logs.get("grad_norm"),
            "learning_rate": logs.get("learning_rate"),
            "mean_completion_len": logs.get("completions/mean_length"),
            "student_ratio": logs.get("ssd/student_ratio"),
            "kl": logs.get("kl"),
            "entropy": logs.get("entropy"),
        }
        # Line-buffered append: a job killed by the wall clock still leaves
        # every completed step on disk.
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
        return control
