"""Step wall clocks and asynchronous CUDA-event phase measurements.

CUDA fields measure elapsed time on the training stream, including stalls, not
kernel utilization. Nested phases must not be added twice. Step wall time
includes rollouts but excludes evaluation and checkpoint saves.
"""

import hashlib
import json
import time
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import torch
from transformers import TrainerCallback


def record_student_fingerprint(trainer):
    """Record hashes for equivalence probes without exporting model weights."""
    hashes = {}
    for name, value in trainer.model.state_dict().items():
        raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
    result = {
        "step": trainer.state.global_step,
        "tensor_sha256": hashes,
        "sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
    }
    (Path(trainer.args.output_dir) / "student-fingerprint.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


def verify_teacher_ema_updates(trainer):
    """Verify the EMA callback ran at every requested optimizer boundary."""
    callbacks = [c for c in trainer.callback_handler.callbacks if hasattr(c, "ema_update_steps")]
    if not trainer.args.sync_ref_model or len(callbacks) != 1:
        raise ValueError("Expected exactly one active teacher EMA callback")
    expected = list(range(trainer.args.ref_model_sync_steps, trainer.state.global_step + 1,
                          trainer.args.ref_model_sync_steps))
    if callbacks[0].ema_update_steps != expected:
        raise ValueError("Teacher EMA updates do not match optimizer-step boundaries")
    optimizer_parameters = {
        id(p) for group in trainer.optimizer.param_groups for p in group["params"]
    }
    if any(id(p) in optimizer_parameters for p in trainer.ref_model.parameters()):
        raise ValueError("EMA teacher must remain outside the optimizer")
    result = {
        "status": "passed", "step": trainer.state.global_step,
        "teacher_policy": "ema", "alpha": trainer.args.ref_model_mixup_alpha,
        "sync_every_n_steps": trainer.args.ref_model_sync_steps,
        "updates": len(expected), "update_steps": expected,
    }
    (Path(trainer.args.output_dir) / "teacher-ema-verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def verify_frozen_teacher(trainer, snapshot):
    """Check the live teacher against the pinned weights without saving a model."""
    from safetensors import safe_open

    from ssd.io_timing import record_io_timing

    started = time.perf_counter()
    started_unix = time.time()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(previous_threads, 4))
    snapshot = Path(snapshot)
    teacher = trainer.ref_model
    if any(p.requires_grad for p in teacher.parameters()):
        raise ValueError("Teacher has trainable parameters")
    state = teacher.state_dict()
    weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    checked = set()
    for shard in sorted(set(weight_map.values())):
        with safe_open(snapshot / shard, framework="pt", device="cpu") as reference:
            for key in reference.keys():  # noqa: SIM118 -- safe_open is not a dict
                if key not in state or not torch.equal(
                    state[key].detach().cpu(), reference.get_tensor(key)
                ):
                    raise ValueError(f"Frozen teacher changed: {key}")
                checked.add(key)
    if set(state) - checked == {"lm_head.weight"}:
        if not torch.equal(state["lm_head.weight"], state["model.embed_tokens.weight"]):
            raise ValueError("Teacher's tied output head changed")
        checked.add("lm_head.weight")
    if set(state) != checked:
        raise ValueError(f"Unchecked teacher tensors: {set(state) - checked}")
    result = {
        "status": "passed",
        "teacher_tensors_bitwise_equal": len(checked),
        "step": trainer.state.global_step,
    }
    (Path(trainer.args.output_dir) / "teacher-verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    record_io_timing(
        trainer.args.output_dir,
        trainer.state.global_step,
        "teacher_verification",
        started_at_unix=started_unix,
        finished_at_unix=time.time(),
        seconds=time.perf_counter() - started,
    )
    torch.set_num_threads(previous_threads)


class PhaseTimer:
    def __init__(self):
        self.active = False
        self.stack = []
        self.records = []

    def start(self, name):
        if not self.active:
            return None
        start = (
            torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        )
        if start is not None:
            start.record()
        self.stack.append(name)
        return name, time.perf_counter(), start

    def finish(self, handle):
        if handle is None:
            return
        name, started, start = handle
        end = torch.cuda.Event(enable_timing=True) if start is not None else None
        if end is not None:
            end.record()
        self.records.append((name, time.perf_counter() - started, start, end))
        if self.stack.pop() != name:
            raise RuntimeError("Unbalanced timing phases")

    @contextmanager
    def phase(self, name):
        handle = self.start(name)
        try:
            yield
        finally:
            self.finish(handle)

    def collect(self):
        # Caller synchronizes once at the existing optimizer-step boundary.
        totals = defaultdict(float)
        for name, wall, start, end in self.records:
            totals[f"{name}_wall_seconds"] += wall
            totals[f"{name}_calls"] += 1
            if start is not None:
                totals[f"{name}_cuda_seconds"] += start.elapsed_time(end) / 1000
        self.records.clear()
        return dict(totals)


class ExperimentTiming(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.timer = PhaseTimer()
        self.cumulative_seconds = 0.0
        self.supervised_token_count = 0
        self.last_end_unix = None
        trainer._experiment_timing = self
        self.originals = []
        self._wrap(trainer, "compute_loss", lambda: "loss_total")
        self._wrap(trainer.accelerator, "backward", lambda: "backward")
        self._wrap(trainer, "_clip_grad_norm", lambda: "gradient_clipping")
        self._wrap(trainer, "_generate_and_score_completions", lambda: "rollout")
        self._wrap(trainer.model, "forward", self._student_phase)
        if getattr(trainer, "ref_model", None) is not None:
            self._wrap(trainer.ref_model, "forward", lambda: "teacher_forward")
            head = trainer.ref_model.get_output_embeddings()
            self._wrap(
                head,
                "forward",
                lambda: (
                    None
                    if "teacher_forward" in self.timer.stack
                    else "teacher_projection"
                ),
            )

    def _student_phase(self):
        if "backward" in self.timer.stack:
            return "student_recompute"
        return (
            "student_forward"
            if "loss_total" in self.timer.stack
            else "student_aux_forward"
        )

    def _wrap(self, owner, attribute, phase_name):
        original = getattr(owner, attribute, None)
        if original is None:
            return

        @wraps(original)
        def measured(*args, **kwargs):
            name = phase_name()
            if not self.timer.active or name is None:
                return original(*args, **kwargs)
            if attribute == "compute_loss" and not hasattr(
                self.trainer, "_cum_num_tokens"
            ):
                inputs = kwargs.get("inputs", args[1] if len(args) > 1 else {})
                if "labels" in inputs:
                    self.supervised_token_count += (
                        inputs["labels"][..., 1:] != -100
                    ).sum()
            with self.timer.phase(name):
                return original(*args, **kwargs)

        self.originals.append((owner, attribute, original))
        setattr(owner, attribute, measured)

    def tokens(self):
        value = getattr(
            self.trainer,
            "_cum_num_tokens",
            getattr(
                self.trainer, "_cum_completion_tokens", self.supervised_token_count
            ),
        )
        return value.item() if isinstance(value, torch.Tensor) else value

    def on_step_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.started = time.perf_counter()
        self.started_unix = time.time()
        self.previous_tokens = self.tokens()
        self.timer.records.clear()
        self.timer.active = True

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self.optimizer_handle = self.timer.start("optimizer")

    def on_optimizer_step(self, args, state, control, **kwargs):
        self.timer.finish(self.optimizer_handle)

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        finished_unix = time.time()
        seconds = time.perf_counter() - self.started
        self.timer.active = False
        phases = self.timer.collect()
        self.cumulative_seconds += seconds
        state.active_training_seconds = self.cumulative_seconds
        tokens = self.tokens() - self.previous_tokens
        metrics = {
            "step_seconds": seconds,
            "active_training_seconds": self.cumulative_seconds,
            "completion_tokens": tokens,
            "tokens_per_second": tokens / max(seconds, 1e-9),
            "training_peak_gpu_bytes": torch.cuda.max_memory_allocated()
            if torch.cuda.is_available()
            else 0,
            **phases,
        }
        if "loss_total_cuda_seconds" in phases:
            metrics["loss_other_cuda_seconds"] = max(
                0.0,
                phases["loss_total_cuda_seconds"]
                - sum(
                    phases.get(f"{name}_cuda_seconds", 0.0)
                    for name in (
                        "student_forward",
                        "teacher_forward",
                        "teacher_projection",
                    )
                ),
            )
            metrics["training_compute_cuda_seconds"] = sum(
                phases.get(f"{name}_cuda_seconds", 0.0)
                for name in ("loss_total", "backward", "gradient_clipping", "optimizer")
            )
        for key, values in self.trainer._metrics["train"].items():
            if (
                key.startswith(("paired/", "onpolicy/"))
                and key.endswith("_seconds")
                and values
            ):
                metrics[key] = sum(values) / len(values)
        for key, value in metrics.items():
            if not key.startswith(("paired/", "onpolicy/")):
                self.trainer._metrics["train"][f"runtime/{key}"].append(value)
        row = {
            "step": state.global_step,
            "started_at_unix": self.started_unix,
            "finished_at_unix": finished_unix,
            "between_steps_seconds": max(0.0, self.started_unix - self.last_end_unix)
            if self.last_end_unix is not None
            else None,
            **metrics,
        }
        self.last_end_unix = finished_unix
        with (Path(args.output_dir) / "step-timings.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    def on_train_end(self, args, state, control, **kwargs):
        self.timer.active = False
        for owner, attribute, original in reversed(self.originals):
            setattr(owner, attribute, original)
        self.originals.clear()
