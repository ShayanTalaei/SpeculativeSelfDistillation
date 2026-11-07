"""Checkpoint housekeeping for runs that span several Slurm allocations.

A 14B student checkpoint is large — measured by scaling the 4B one actually
written here (47 GB: 30 GB optimizer + 17 GB fp32 master), a 14.8B model lands
around **177 GB**. That collides with two things at once:

1. ``save_total_limit`` rotates the OLD checkpoint away only *after* the new
   one is written, so the real requirement is ``(limit + 1) x size``, not
   ``limit x size``. At 14B that is ~354 GB for limit=1 — more than the free
   space on this filesystem, so training would die partway through a save with
   a confusing ENOSPC rather than at a sensible boundary.
2. The job has a wall-clock limit, so this happens repeatedly.

``PruneCheckpointsBeforeSave`` deletes stale checkpoints in the window *before*
Trainer writes the new one. It runs from ``on_step_end``, which the training
loop calls immediately before ``_maybe_log_save_evaluate`` — so ``should_save``
is already set and the disk is freed just in time.

It is deliberately conservative: it keeps one previous checkpoint whenever
there is room for two, and only falls back to keeping none when there is not.
Keeping none is a real risk (a crash mid-save leaves nothing to resume from),
so it is taken only when the alternative is a guaranteed ENOSPC.
"""
from __future__ import annotations

import os
import re
import shutil

from transformers import TrainerCallback

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_checkpoints(output_dir: str) -> list[tuple[int, str]]:
    """(step, path) for every checkpoint in output_dir, oldest first."""
    if not os.path.isdir(output_dir):
        return []
    out = []
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if m:
            out.append((int(m.group(1)), os.path.join(output_dir, name)))
    return sorted(out)


def is_complete(path: str) -> bool:
    """Did this checkpoint finish writing?

    ``trainer_state.json`` is written at the END of Trainer._save_checkpoint,
    so its presence is the marker that the save ran to completion. A job killed
    by the wall clock — or crashing partway through the save, which is exactly
    how the FSDP optimizer-gather failure presents — leaves a directory holding
    only model shards. Resuming from one of those fails in a far more confusing
    place than here.
    """
    return (
        os.path.isfile(os.path.join(path, "trainer_state.json"))
        and not os.path.exists(os.path.join(path, ".publication-pending"))
    )


def latest_checkpoint(output_dir: str) -> str | None:
    """Newest COMPLETE checkpoint, or None. Used to auto-resume a chained job."""
    for step, path in reversed(find_checkpoints(output_dir)):
        if is_complete(path):
            return path
        print(f"[ckpt] ignoring incomplete checkpoint {os.path.basename(path)} "
              f"(missing trainer state or publication still pending)", flush=True)
    return None


def _dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


class PruneCheckpointsBeforeSave(TrainerCallback):
    """Free room for the next checkpoint before the Trainer writes it."""

    def __init__(self, output_dir: str, min_keep: int = 1, safety: float = 1.15):
        self.output_dir = output_dir
        self.min_keep = min_keep
        # Checkpoint size varies a little run to run (optimizer state layout,
        # tokenizer files); don't plan to land exactly on empty.
        self.safety = safety
        self._size_hint = 0

    def on_step_end(self, args, state, control, **kwargs):
        # Fires just before _maybe_log_save_evaluate, which is what makes this
        # a *pre*-save hook rather than another post-save rotation.
        if not control.should_save or not state.is_world_process_zero:
            return control

        cks = [
            (step, path) for step, path in find_checkpoints(self.output_dir)
            if not os.path.exists(os.path.join(path, ".publication-pending"))
        ]
        if not cks:
            return control

        if not self._size_hint:
            self._size_hint = _dir_size(cks[-1][1])
        need = int(self._size_hint * self.safety)
        if need <= 0:
            return control

        try:
            free = shutil.disk_usage(self.output_dir).free
        except OSError:
            return control

        if free >= need:
            keep = max(self.min_keep, 1)
        else:
            # Not enough room to hold old + new. Prune unprotected
            # checkpoints, but never sacrifice the best measured weights.
            keep = 0
            print(
                f"[ckpt] only {free / 2**30:.0f} GiB free but the next "
                f"checkpoint needs ~{need / 2**30:.0f} GiB — pruning "
                f"unprotected checkpoints to make room; best weights are retained.",
                flush=True,
            )

        doomed = cks[:-keep] if keep else cks
        best = getattr(state, "best_model_checkpoint", None)
        for step, path in doomed:
            if os.path.exists(os.path.join(path, ".publication-pending")):
                continue
            if best and os.path.realpath(path) == os.path.realpath(best):
                continue
            print(f"[ckpt] pruning {os.path.basename(path)} "
                  f"({_dir_size(path) / 2**30:.0f} GiB)", flush=True)
            shutil.rmtree(path, ignore_errors=True)
        return control
