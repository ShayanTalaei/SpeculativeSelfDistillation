"""Lightweight per-step profiler for the SSD trainer.

Records cumulative phase timings and CUDA memory snapshots per training step,
flushed to a JSONL file (one record per step). Designed to add ~zero overhead
when disabled and minimal overhead when enabled (no torch.profiler trace, no
NVTX, no event timelines — just `time.perf_counter()` + cuda mem stats).

Usage::

    from ssd.profiling import init_profiler, get_profiler

    init_profiler(output_dir="output/run/profile", enabled=True)
    p = get_profiler()

    p.start_step(step)
    with p.phase("generation/total"):
        ...
        with p.phase("generation/decode_step"):
            ...
    with p.phase("loss/student_forward"):
        ...
    p.snapshot_mem("end_of_step")
    p.dump_step(step)
"""

import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager

import torch


class StepProfiler:
    """Per-step profiler. Accumulates phase timings + memory snapshots, flushes per step."""

    def __init__(self, output_dir: str, enabled: bool = True, sync_cuda: bool = True):
        self.enabled = enabled
        self.sync_cuda = sync_cuda and torch.cuda.is_available()
        self.output_dir = output_dir
        self._timings: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._mem: dict[str, dict] = {}
        self._step: int | None = None
        self._file = None
        if self.enabled:
            os.makedirs(output_dir, exist_ok=True)
            self._file = open(os.path.join(output_dir, "profile.jsonl"), "w")

    @contextmanager
    def phase(self, name: str):
        """Time a code region. Accumulates across multiple calls in the same step."""
        if not self.enabled:
            yield
            return
        if self.sync_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.sync_cuda:
                torch.cuda.synchronize()
            self._timings[name] += time.perf_counter() - t0
            self._counts[name] += 1

    def snapshot_mem(self, label: str) -> None:
        if not self.enabled or not torch.cuda.is_available():
            return
        self._mem[label] = {
            "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 4),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 4),
            "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 4),
        }

    def reset_peak(self) -> None:
        if self.enabled and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def start_step(self, step: int) -> None:
        if not self.enabled:
            return
        self._step = step
        self._timings.clear()
        self._counts.clear()
        self._mem.clear()
        self.reset_peak()

    def dump_step(self, step: int | None = None) -> None:
        if not self.enabled or self._file is None:
            return
        s = step if step is not None else self._step
        record = {
            "step": s,
            "timings_sec": {k: round(v, 6) for k, v in self._timings.items()},
            "counts": dict(self._counts),
            "memory_gb": dict(self._mem),
        }
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()
        self._timings.clear()
        self._counts.clear()
        self._mem.clear()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None


_GLOBAL: StepProfiler | None = None


def init_profiler(output_dir: str, enabled: bool = True) -> StepProfiler:
    """Initialize the module-level profiler. Subsequent get_profiler() returns this."""
    global _GLOBAL
    _GLOBAL = StepProfiler(output_dir=output_dir, enabled=enabled)
    return _GLOBAL


def get_profiler() -> StepProfiler | None:
    """Return the module-level profiler, or None if not initialized."""
    return _GLOBAL


@contextmanager
def maybe_phase(name: str):
    """Context manager that uses the global profiler if available, no-op otherwise."""
    p = get_profiler()
    if p is None or not p.enabled:
        yield
        return
    with p.phase(name):
        yield
