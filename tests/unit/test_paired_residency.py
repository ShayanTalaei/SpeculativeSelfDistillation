"""Resident inference keeps fresh weights and releases borrowed allocations safely."""

import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
from ssd.paired_runner import PairedRunner

from paired_vllm import cuda_weights


def test_mapping_cache_reuses_live_storage_and_closes_before_remapping(monkeypatch):
    events = []
    source = torch.tensor([1.0, 2.0])

    @contextmanager
    def imported(payload):
        events.append(("open", payload["version"]))
        try:
            yield {"weight": source}
        finally:
            events.append(("close", payload["version"]))

    monkeypatch.setattr(cuda_weights, "import_cuda_weights", imported)
    cache = cuda_weights.CudaWeightMappingCache()
    first, hit = cache.get({"version": 1})
    assert not hit
    source.add_(3)
    second, hit = cache.get({"version": 1})
    assert hit and second is first
    assert second["weight"].tolist() == [4.0, 5.0]
    cache.get({"version": 2})
    assert events == [("open", 1), ("close", 1), ("open", 2)]
    cache.close()
    cache.close()
    assert events[-1] == ("close", 2) and len(events) == 4


def test_resident_rollout_restores_training_and_refreshes_each_version(monkeypatch):
    runner = PairedRunner.__new__(PairedRunner)
    runner.keep_student_resident = True
    runner.cache_weight_handles = True
    runner.process = object()
    runner.sleeping = False
    runner.offloaded = True
    runner.device = "cuda:0"
    runner.phase_evaluation = False
    runner.phase_start = time.perf_counter()
    runner.maps = {"a": {}, "b": {}}
    runner.cuda_weights = {}
    runner.previous_cuda_weight_refs = [object()]
    runner.scratch = Path("/unused")
    runner.sync_count = 0
    runner.trainer = SimpleNamespace(
        args=SimpleNamespace(
            paired_keep_training_resident=True, sync_ref_model=False, seed=42
        ),
        state=SimpleNamespace(global_step=3),
        _metrics={"train": defaultdict(list)},
    )
    calls, moves = [], []
    runner._call = lambda command, payload=None: calls.append((command, payload))
    runner._move_training = moves.append
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    for step in (3, 4):
        runner.trainer.state.global_step = step
        runner.sleep()
        assert not runner.sleeping and not runner.offloaded
        assert runner.phase_start is None
        runner.wake_up()
        runner._sync()
    assert moves == ["cuda:0"]
    assert [c[0] for c in calls] == ["SYNC", "SYNC"]
    assert [c[1]["version"] for c in calls] == ["3", "4"]
    assert calls[0][1]["sync_teacher"] and not calls[1][1]["sync_teacher"]
    assert not runner.previous_cuda_weight_refs
