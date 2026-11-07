from types import SimpleNamespace

import pytest

from paired_vllm.lockstep_engine import LockstepPairedEngine
from paired_vllm.lockstep_training import LockstepTrainingBridge


def test_eval_bridge_bypasses_paired_generation_and_target_capture():
    payload = {"eval": True, "student": [[1], [2]]}
    bridge = object.__new__(LockstepTrainingBridge)
    bridge.engine = SimpleNamespace(evaluate_student=lambda value: value)
    assert bridge.call("GENERATE", payload) is payload


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("sleeping", True),
        ("weights_ready", False),
        ("target_leases", {"live"}),
        ("closed", True),
    ],
)
def test_student_eval_requires_synced_weights_and_released_targets(attribute, value):
    engine = object.__new__(LockstepPairedEngine)
    engine.closed = engine.sleeping = False
    engine.weights_ready = True
    engine.target_leases = set()
    setattr(engine, attribute, value)
    with pytest.raises(RuntimeError):
        engine.evaluate_student({})
