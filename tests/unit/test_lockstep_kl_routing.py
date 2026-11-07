import pytest
import torch

from paired_vllm.async_engine import AsyncConfig
from paired_vllm.core import PairConfig
from paired_vllm.engine import RuntimeConfig
from paired_vllm.lockstep_engine import LockstepConfig
from paired_vllm.lockstep_head import routing_score


def test_reverse_kl_threshold_one_does_not_use_forward_kl_or_jsd():
    teacher = torch.tensor([[0.99, 0.01], [0.5, 0.5]]).log()
    student = teacher.flip(0)
    expected = torch.distributions.kl_divergence(
        torch.distributions.Categorical(logits=student),
        torch.distributions.Categorical(logits=teacher),
    )
    actual = routing_score("kl_reverse:1.0")(teacher, student)
    torch.testing.assert_close(actual, expected)
    assert (actual > 1).tolist() == [True, False]
    assert (routing_score("kl_forward:1.0")(teacher, student) > 1).tolist() == [False, True]
    assert not (routing_score("jsd:1.0")(teacher, student) > 1).any()


def test_only_lockstep_accepts_kl_routing():
    for policy in ("kl_reverse:1", "kl_forward:1"):
        runtime = RuntimeConfig(
            "teacher", "student", PairConfig(capacity=128, token_budget=32, max_requests=2),
            policy=policy, max_model_len=64,
        )
        assert LockstepConfig(runtime).threshold == 1
        with pytest.raises(ValueError, match="unsupported routing"):
            AsyncConfig(runtime)
