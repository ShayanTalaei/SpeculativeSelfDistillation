"""The fused EMA update must match the per-parameter reference bit for bit."""

import torch

from distil_trainer import MemoryEfficientSyncRefModelCallback


import pytest


def _models(dtype, device="cpu"):
    torch.manual_seed(0)
    student = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.Linear(32, 8)).to(dtype).to(device)
    teacher = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.Linear(32, 8)).to(dtype).to(device)
    return student, teacher


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fused_ema_matches_reference_rounding_in_bf16(device):
    from accelerate import Accelerator

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    Accelerator(cpu=True)  # The callback consults AcceleratorState for ZeRO-3.
    student, teacher = _models(torch.bfloat16, device)
    reference = {n: p.detach().clone() for n, p in teacher.named_parameters()}
    for _ in range(5):
        for s, (n, r) in zip(student.parameters(), reference.items()):
            r.mul_(1.0 - 0.01).add_(s.data, alpha=0.01)
        MemoryEfficientSyncRefModelCallback.sync_target_model_memory_efficient(student, teacher, 0.01)
    for n, p in teacher.named_parameters():
        assert torch.equal(p.data, reference[n]), n


def test_fused_ema_rejects_mismatched_parameter_order():
    from accelerate import Accelerator

    Accelerator(cpu=True)
    student, _ = _models(torch.float32)
    teacher = torch.nn.Sequential(torch.nn.Linear(64, 8))
    try:
        MemoryEfficientSyncRefModelCallback.sync_target_model_memory_efficient(student, teacher, 0.01)
    except ValueError as error:
        assert "order mismatch" in str(error)
    else:
        raise AssertionError("expected a parameter order mismatch error")
