"""Contracts for feeding rollout targets into the original trainer's loss."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Self-Distillation"))
from ssd.paired_cache import FullTeacherCache, full_teacher_logps
from ssd.teacher_cache import (
    TeacherTopKCache,
    build_partition_logprobs,
    capture_topk_rows,
    stack_caches,
)


def test_checkpoint_load_restores_noninitial_ema_and_invalidates_rollout_weights(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from distil_trainer import DistilTrainer

    trainer = object.__new__(DistilTrainer)
    trainer.args = SimpleNamespace(persist_ref_model=True)
    trainer.ref_model = torch.nn.Linear(3, 2, bias=False)
    expected = {"weight": torch.full_like(trainer.ref_model.weight, 7.0)}
    torch.save(expected, tmp_path / "teacher_state.pt")
    trainer._last_loaded_step = trainer._last_synced_step = 12
    parent = DistilTrainer.__mro__[1]
    monkeypatch.setattr(
        parent, "_load_from_checkpoint", lambda *a, **kw: "student restored"
    )
    assert trainer._load_from_checkpoint(str(tmp_path)) == "student restored"
    torch.testing.assert_close(trainer.ref_model.weight, expected["weight"])
    assert trainer._last_loaded_step == trainer._last_synced_step == -1


def test_resume_rejects_missing_ema_checkpoint(tmp_path):
    from types import SimpleNamespace

    from distil_trainer import DistilTrainer

    trainer = object.__new__(DistilTrainer)
    trainer.args = SimpleNamespace(persist_ref_model=True)
    with pytest.raises(ValueError, match="teacher_state.pt"):
        trainer._load_from_checkpoint(str(tmp_path))


@pytest.mark.parametrize("kind", ["hidden", "logits"])
def test_cached_targets_preserve_full_loss_gradient_and_padding(kind):
    torch.manual_seed(42)
    head = torch.nn.Linear(5, 11, bias=False)
    hidden = [torch.randn(3, 5), torch.randn(1, 5)]
    targets = [head(h).detach() for h in hidden]
    caches = [
        FullTeacherCache(kind, h if kind == "hidden" else t, 7, list(range(len(h))))
        for h, t in zip(hidden, targets)
    ]
    # Simulate the original trainer's shuffled accumulation slices.
    caches = caches[::-1]
    ids = torch.tensor([[0, 99, 99], [0, 1, 2]])
    actual = full_teacher_logps(caches, head, ids, 7, 0.8, "cpu")
    expected = torch.zeros(2, 3, 11)
    for i, target in enumerate(targets[::-1]):
        expected[i, : len(target)] = (target / 0.8).log_softmax(-1)
    mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    student = torch.randn(2, 3, 11, requires_grad=True)

    def loss(t):
        kl = torch.nn.functional.kl_div(
            student.log_softmax(-1), t, reduction="none", log_target=True
        ).sum(-1)
        return ((kl * mask).sum(-1) / mask.sum(-1)).mean() / 8

    a, b = loss(actual), loss(expected)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(
        torch.autograd.grad(a, student)[0], torch.autograd.grad(b, student)[0]
    )
    assert not actual.requires_grad


def test_cache_rejects_stale_teacher_and_wrong_completion():
    head = torch.nn.Linear(3, 5, bias=False)
    cache = FullTeacherCache("hidden", torch.ones(2, 3), 9, [1, 2])
    with pytest.raises(ValueError, match="version"):
        full_teacher_logps([cache], head, torch.tensor([[1, 2]]), 10, 1.0, "cpu")
    with pytest.raises(ValueError, match="token"):
        full_teacher_logps([cache], head, torch.tensor([[2, 1]]), 9, 1.0, "cpu")


def test_topk_keeps_existing_union_and_tail_algorithm():
    torch.manual_seed(42)
    student, teacher = torch.randn(2, 17), torch.randn(2, 17)
    idx, val, lse = capture_topk_rows(student, teacher, 100, 1.0)
    cache = TeacherTopKCache(idx, val, lse, 1.0)
    stacked = stack_caches([cache], 2, "cpu")
    p, q, _, _ = build_partition_logprobs(student.log_softmax(-1)[None], *stacked, 1.0)
    actual = torch.nn.functional.kl_div(p, q, reduction="none", log_target=True).sum(-1)
    expected = torch.nn.functional.kl_div(
        student.log_softmax(-1),
        teacher.log_softmax(-1),
        reduction="none",
        log_target=True,
    ).sum(-1)
    torch.testing.assert_close(actual[0], expected, atol=1e-6, rtol=1e-5)


def test_worker_topk_capture_is_identical_to_existing_cache():
    from paired_vllm.training_ops import capture_topk

    torch.manual_seed(43)
    s, t = torch.randn(4, 153), torch.randn(4, 153)
    actual = capture_topk(s, t, 100, 0.7)
    expected = capture_topk_rows(s, t, 100, 0.7)
    for value, reference in zip(actual.values(), expected):
        torch.testing.assert_close(value, reference, atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["hidden", "logits"])
@pytest.mark.parametrize(
    "mask_strategy", ["all", "directional", "teacher_only", "student_only"]
)
@pytest.mark.parametrize("padding", [0, 3])
def test_actual_original_trainer_loss_and_gradient_match_recompute(kind, mask_strategy, padding):
    from collections import defaultdict
    from types import SimpleNamespace

    from distil_trainer import DistilTrainer
    from ssd.distil_trainer import SSDDistilTrainer

    torch.manual_seed(42)
    head = torch.nn.Linear(3, 7, bias=False)
    hidden = torch.randn(2, 3 + padding, 3)
    targets = head(hidden).detach()
    student = torch.randn(2, 3 + padding, 7, requires_grad=True)
    ids = torch.tensor([[2, 1, 3] + [0] * padding, [1, 2, 0] + [0] * padding])
    calls = []
    trainer = SimpleNamespace(
        print_memory_usage=False,
        teacher_cache_mode="off",
        temperature=1.0,
        num_loss_tokens_to_skip=1,
        loss_mask_strategy=mask_strategy,
        current_gradient_accumulation_steps=8,
        alpha=0.0,
        kl_strategy="standard",
        state=SimpleNamespace(global_step=4),
        model=torch.nn.Linear(1, 1),
        ref_model=SimpleNamespace(get_output_embeddings=lambda: head),
        accelerator=SimpleNamespace(gather=lambda x: x),
        _metrics=defaultdict(lambda: defaultdict(list)),
    )

    def forward(model, input_ids, attention_mask, logits_to_keep, **kwargs):
        calls.append(model)
        logps = (targets if model is trainer.ref_model else student)[:, :logits_to_keep].log_softmax(-1)
        return logps.gather(-1, ids[:, :logits_to_keep, None]).squeeze(-1), logps, torch.ones(2, logits_to_keep)

    trainer._get_per_token_logps_and_entropies = forward
    trainer._compute_kl_per_token = lambda s, t: DistilTrainer._compute_kl_per_token(
        trainer, s, t
    )
    inputs = {
        "prompt_ids": ids[:, :1],
        "prompt_mask": torch.ones(2, 1),
        "teacher_prompt_ids": ids[:, :2],
        "teacher_prompt_mask": torch.ones(2, 2),
        "completion_ids": ids,
        "completion_mask": torch.tensor([[1, 1, 1] + [0] * padding, [1, 1, 0] + [0] * padding]),
        "provenance": torch.tensor([[1, 2, 1] + [0] * padding, [2, 1, 0] + [0] * padding]),
    }
    reference = SSDDistilTrainer._compute_loss(trainer, trainer.model, inputs)
    expected_grad = torch.autograd.grad(reference, student)[0]
    inputs["full_teacher_cache"] = [
        FullTeacherCache(
            kind,
            (hidden if kind == "hidden" else targets)[i, :n],
            4,
            ids[i, :n].tolist(),
        )
        for i, n in enumerate((3, 2))
    ]
    calls.clear()
    actual = SSDDistilTrainer._compute_loss(trainer, trainer.model, inputs)
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(torch.autograd.grad(actual, student)[0], expected_grad)
    assert inputs["completion_ids"].shape[1] == 3 + padding
    assert calls == [trainer.model], (
        "reuse must not execute teacher transformer forward"
    )
    trainer.args = SimpleNamespace(paired_backend="lockstep")
    inputs.pop("full_teacher_cache")
    calls.clear()
    with pytest.raises(RuntimeError, match="requires reused teacher hidden"):
        SSDDistilTrainer._compute_loss(trainer, trainer.model, inputs)
    assert calls == [trainer.model], "missing targets must never recompute the teacher"
