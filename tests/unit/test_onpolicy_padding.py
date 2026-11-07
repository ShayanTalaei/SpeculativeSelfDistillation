"""Padding removal preserves the original on-policy loss and gradients."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from completion_padding import trim_completion_padding


def test_lengths_follow_shuffle_and_accumulation_splits():
    from trl.trainer.utils import shuffle_sequence_dict, split_tensor_dict

    inputs = {
        "completion_ids": torch.tensor([[11, 0, 0, 0], [21, 22, 0, 0], [31, 32, 33, 0], [41, 42, 43, 44]]),
        "completion_mask": torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]),
        "completion_lengths": [1, 2, 3, 4],
    }
    for batch in split_tensor_dict(shuffle_sequence_dict(inputs), 4):
        trimmed = trim_completion_padding(batch)
        n = batch["completion_lengths"][0]
        assert trimmed["completion_ids"].shape == (1, n)
        assert trimmed["completion_mask"].all()
        assert trimmed["completion_ids"][0, 0].item() // 10 == n
    assert inputs["completion_ids"].shape == (4, 4)


def test_lengths_are_independent_of_the_loss_mask_and_slice_aligned_fields():
    inputs = {
        "prompt_ids": torch.ones(2, 3, dtype=torch.long),
        "completion_ids": torch.ones(2, 7, dtype=torch.long),
        "completion_mask": torch.zeros(2, 7),
        "completion_lengths": [4, 2],
        "token_type_ids": torch.zeros(2, 10),
        "advantages": torch.ones(2),
        "importance_sampling_ratio": torch.ones(2, 7),
        "old_per_token_logps": torch.zeros(2, 7),
        "ref_per_token_logps": torch.zeros(2, 7),
    }
    trimmed = trim_completion_padding(inputs)
    assert trimmed["completion_ids"].shape == (2, 4)
    assert trimmed["token_type_ids"].shape == (2, 7)
    assert trimmed["advantages"] is inputs["advantages"]
    for key in ("importance_sampling_ratio", "old_per_token_logps", "ref_per_token_logps"):
        assert trimmed[key].shape == (2, 4)
    assert inputs["completion_ids"].shape == (2, 7)


@pytest.mark.parametrize("importance_sampling", [False, True])
@pytest.mark.parametrize("skip", [0, 1])
def test_actual_onpolicy_forward_loss_and_gradients_match_padded(importance_sampling, skip):
    from distil_trainer import DistilTrainer
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(42)
    config = Qwen3Config(
        vocab_size=23, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, attention_dropout=0.0, pad_token_id=0,
    )
    student = Qwen3ForCausalLM(config)
    teacher = Qwen3ForCausalLM(config).eval()
    trainer = SimpleNamespace(
        model=student, ref_model=teacher, model_kwarg_keys={"logits_to_keep"},
        temperature=1.0, num_loss_tokens_to_skip=skip,
        current_gradient_accumulation_steps=8, alpha=0.0, kl_strategy="standard",
        beta=0.1, top_entropy_quantile=1.0, use_vllm=True,
        vllm_importance_sampling_correction=importance_sampling,
        generate_from_teacher=False, _mem_logged=True, _kl_mem_logged=True,
        accelerator=SimpleNamespace(gather=lambda x: x),
        _metrics=defaultdict(lambda: defaultdict(list)),
    )
    trainer._get_per_token_logps_and_entropies = lambda *a, **kw: DistilTrainer._get_per_token_logps_and_entropies.__wrapped__(trainer, *a, **kw)
    trainer._compute_kl_per_token = lambda *a: DistilTrainer._compute_kl_per_token(trainer, *a)
    inputs = {
        "prompt_ids": torch.tensor([[2, 3], [4, 5]]),
        "prompt_mask": torch.ones(2, 2, dtype=torch.long),
        "teacher_prompt_ids": torch.tensor([[2, 6, 7], [4, 8, 9]]),
        "teacher_prompt_mask": torch.ones(2, 3, dtype=torch.long),
        "completion_ids": torch.tensor([[10, 11, 12, 0, 0, 0], [13, 0, 0, 0, 0, 0]]),
        "completion_mask": torch.tensor([[1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]]),
        "importance_sampling_ratio": torch.tensor([[0.9, 1.2, 1.1, 2., 2., 2.], [0.8, 2., 2., 2., 2., 2.]]),
        "ref_per_token_logps": torch.zeros(2, 6),
    }
    expected = DistilTrainer._compute_loss(trainer, student, inputs)
    expected_grads = torch.autograd.grad(expected, tuple(student.parameters()))
    actual_inputs = {**inputs, "completion_lengths": [3, 1]}
    actual = DistilTrainer._compute_loss(trainer, student, actual_inputs)
    actual_grads = torch.autograd.grad(actual, tuple(student.parameters()))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-6)
    assert actual_inputs["completion_ids"].shape == (2, 6)
