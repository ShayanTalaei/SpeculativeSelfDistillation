"""Tests for batched SSD generation.

Verifies that ssd_generate_batch produces identical results to
calling ssd_generate sequentially for each sample.
"""

import torch
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

pytestmark = pytest.mark.slow

from ssd.generation import ssd_generate, ssd_generate_batch
from ssd.strategies import (
    PureStudentStrategy,
    PureTeacherStrategy,
    KLSwitchStrategy,
)


MODEL_ID = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def models():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    student = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="flash_attention_2",
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda:1",
        attn_implementation="flash_attention_2",
    )
    student.eval()
    teacher.eval()
    return student, teacher, tokenizer


PROMPTS = [
    ("The quick brown fox", "The quick brown fox"),
    ("What is the capital of France?", "Paris is the capital. What is the capital of France?"),
    ("Count to five:", "Count to five:"),
]


class TestBatchMatchesSingle:
    """Batch generation must produce identical tokens to sequential single generation."""

    @pytest.mark.parametrize("strategy_cls", [PureStudentStrategy, PureTeacherStrategy])
    def test_batch_matches_sequential(self, models, strategy_cls):
        student, teacher, tokenizer = models
        strategy = strategy_cls()
        max_new = 10

        s_prompts = [p[0] for p in PROMPTS]
        t_prompts = [p[1] for p in PROMPTS]

        # Sequential (reference)
        sequential_outputs = []
        for sp, tp in PROMPTS:
            out = ssd_generate(
                student_model=student, teacher_model=teacher,
                tokenizer=tokenizer, student_prompt=sp, teacher_prompt=tp,
                strategy=strategy, max_new_tokens=max_new, temperature=0.0,
            )
            sequential_outputs.append(out)

        # Batched
        batch_outputs = ssd_generate_batch(
            student_model=student, teacher_model=teacher,
            tokenizer=tokenizer, student_prompts=s_prompts, teacher_prompts=t_prompts,
            strategy=strategy, max_new_tokens=max_new, temperature=0.0,
        )

        assert len(batch_outputs) == len(sequential_outputs)
        for i, (seq, bat) in enumerate(zip(sequential_outputs, batch_outputs)):
            assert seq.token_ids == bat.token_ids, (
                f"Sample {i}: sequential={seq.token_ids} vs batch={bat.token_ids}"
            )
            assert seq.provenance == bat.provenance


class TestBatchEdgeCases:
    """Edge cases for batched generation."""

    def test_single_sample_batch(self, models):
        """Batch of 1 should work identically to single call."""
        student, teacher, tokenizer = models
        strategy = PureStudentStrategy()

        single = ssd_generate(
            student_model=student, teacher_model=teacher,
            tokenizer=tokenizer, student_prompt="Hello world",
            teacher_prompt="Hello world",
            strategy=strategy, max_new_tokens=5, temperature=0.0,
        )
        batch = ssd_generate_batch(
            student_model=student, teacher_model=teacher,
            tokenizer=tokenizer, student_prompts=["Hello world"],
            teacher_prompts=["Hello world"],
            strategy=strategy, max_new_tokens=5, temperature=0.0,
        )
        assert len(batch) == 1
        assert single.token_ids == batch[0].token_ids

    def test_different_length_prompts(self, models):
        """Prompts of very different lengths should still produce correct results."""
        student, teacher, tokenizer = models
        strategy = PureStudentStrategy()

        s_prompts = ["Hi", "This is a much longer prompt that has many tokens in it"]
        t_prompts = ["Hi", "This is a much longer prompt that has many tokens in it"]

        batch = ssd_generate_batch(
            student_model=student, teacher_model=teacher,
            tokenizer=tokenizer, student_prompts=s_prompts, teacher_prompts=t_prompts,
            strategy=strategy, max_new_tokens=8, temperature=0.0,
        )

        # Verify each sample individually
        for i in range(2):
            single = ssd_generate(
                student_model=student, teacher_model=teacher,
                tokenizer=tokenizer, student_prompt=s_prompts[i],
                teacher_prompt=t_prompts[i],
                strategy=strategy, max_new_tokens=8, temperature=0.0,
            )
            assert single.token_ids == batch[i].token_ids, (
                f"Sample {i}: single={single.token_ids} vs batch={batch[i].token_ids}"
            )

    def test_output_fields_populated(self, models):
        """All GenerationOutput fields should be populated for each sample."""
        student, teacher, tokenizer = models
        batch = ssd_generate_batch(
            student_model=student, teacher_model=teacher,
            tokenizer=tokenizer,
            student_prompts=["Test one", "Test two"],
            teacher_prompts=["Test one", "Test two"],
            strategy=PureStudentStrategy(),
            max_new_tokens=5, temperature=0.0,
        )
        for out in batch:
            n = len(out.tokens)
            assert n > 0
            assert len(out.token_ids) == n
            assert len(out.provenance) == n
            assert len(out.student_logprobs) == n
            assert len(out.teacher_logprobs) == n
            assert len(out.kl_forward) == n
            assert len(out.kl_reverse) == n
            assert len(out.student_entropy) == n
            assert len(out.teacher_entropy) == n
            assert len(out.decisions) == n
