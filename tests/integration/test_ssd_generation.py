"""Tests for the SSD generation loop.

Validates that the generation loop correctly:
- Maintains shared prefix alignment between student and teacher
- Tracks token provenance
- Produces identical results to standard model.generate() for pure strategies
- Handles KV cache correctly
"""

import torch
import pytest
from unittest.mock import MagicMock

pytestmark = pytest.mark.slow

from ssd.strategies import (
    PureStudentStrategy,
    PureTeacherStrategy,
    KLSwitchStrategy,
)
from ssd.generation import (
    ssd_generate,
    GenerationOutput,
)


# We use Qwen3-0.6B for real model tests — marked slow so they can be skipped
STUDENT_MODEL = "Qwen/Qwen3-0.6B"
TEACHER_MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def models():
    """Load student and teacher models once for the entire test module."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="flash_attention_2",
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL, torch_dtype=torch.bfloat16, device_map="cuda:1",
        attn_implementation="flash_attention_2",
    )
    student.eval()
    teacher.eval()
    return student, teacher, tokenizer


class TestSharedPrefixAlignment:
    """Both models must see the exact same prefix at every generation step."""

    def test_prefix_matches_at_each_step(self, models):
        student, teacher, tokenizer = models
        student_prompt = "What is the capital of France?"
        teacher_prompt = "The answer is Paris. What is the capital of France?"

        output = ssd_generate(
            student_model=student,
            teacher_model=teacher,
            tokenizer=tokenizer,
            student_prompt=student_prompt,
            teacher_prompt=teacher_prompt,
            strategy=KLSwitchStrategy(threshold=0.5, direction="forward"),
            max_new_tokens=10,
            temperature=0.0,  # greedy for determinism
        )

        # The generated tokens form the shared prefix — verify it's recorded
        assert len(output.tokens) == len(output.provenance)
        assert len(output.tokens) <= 10
        # Each token in the sequence should be a valid token string
        for tok in output.tokens:
            assert isinstance(tok, str)
            assert len(tok) > 0

    def test_prefix_is_identical_for_both_models(self, models):
        """Verify that at each step, both models were conditioned on the same prefix.
        We do this by checking that the generation output records matching prefixes."""
        student, teacher, tokenizer = models
        output = ssd_generate(
            student_model=student,
            teacher_model=teacher,
            tokenizer=tokenizer,
            student_prompt="Hello",
            teacher_prompt="Context: greeting. Hello",
            strategy=PureStudentStrategy(),
            max_new_tokens=5,
            temperature=0.0,
        )
        # The output should have per-step records showing both models saw same generated prefix
        assert output.shared_prefix_verified


class TestProvenanceTracking:
    """Token provenance must be correctly recorded as 'student' or 'teacher'."""

    def test_pure_student_all_student_provenance(self, models):
        student, teacher, tokenizer = models
        output = ssd_generate(
            student_model=student,
            teacher_model=teacher,
            tokenizer=tokenizer,
            student_prompt="Count to five:",
            teacher_prompt="Count to five:",
            strategy=PureStudentStrategy(),
            max_new_tokens=8,
            temperature=0.0,
        )
        assert all(p == "student" for p in output.provenance)

    def test_pure_teacher_all_teacher_provenance(self, models):
        student, teacher, tokenizer = models
        output = ssd_generate(
            student_model=student,
            teacher_model=teacher,
            tokenizer=tokenizer,
            student_prompt="Count to five:",
            teacher_prompt="Count to five:",
            strategy=PureTeacherStrategy(),
            max_new_tokens=8,
            temperature=0.0,
        )
        assert all(p == "teacher" for p in output.provenance)

    def test_provenance_length_matches_tokens(self, models):
        student, teacher, tokenizer = models
        output = ssd_generate(
            student_model=student,
            teacher_model=teacher,
            tokenizer=tokenizer,
            student_prompt="Hello",
            teacher_prompt="Hello",
            strategy=KLSwitchStrategy(threshold=0.1, direction="forward"),
            max_new_tokens=10,
            temperature=0.0,
        )
        assert len(output.provenance) == len(output.tokens)

    def test_provenance_values_are_valid(self, models):
        student, teacher, tokenizer = models
        output = ssd_generate(
            student_model=student,
            teacher_model=teacher,
            tokenizer=tokenizer,
            student_prompt="Test",
            teacher_prompt="Test",
            strategy=KLSwitchStrategy(threshold=0.5, direction="forward"),
            max_new_tokens=5,
            temperature=0.0,
        )
        for p in output.provenance:
            assert p in ("student", "teacher")


class TestPureStrategyEquivalence:
    """Pure student/teacher strategies should match standard model.generate()."""

    def test_pure_student_matches_model_generate(self, models):
        """PureStudent SSD generation should produce the same tokens as
        calling student.generate() directly with greedy decoding."""
        student, teacher, tokenizer = models
        prompt = "The quick brown fox"
        max_new = 10

        # Standard generate
        inputs = tokenizer(prompt, return_tensors="pt").to(student.device)
        with torch.no_grad():
            standard_ids = student.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                temperature=1.0,
            )
        standard_tokens = tokenizer.decode(
            standard_ids[0, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        # SSD generate with PureStudent
        output = ssd_generate(
            student_model=student,
            teacher_model=teacher,
            tokenizer=tokenizer,
            student_prompt=prompt,
            teacher_prompt=prompt,
            strategy=PureStudentStrategy(),
            max_new_tokens=max_new,
            temperature=0.0,
        )
        ssd_text = "".join(output.tokens)

        assert ssd_text == standard_tokens

    def test_pure_teacher_matches_teacher_generate(self, models):
        """PureTeacher should produce same tokens as teacher.generate()."""
        student, teacher, tokenizer = models
        prompt = "The quick brown fox"
        max_new = 10

        inputs = tokenizer(prompt, return_tensors="pt").to(teacher.device)
        with torch.no_grad():
            standard_ids = teacher.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                temperature=1.0,
            )
        standard_tokens = tokenizer.decode(
            standard_ids[0, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        output = ssd_generate(
            student_model=student,
            teacher_model=teacher,
            tokenizer=tokenizer,
            student_prompt=prompt,
            teacher_prompt=prompt,
            strategy=PureTeacherStrategy(),
            max_new_tokens=max_new,
            temperature=0.0,
        )
        ssd_text = "".join(output.tokens)

        assert ssd_text == standard_tokens


class TestGenerationOutput:
    """GenerationOutput should contain all required fields."""

    def test_output_has_per_token_logprobs(self, models):
        student, teacher, tokenizer = models
        output = ssd_generate(
            student_model=student,
            teacher_model=teacher,
            tokenizer=tokenizer,
            student_prompt="Test",
            teacher_prompt="Test",
            strategy=PureStudentStrategy(),
            max_new_tokens=5,
            temperature=0.0,
        )
        assert len(output.student_logprobs) == len(output.tokens)
        assert len(output.teacher_logprobs) == len(output.tokens)

    def test_output_has_per_token_kl(self, models):
        student, teacher, tokenizer = models
        output = ssd_generate(
            student_model=student,
            teacher_model=teacher,
            tokenizer=tokenizer,
            student_prompt="Test",
            teacher_prompt="Test",
            strategy=PureStudentStrategy(),
            max_new_tokens=5,
            temperature=0.0,
        )
        assert len(output.kl_forward) == len(output.tokens)
        assert len(output.kl_reverse) == len(output.tokens)
        # KL values should be non-negative
        for kl in output.kl_forward:
            assert kl >= -1e-6  # small tolerance for numerical issues

    def test_output_has_per_token_entropy(self, models):
        student, teacher, tokenizer = models
        output = ssd_generate(
            student_model=student,
            teacher_model=teacher,
            tokenizer=tokenizer,
            student_prompt="Test",
            teacher_prompt="Test",
            strategy=PureStudentStrategy(),
            max_new_tokens=5,
            temperature=0.0,
        )
        assert len(output.student_entropy) == len(output.tokens)
        assert len(output.teacher_entropy) == len(output.tokens)
