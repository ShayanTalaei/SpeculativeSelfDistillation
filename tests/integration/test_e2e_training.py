"""End-to-end training test with Qwen3-0.6B.

Tests the full training pipeline:
- SSD generation with different strategies
- KL loss computation with provenance masking
- Gradient flow through the student model
- Mini-batch updates
- Rollout logging

Uses the standalone training loop for simplicity (TRL integration tested separately).
"""

import shutil
import tempfile
from pathlib import Path

import torch
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

pytestmark = pytest.mark.slow

from ssd.generation import ssd_generate
from ssd.loss import compute_kl_loss, compute_masked_loss
from ssd.strategies import (
    PureStudentStrategy,
    PureTeacherStrategy,
    KLSwitchStrategy,
    EntropySwitchStrategy,
)
from ssd.rollout_logger import RolloutLogger


MODEL_ID = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def models():
    """Load student (trainable) and teacher (frozen) on separate GPUs."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    student = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="flash_attention_2",
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda:1",
        attn_implementation="flash_attention_2",
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return student, teacher, tokenizer


# -- Sample data for testing --------------------------------------------------

SAMPLES = [
    {
        "student_prompt": "What medication was prescribed to the patient on February 10th?",
        "teacher_prompt": (
            "Clinical Document:\n"
            "Patient John Miller, ID patient_05.\n"
            "Feb 10: Prescribed Metformin 500mg twice daily for diabetes management.\n"
            "Feb 15: Follow-up showed improved glucose levels.\n\n"
            "Question: What medication was prescribed to the patient on February 10th?"
        ),
    },
    {
        "student_prompt": "What was the hemoglobin level on admission?",
        "teacher_prompt": (
            "Clinical Document:\n"
            "Patient Anna Sample, ID patient_03.\n"
            "Admission labs: Hemoglobin 8.2 g/dL, WBC 12.3, Platelets 145.\n"
            "Transfusion of 2 units pRBC ordered.\n\n"
            "Question: What was the hemoglobin level on admission?"
        ),
    },
]


class TestGradientFlow:
    """Verify that gradients flow through the student during training."""

    def test_student_params_update_after_backward(self, models):
        student, teacher, tokenizer = models
        student.train()

        # Record initial weights
        param = next(student.parameters())
        initial_weight = param.data.clone()

        # Generate rollout
        gen = ssd_generate(
            student_model=student,
            teacher_model=teacher,
            tokenizer=tokenizer,
            student_prompt=SAMPLES[0]["student_prompt"],
            teacher_prompt=SAMPLES[0]["teacher_prompt"],
            strategy=PureTeacherStrategy(),
            max_new_tokens=8,
            temperature=0.0,
        )

        # Build sequences for loss computation
        s_ids = tokenizer.encode(SAMPLES[0]["student_prompt"], return_tensors="pt").to(student.device)
        t_ids = tokenizer.encode(SAMPLES[0]["teacher_prompt"], return_tensors="pt").to(teacher.device)
        gen_tensor = torch.tensor([gen.token_ids], dtype=torch.long)

        s_input = torch.cat([s_ids, gen_tensor.to(student.device)], dim=1)
        t_input = torch.cat([t_ids, gen_tensor.to(teacher.device)], dim=1)

        s_prompt_len = s_ids.shape[1]
        gen_len = gen_tensor.shape[1]

        # Forward student (with grads)
        s_out = student(input_ids=s_input, use_cache=False)
        s_logits = s_out.logits[:, s_prompt_len - 1 : s_prompt_len + gen_len - 1, :]

        # Forward teacher (no grads)
        with torch.no_grad():
            t_out = teacher(input_ids=t_input, use_cache=False)
            t_prompt_len = t_ids.shape[1]
            t_logits = t_out.logits[:, t_prompt_len - 1 : t_prompt_len + gen_len - 1, :]
            t_logits = t_logits.to(s_logits.device)

        # Compute KL loss
        mask = torch.ones(1, gen_len, device=s_logits.device)
        loss_per_token = compute_masked_loss(s_logits, t_logits, gen.provenance, mask, strategy="all")
        loss = loss_per_token.sum() / gen_len

        # Backward
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify weights changed
        assert not torch.equal(param.data, initial_weight), "Student weights did not update after backward"

    def test_teacher_params_unchanged(self, models):
        """Teacher model weights must never change."""
        student, teacher, tokenizer = models

        teacher_param = next(teacher.parameters())
        initial_weight = teacher_param.data.clone()

        gen = ssd_generate(
            student_model=student,
            teacher_model=teacher,
            tokenizer=tokenizer,
            student_prompt=SAMPLES[0]["student_prompt"],
            teacher_prompt=SAMPLES[0]["teacher_prompt"],
            strategy=KLSwitchStrategy(threshold=0.5, direction="forward"),
            max_new_tokens=5,
            temperature=0.0,
        )

        assert torch.equal(teacher_param.data, initial_weight), "Teacher weights changed!"


class TestMultiStepTraining:
    """Test running multiple training steps with different strategies."""

    def _run_steps(self, models, strategy, num_steps=3, max_tokens=8):
        student, teacher, tokenizer = models
        student.train()
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5)
        losses = []

        for step in range(num_steps):
            sample = SAMPLES[step % len(SAMPLES)]

            # Generate
            student.eval()
            gen = ssd_generate(
                student_model=student, teacher_model=teacher, tokenizer=tokenizer,
                student_prompt=sample["student_prompt"],
                teacher_prompt=sample["teacher_prompt"],
                strategy=strategy, max_new_tokens=max_tokens, temperature=0.0,
            )
            student.train()

            if not gen.token_ids:
                continue

            # Compute loss
            s_ids = tokenizer.encode(sample["student_prompt"], return_tensors="pt").to(student.device)
            t_ids = tokenizer.encode(sample["teacher_prompt"], return_tensors="pt").to(teacher.device)
            g = torch.tensor([gen.token_ids], dtype=torch.long)
            s_input = torch.cat([s_ids, g.to(student.device)], dim=1)
            t_input = torch.cat([t_ids, g.to(teacher.device)], dim=1)

            s_out = student(input_ids=s_input, use_cache=False)
            s_logits = s_out.logits[:, s_ids.shape[1]-1:s_ids.shape[1]+g.shape[1]-1, :]
            with torch.no_grad():
                t_out = teacher(input_ids=t_input, use_cache=False)
                t_logits = t_out.logits[:, t_ids.shape[1]-1:t_ids.shape[1]+g.shape[1]-1, :]
                t_logits = t_logits.to(s_logits.device)

            mask = torch.ones(1, g.shape[1], device=s_logits.device)
            ptl = compute_masked_loss(s_logits, t_logits, gen.provenance, mask, strategy="all")
            loss = ptl.sum() / g.shape[1]
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return losses

    def test_pure_student_training(self, models):
        losses = self._run_steps(models, PureStudentStrategy())
        assert len(losses) > 0
        assert all(l >= 0 for l in losses)

    def test_pure_teacher_training(self, models):
        losses = self._run_steps(models, PureTeacherStrategy())
        assert len(losses) > 0

    def test_kl_switch_training(self, models):
        losses = self._run_steps(models, KLSwitchStrategy(threshold=0.5, direction="forward"))
        assert len(losses) > 0

    def test_entropy_switch_training(self, models):
        losses = self._run_steps(models, EntropySwitchStrategy(threshold=2.0))
        assert len(losses) > 0


class TestLossMaskingStrategies:
    """Test that different loss masking strategies produce different loss values."""

    def _compute_loss_with_mask(self, models, mask_strategy):
        student, teacher, tokenizer = models
        student.eval()

        gen = ssd_generate(
            student_model=student, teacher_model=teacher, tokenizer=tokenizer,
            student_prompt=SAMPLES[0]["student_prompt"],
            teacher_prompt=SAMPLES[0]["teacher_prompt"],
            strategy=KLSwitchStrategy(threshold=0.1, direction="forward"),
            max_new_tokens=10, temperature=0.0,
        )

        if not gen.token_ids:
            return 0.0, gen.provenance

        s_ids = tokenizer.encode(SAMPLES[0]["student_prompt"], return_tensors="pt").to(student.device)
        t_ids = tokenizer.encode(SAMPLES[0]["teacher_prompt"], return_tensors="pt").to(teacher.device)
        g = torch.tensor([gen.token_ids], dtype=torch.long)

        s_input = torch.cat([s_ids, g.to(student.device)], dim=1)
        t_input = torch.cat([t_ids, g.to(teacher.device)], dim=1)

        with torch.no_grad():
            s_out = student(input_ids=s_input, use_cache=False)
            s_logits = s_out.logits[:, s_ids.shape[1]-1:s_ids.shape[1]+g.shape[1]-1, :]
            t_out = teacher(input_ids=t_input, use_cache=False)
            t_logits = t_out.logits[:, t_ids.shape[1]-1:t_ids.shape[1]+g.shape[1]-1, :]
            t_logits = t_logits.to(s_logits.device)

        mask = torch.ones(1, g.shape[1], device=s_logits.device)
        ptl = compute_masked_loss(s_logits, t_logits, gen.provenance, mask, strategy=mask_strategy)
        return ptl.sum().item(), gen.provenance

    def test_teacher_only_masks_student_tokens(self, models):
        loss_all, prov = self._compute_loss_with_mask(models, "all")
        loss_teacher, _ = self._compute_loss_with_mask(models, "teacher_only")
        # teacher_only should be <= all (fewer active tokens)
        assert loss_teacher <= loss_all + 1e-4

    def test_student_only_masks_teacher_tokens(self, models):
        loss_all, prov = self._compute_loss_with_mask(models, "all")
        loss_student, _ = self._compute_loss_with_mask(models, "student_only")
        assert loss_student <= loss_all + 1e-4


class TestRolloutLogging:
    """Test that rollout logging produces valid output files."""

    def test_logger_creates_files(self, models):
        student, teacher, tokenizer = models
        tmpdir = tempfile.mkdtemp()

        logger = RolloutLogger(output_dir=tmpdir, log_interval=1)

        gen = ssd_generate(
            student_model=student, teacher_model=teacher, tokenizer=tokenizer,
            student_prompt="Test question",
            teacher_prompt="Context. Test question",
            strategy=PureStudentStrategy(),
            max_new_tokens=5, temperature=0.0,
        )

        logger.log_rollout(
            step=1,
            student_prompt="Test question",
            teacher_prompt="Context. Test question",
            gen_output=gen,
            per_token_loss=[0.1] * len(gen.tokens),
            strategy_name="pure_student",
            strategy_config={},
            total_loss=0.5,
        )
        logger.flush(1)

        # Verify file was created
        log_dir = Path(tmpdir) / "rollout_logs"
        files = list(log_dir.glob("*.json"))
        assert len(files) == 1

        # Verify file is valid JSON with correct structure
        rollouts = logger.load_all()
        assert len(rollouts) == 1
        assert rollouts[0].step == 1
        assert len(rollouts[0].tokens) == len(gen.tokens)
        assert rollouts[0].tokens[0].source in ("student", "teacher")

        shutil.rmtree(tmpdir)
