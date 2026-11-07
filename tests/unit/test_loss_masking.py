"""Tests for loss computation and token provenance-based masking.

Validates that:
- KL loss computation matches manual calculation
- Loss masking correctly applies based on token provenance
- Directional masking uses reverse KL on student tokens and forward KL on teacher tokens
"""

import torch
import torch.nn.functional as F
import pytest

from ssd.loss import (
    compute_kl_loss,
    apply_provenance_mask,
    compute_masked_loss,
)


def make_logits(batch_size: int, seq_len: int, vocab_size: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(batch_size, seq_len, vocab_size)


class TestKLLossComputation:
    """Verify KL divergence computation against manual calculation."""

    def test_forward_kl_matches_manual(self):
        """Forward KL: KL(teacher || student)."""
        vocab = 50
        student_logits = make_logits(1, 1, vocab, seed=0)
        teacher_logits = make_logits(1, 1, vocab, seed=1)

        s_logprobs = F.log_softmax(student_logits, dim=-1)
        t_logprobs = F.log_softmax(teacher_logits, dim=-1)

        # Manual: KL(T||S) = sum(T * (log T - log S))
        t_probs = t_logprobs.exp()
        manual_kl = (t_probs * (t_logprobs - s_logprobs)).sum(-1)

        computed = compute_kl_loss(student_logits, teacher_logits, direction="forward")
        assert torch.allclose(computed, manual_kl, atol=1e-5)

    def test_reverse_kl_matches_manual(self):
        """Reverse KL: KL(student || teacher)."""
        vocab = 50
        student_logits = make_logits(1, 1, vocab, seed=0)
        teacher_logits = make_logits(1, 1, vocab, seed=1)

        s_logprobs = F.log_softmax(student_logits, dim=-1)
        t_logprobs = F.log_softmax(teacher_logits, dim=-1)

        s_probs = s_logprobs.exp()
        manual_kl = (s_probs * (s_logprobs - t_logprobs)).sum(-1)

        computed = compute_kl_loss(student_logits, teacher_logits, direction="reverse")
        assert torch.allclose(computed, manual_kl, atol=1e-5)

    def test_kl_identical_distributions_is_zero(self):
        """KL between identical distributions should be zero."""
        logits = make_logits(2, 5, 100)
        kl = compute_kl_loss(logits, logits, direction="forward")
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)

    def test_kl_is_non_negative(self):
        """KL divergence is always >= 0."""
        student = make_logits(4, 10, 100, seed=42)
        teacher = make_logits(4, 10, 100, seed=99)
        kl_fwd = compute_kl_loss(student, teacher, direction="forward")
        kl_rev = compute_kl_loss(student, teacher, direction="reverse")
        assert (kl_fwd >= -1e-6).all()
        assert (kl_rev >= -1e-6).all()

    def test_batch_and_sequence_dimensions(self):
        """KL output shape should be (batch, seq_len) — summed over vocab."""
        B, L, V = 3, 7, 200
        student = make_logits(B, L, V)
        teacher = make_logits(B, L, V, seed=1)
        kl = compute_kl_loss(student, teacher, direction="forward")
        assert kl.shape == (B, L)


class TestProvenanceMask:
    """Test that provenance masks correctly select tokens."""

    def test_mask_all_no_masking(self):
        """'all' strategy: every token contributes to loss."""
        provenance = ["student", "teacher", "student", "teacher", "student"]
        mask = apply_provenance_mask(provenance, strategy="all")
        assert mask.shape == (5,)
        assert mask.sum() == 5

    def test_mask_teacher_only(self):
        """'teacher_only': only teacher tokens contribute."""
        provenance = ["student", "teacher", "student", "teacher", "student"]
        mask = apply_provenance_mask(provenance, strategy="teacher_only")
        expected = torch.tensor([0, 1, 0, 1, 0], dtype=torch.float32)
        assert torch.equal(mask, expected)

    def test_mask_student_only(self):
        """'student_only': only student tokens contribute."""
        provenance = ["student", "teacher", "student", "teacher", "student"]
        mask = apply_provenance_mask(provenance, strategy="student_only")
        expected = torch.tensor([1, 0, 1, 0, 1], dtype=torch.float32)
        assert torch.equal(mask, expected)

    def test_mask_all_student_provenance(self):
        """All student tokens with teacher_only → empty mask."""
        provenance = ["student"] * 5
        mask = apply_provenance_mask(provenance, strategy="teacher_only")
        assert mask.sum() == 0

    def test_mask_all_teacher_provenance(self):
        """All teacher tokens with student_only → empty mask."""
        provenance = ["teacher"] * 5
        mask = apply_provenance_mask(provenance, strategy="student_only")
        assert mask.sum() == 0


class TestDirectionalLoss:
    """Directional masking: reverse KL on student tokens, forward KL on teacher tokens."""

    def test_directional_applies_correct_kl_per_token(self):
        """For each token, the loss should use the correct KL direction based on provenance."""
        B, L, V = 1, 4, 50
        student_logits = make_logits(B, L, V, seed=0)
        teacher_logits = make_logits(B, L, V, seed=1)
        provenance = ["student", "teacher", "student", "teacher"]
        completion_mask = torch.ones(B, L)

        loss_per_token = compute_masked_loss(
            student_logits, teacher_logits, provenance, completion_mask, strategy="directional"
        )
        assert loss_per_token.shape == (B, L)

        # Manually compute what we expect:
        # student tokens (pos 0, 2): reverse KL = KL(T||S)
        # teacher tokens (pos 1, 3): forward KL = KL(S||T)
        fwd_kl = compute_kl_loss(student_logits, teacher_logits, direction="forward")
        rev_kl = compute_kl_loss(student_logits, teacher_logits, direction="reverse")

        assert torch.allclose(loss_per_token[0, 0], rev_kl[0, 0], atol=1e-5)
        assert torch.allclose(loss_per_token[0, 1], fwd_kl[0, 1], atol=1e-5)
        assert torch.allclose(loss_per_token[0, 2], rev_kl[0, 2], atol=1e-5)
        assert torch.allclose(loss_per_token[0, 3], fwd_kl[0, 3], atol=1e-5)

    def test_directional_with_completion_mask(self):
        """Completion mask should zero out padded positions."""
        B, L, V = 1, 4, 50
        student_logits = make_logits(B, L, V)
        teacher_logits = make_logits(B, L, V, seed=1)
        provenance = ["student", "teacher", "student", "teacher"]
        completion_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.float32)

        loss = compute_masked_loss(
            student_logits, teacher_logits, provenance, completion_mask, strategy="directional"
        )
        # Masked positions should be zero
        assert loss[0, 2] == 0.0
        assert loss[0, 3] == 0.0
        # Unmasked positions should be non-zero (different logits → nonzero KL)
        assert loss[0, 0] > 0.0
        assert loss[0, 1] > 0.0


class TestMaskedLossIntegration:
    """Integration tests: compute_masked_loss with different strategies."""

    def test_all_strategy_equals_forward_kl(self):
        """'all' strategy with forward direction should equal plain forward KL."""
        B, L, V = 2, 5, 100
        student = make_logits(B, L, V)
        teacher = make_logits(B, L, V, seed=1)
        provenance = ["student"] * L
        mask = torch.ones(B, L)

        loss = compute_masked_loss(student, teacher, provenance, mask, strategy="all")
        expected = compute_kl_loss(student, teacher, direction="forward")
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_teacher_only_zeros_student_positions(self):
        B, L, V = 1, 4, 50
        student = make_logits(B, L, V)
        teacher = make_logits(B, L, V, seed=1)
        provenance = ["student", "teacher", "student", "teacher"]
        mask = torch.ones(B, L)

        loss = compute_masked_loss(student, teacher, provenance, mask, strategy="teacher_only")
        assert loss[0, 0] == 0.0  # student token → masked
        assert loss[0, 1] > 0.0   # teacher token → active
        assert loss[0, 2] == 0.0
        assert loss[0, 3] > 0.0
