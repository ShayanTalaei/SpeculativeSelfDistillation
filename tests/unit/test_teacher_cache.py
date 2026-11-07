"""Unit tests for the sparse teacher-distribution cache.

Covers the three properties the loss depends on:

1. Kept entries carry the teacher's *exact* log-probability, recovered from
   the stored full-vocabulary logsumexp rather than renormalised over the
   subset.
2. KL over the {kept} ∪ {tail} partition lower-bounds the true KL, and closes
   on it as K grows.
3. Empty slots — duplicates dropped from the union, and padding past the end
   of a sequence — contribute exactly zero, with no NaN.
"""

import pytest
import torch
import torch.nn.functional as F

from ssd.teacher_cache import (
    EMPTY_SLOT_LOGPROB,
    build_partition_logprobs,
    cache_diagnostics,
    capture_topk_rows,
    empty_cache,
    pack_steps,
    stack_caches,
    teacher_logprobs_from_cache,
)


CPU = torch.device("cpu")


def _lm_like_logits(rows, vocab, zipf_exponent=2.2, seed=0):
    """Logits shaped like a real next-token distribution (Zipf-ish tail)."""
    g = torch.Generator().manual_seed(seed)
    rank = torch.arange(1, vocab + 1, dtype=torch.float32)
    base = -zipf_exponent * torch.log(rank)
    out = torch.empty(rows, vocab)
    for r in range(rows):
        out[r] = base[torch.randperm(vocab, generator=g)]
    return out + torch.randn(rows, vocab, generator=g) * 0.3


def _pack_rows(idx, val, lse, temperature=1.0):
    return pack_steps(
        [(idx[i], val[i], lse[i]) for i in range(idx.shape[0])], temperature
    )


class TestExactness:
    """Kept entries must carry true probabilities, not subset-renormalised ones."""

    @pytest.mark.parametrize("temperature", [1.0, 0.7, 2.0])
    def test_kept_logprobs_are_exact(self, temperature):
        vocab, rows, k = 2000, 6, 50
        s = _lm_like_logits(rows, vocab, seed=1)
        t = _lm_like_logits(rows, vocab, seed=2)

        idx, val, lse = capture_topk_rows(s, t, k, temperature)
        valid = idx >= 0
        cached = teacher_logprobs_from_cache(val, lse, valid, temperature)

        exact = F.log_softmax(t / temperature, dim=-1).gather(-1, idx.clamp(min=0))
        err = ((cached - exact).abs() * valid).max()
        assert err < 1e-5, f"cached teacher log-probs drifted by {err}"

    def test_kept_mass_is_true_mass_not_one(self):
        """A renormalised cache would report mass 1.0 and silently overstate."""
        vocab, k = 5000, 20
        t = _lm_like_logits(1, vocab, zipf_exponent=1.2, seed=3)
        idx, val, lse = capture_topk_rows(t, t, k, 1.0)
        valid = idx >= 0
        mass = (teacher_logprobs_from_cache(val, lse, valid, 1.0).exp() * valid).sum()
        assert mass < 0.99, "top-k of a flat distribution should not cover ~all mass"

        true_mass = F.softmax(t, dim=-1).gather(-1, idx.clamp(min=0))[valid].sum()
        assert torch.allclose(mass, true_mass, atol=1e-5)


class TestUnion:
    """The union must cover both KL directions and count each token once."""

    def test_student_top_tokens_are_covered(self):
        vocab, k = 3000, 40
        # Deliberately opposed: the student's peak is the teacher's trough.
        t = _lm_like_logits(4, vocab, seed=4)
        s = -t

        idx, _, _ = capture_topk_rows(s, t, k, 1.0)
        s_top = s.topk(k, dim=-1).indices
        for row in range(s.shape[0]):
            kept = set(idx[row][idx[row] >= 0].tolist())
            missing = set(s_top[row].tolist()) - kept
            assert not missing, f"student top-{k} not covered: {sorted(missing)[:5]}"

    def test_no_duplicate_indices(self):
        vocab, k = 500, 100  # small vocab forces heavy overlap
        t = _lm_like_logits(5, vocab, seed=5)
        s = t + torch.randn(5, vocab, generator=torch.Generator().manual_seed(6)) * 0.05

        idx, _, _ = capture_topk_rows(s, t, k, 1.0)
        for row in range(idx.shape[0]):
            kept = idx[row][idx[row] >= 0].tolist()
            assert len(kept) == len(set(kept)), "union double-counted a token"

    def test_identical_distributions_dedupe_completely(self):
        vocab, k = 800, 30
        t = _lm_like_logits(3, vocab, seed=7)
        idx, _, _ = capture_topk_rows(t, t, k, 1.0)
        # Every student pick duplicates a teacher pick, so exactly k survive.
        assert ((idx >= 0).sum(-1) == k).all()


class TestPartitionKL:
    """KL over the coarsened partition lower-bounds the true KL."""

    @pytest.mark.parametrize("k", [10, 50, 200])
    def test_lower_bounds_true_kl(self, k):
        vocab, rows = 4000, 8
        s = _lm_like_logits(rows, vocab, seed=8)
        t = _lm_like_logits(rows, vocab, seed=9)

        s_all = F.log_softmax(s, -1).unsqueeze(0)
        t_all = F.log_softmax(t, -1).unsqueeze(0)

        idx, val, lse = capture_topk_rows(s, t, k, 1.0)
        stacked = stack_caches([_pack_rows(idx, val, lse)], rows, CPU)
        sp, tp, _, _ = build_partition_logprobs(s_all, *stacked, 1.0)

        for approx, exact in (
            (F.kl_div(sp, tp, reduction="none", log_target=True).sum(-1),
             F.kl_div(s_all, t_all, reduction="none", log_target=True).sum(-1)),
            (F.kl_div(tp, sp, reduction="none", log_target=True).sum(-1),
             F.kl_div(t_all, s_all, reduction="none", log_target=True).sum(-1)),
        ):
            assert (approx <= exact + 1e-4).all(), "partition KL exceeded true KL"
            assert (approx >= 0).all(), "KL went negative"

    def test_tightens_as_k_grows(self):
        vocab, rows = 3000, 6
        s = _lm_like_logits(rows, vocab, seed=10)
        t = _lm_like_logits(rows, vocab, seed=11)
        s_all = F.log_softmax(s, -1).unsqueeze(0)
        t_all = F.log_softmax(t, -1).unsqueeze(0)
        exact = F.kl_div(s_all, t_all, reduction="none", log_target=True).sum(-1).mean()

        errs = []
        for k in (5, 25, 125):
            idx, val, lse = capture_topk_rows(s, t, k, 1.0)
            stacked = stack_caches([_pack_rows(idx, val, lse)], rows, CPU)
            sp, tp, _, _ = build_partition_logprobs(s_all, *stacked, 1.0)
            approx = F.kl_div(sp, tp, reduction="none", log_target=True).sum(-1).mean()
            errs.append(float(exact - approx))

        assert errs == sorted(errs, reverse=True), f"error did not shrink with k: {errs}"

    def test_k_covering_whole_vocab_is_exact(self):
        vocab, rows = 64, 4
        s = _lm_like_logits(rows, vocab, seed=12)
        t = _lm_like_logits(rows, vocab, seed=13)
        s_all = F.log_softmax(s, -1).unsqueeze(0)
        t_all = F.log_softmax(t, -1).unsqueeze(0)

        idx, val, lse = capture_topk_rows(s, t, vocab, 1.0)
        stacked = stack_caches([_pack_rows(idx, val, lse)], rows, CPU)
        sp, tp, _, _ = build_partition_logprobs(s_all, *stacked, 1.0)

        approx = F.kl_div(sp, tp, reduction="none", log_target=True).sum(-1)
        exact = F.kl_div(s_all, t_all, reduction="none", log_target=True).sum(-1)
        assert torch.allclose(approx, exact, atol=1e-5), (approx, exact)

    def test_gradient_flows_to_student(self):
        vocab, rows, k = 1000, 3, 20
        s = _lm_like_logits(rows, vocab, seed=14).requires_grad_(True)
        t = _lm_like_logits(rows, vocab, seed=15)

        idx, val, lse = capture_topk_rows(s.detach(), t, k, 1.0)
        stacked = stack_caches([_pack_rows(idx, val, lse)], rows, CPU)
        sp, tp, _, _ = build_partition_logprobs(F.log_softmax(s, -1).unsqueeze(0), *stacked, 1.0)
        F.kl_div(sp, tp, reduction="none", log_target=True).sum().backward()

        assert s.grad is not None and torch.isfinite(s.grad).all()
        assert s.grad.abs().sum() > 0


class TestEmptySlots:
    """Padding and dropped duplicates must contribute exactly nothing."""

    def test_padded_positions_are_zero_and_finite(self):
        vocab, k, seq = 1000, 20, 5
        s_all = F.log_softmax(_lm_like_logits(seq, vocab, seed=16), -1).unsqueeze(0)

        # A 2-token cache padded out to a 5-token completion.
        t = _lm_like_logits(2, vocab, seed=17)
        idx, val, lse = capture_topk_rows(t, t, k, 1.0)
        stacked = stack_caches([_pack_rows(idx, val, lse)], seq, CPU)
        sp, tp, p_mass, q_mass = build_partition_logprobs(s_all, *stacked, 1.0)

        kl = F.kl_div(sp, tp, reduction="none", log_target=True).sum(-1)
        assert torch.isfinite(kl).all(), "padding produced NaN/inf"
        assert torch.allclose(kl[0, 2:], torch.zeros(3), atol=1e-6), kl[0, 2:]
        assert (q_mass[0, 2:] == 0).all()

    def test_empty_cache_is_all_padding(self):
        vocab, seq = 500, 4
        s_all = F.log_softmax(_lm_like_logits(seq, vocab, seed=18), -1).unsqueeze(0)
        stacked = stack_caches([empty_cache(40, 1.0)], seq, CPU)
        sp, tp, _, _ = build_partition_logprobs(s_all, *stacked, 1.0)
        kl = F.kl_div(sp, tp, reduction="none", log_target=True).sum(-1)
        assert torch.isfinite(kl).all() and torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)

    def test_sentinel_is_finite(self):
        """-inf would give 0 * inf = NaN in the KL; the sentinel must not be."""
        assert torch.isfinite(torch.tensor(EMPTY_SLOT_LOGPROB))
        assert float(torch.tensor(EMPTY_SLOT_LOGPROB).exp()) == 0.0


class TestStacking:
    """Ragged sequences must line up with right-padded completion_ids."""

    def test_each_sequence_keeps_its_own_rows(self):
        vocab, k = 600, 10
        caches = []
        expected = []
        for i, n in enumerate((3, 1, 5)):
            t = _lm_like_logits(n, vocab, seed=20 + i)
            idx, val, lse = capture_topk_rows(t, t, k, 1.0)
            caches.append(_pack_rows(idx, val, lse))
            expected.append(idx)

        indices, _, _, valid = stack_caches(caches, seq_len=5, device=CPU)
        assert indices.shape == (3, 5, 2 * k)
        for b, (idx, n) in enumerate(zip(expected, (3, 1, 5))):
            assert torch.equal(indices[b, :n], idx), f"sequence {b} misaligned"
            assert not valid[b, n:].any(), f"sequence {b} padding not marked invalid"

    def test_cache_longer_than_seq_len_is_truncated(self):
        vocab, k = 400, 8
        t = _lm_like_logits(6, vocab, seed=30)
        idx, val, lse = capture_topk_rows(t, t, k, 1.0)
        indices, _, _, _ = stack_caches([_pack_rows(idx, val, lse)], seq_len=4, device=CPU)
        assert indices.shape[1] == 4
        assert torch.equal(indices[0], idx[:4])


class TestDropsIntoTheRealLoss:
    """The partition is just a narrower vocabulary axis.

    Every rule in DistilTrainer._compute_kl_per_token is elementwise then
    summed over the last dim, so with K covering the whole vocabulary the
    partition form must reproduce the full-vocabulary result exactly — for
    each kl_strategy/alpha the trainer supports, not just forward KL.
    """

    @staticmethod
    def _partition_pair(vocab, rows, seed):
        s = _lm_like_logits(rows, vocab, seed=seed)
        t = _lm_like_logits(rows, vocab, seed=seed + 100)
        s_all = F.log_softmax(s, -1).unsqueeze(0)
        t_all = F.log_softmax(t, -1).unsqueeze(0)
        idx, val, lse = capture_topk_rows(s, t, vocab, 1.0)  # K = whole vocab
        stacked = stack_caches([_pack_rows(idx, val, lse)], rows, CPU)
        sp, tp, _, _ = build_partition_logprobs(s_all, *stacked, 1.0)
        return s_all, t_all, sp, tp

    @pytest.mark.parametrize(
        "kl_strategy,alpha",
        [("standard", 0.0), ("standard", 1.0), ("standard", 0.5), ("todi", 0.0)],
    )
    def test_matches_full_vocab(self, kl_strategy, alpha):
        from distil_trainer import DistilTrainer

        class _Args:
            pass

        args = _Args()
        args.kl_strategy = kl_strategy
        args.alpha = alpha

        s_all, t_all, sp, tp = self._partition_pair(vocab=48, rows=5, seed=50)

        full = DistilTrainer._compute_kl_per_token(args, s_all, t_all)
        part = DistilTrainer._compute_kl_per_token(args, sp, tp)

        assert torch.allclose(full, part, atol=1e-4), (
            f"{kl_strategy}/alpha={alpha}: {full} vs {part}"
        )

    def test_directional_branch_matches_full_vocab(self):
        """The directional loss mixes both KL directions per token."""
        s_all, t_all, sp, tp = self._partition_pair(vocab=48, rows=6, seed=60)
        is_student = torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0, 1.0]])
        is_teacher = 1.0 - is_student

        def directional(s, t):
            fwd = F.kl_div(s, t, reduction="none", log_target=True).sum(-1)
            rev = F.kl_div(t, s, reduction="none", log_target=True).sum(-1)
            return rev * is_student + fwd * is_teacher

        assert torch.allclose(
            directional(s_all, t_all), directional(sp, tp), atol=1e-4
        )


class TestDiagnostics:
    """The verify-mode comparison must catch a misalignment."""

    def test_aligned_cache_reports_near_zero_error(self):
        vocab, rows, k = 2000, 5, 30
        t = _lm_like_logits(rows, vocab, seed=40)
        idx, val, lse = capture_topk_rows(t, t, k, 1.0)
        stacked = stack_caches([_pack_rows(idx, val, lse)], rows, CPU)
        t_all = F.log_softmax(t, -1).unsqueeze(0)

        stats = cache_diagnostics(
            t_all, *stacked, temperature=1.0, token_mask=torch.ones(1, rows)
        )
        assert stats["logprob_max_abs_err"] < 1e-4, stats
        assert stats["teacher_kept_mass"] > 0.5

    def test_off_by_one_is_caught(self):
        """Shifting the cache by one token must show up as a large error."""
        vocab, rows, k = 2000, 6, 30
        t = _lm_like_logits(rows, vocab, seed=41)
        idx, val, lse = capture_topk_rows(t, t, k, 1.0)
        stacked = stack_caches([_pack_rows(idx, val, lse)], rows, CPU)

        shifted = F.log_softmax(t.roll(1, dims=0), -1).unsqueeze(0)
        stats = cache_diagnostics(
            shifted, *stacked, temperature=1.0, token_mask=torch.ones(1, rows)
        )
        assert stats["logprob_max_abs_err"] > 1.0, (
            f"a one-token shift went unnoticed: {stats}"
        )
