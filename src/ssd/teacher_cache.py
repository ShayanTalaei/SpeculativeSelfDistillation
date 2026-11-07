"""Sparse teacher-distribution cache captured during SSD generation.

Motivation
----------
SSD evaluates the teacher twice per rollout token. Once during generation —
the routing strategy needs the teacher's next-token distribution to decide
whether the teacher takes over (``ssd.generation`` for the HF loop,
``EnsembleForCausalLM.compute_logits`` for the vLLM ensemble). Then again in
``_compute_loss``, where ``ref_model`` is run over ``teacher_prompt +
completion`` to rebuild the very same distributions for the KL term.

The two are the same quantity at the same positions. Generation step ``t``
produces the distribution used to emit completion token ``t``; the loss keeps
``logits[:, :-1, :]`` and then the last ``logits_to_keep`` positions, so its
position ``t`` also predicts completion token ``t``. The HF loop breaks on EOS
*without* appending, and the vLLM trail is trimmed to ``len(token_ids)``, so
the step count matches the token count exactly on both engines — a clean 1:1
with no off-by-one.

Storing the full ``(B, L, V)`` teacher distribution would be exact but costs
~121 MB per 400-token sequence at Qwen3's V=151936, and the vLLM path
generates the whole *global* batch on every rank. So we store a sparse
snapshot instead.

What is stored
--------------
Per generated token, the union of the teacher's top-K and the student's top-K
vocabulary entries (K configurable, default 100), plus:

* ``logits`` — the teacher's **raw** logits at those indices. Raw, not
  log-probs, because the loss divides logits by ``self.temperature`` before
  ``log_softmax`` and a log-prob captured at one temperature cannot be
  rescaled to another.
* ``lse`` — ``logsumexp`` of the temperature-scaled logits over the **full**
  vocabulary, in fp32.

The ``lse`` is what makes this work. With it, the teacher's log-probability on
every kept index is recovered *exactly*:

    log q(v) = logits[v] / temperature - lse

No renormalisation over the kept subset, no approximation — the kept entries
carry their true probability, and ``1 - sum(exp(log q))`` is the true mass the
teacher put on everything we dropped.

Why the union of both top-Ks
----------------------------
The directional loss needs both KL directions. ``KL(T||S)`` is weighted by the
teacher's probabilities, so the teacher's top-K covers it. ``KL(S||T)`` is
weighted by the *student's*, and the student can be confident on a token the
teacher ranks nowhere — a teacher-only top-K would truncate exactly the terms
that direction is made of. Both distributions are in hand at capture time, so
taking the union costs nothing.

How the truncation is accounted for
-----------------------------------
The dropped tail is not silently discarded. It is folded into one extra
"everything else" bucket, giving a distribution over the partition

    {kept token 1} ... {kept token S}, {tail}

for the student and the teacher alike. The KL between the two coarsened
distributions is, by the log-sum inequality, a **lower bound** on the true
full-vocabulary KL, tightening to equality as K grows. So the cached loss is a
principled underestimate rather than an uncontrolled approximation, and
``kept_mass`` diagnostics report how much of each distribution the bound
actually covers.

Because the partition is just a shorter vocabulary axis, the resulting
``(B, L, S+1)`` log-prob pair drops straight into the existing KL code —
``_compute_kl_per_token`` and the directional ``F.kl_div`` branch both operate
elementwise then sum over the last dim, so they need no modification.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Log-prob written into slots that carry no token (duplicates removed from the
# union, or padding past the end of a sequence). Both sides of the KL get the
# same sentinel, so every formula in _compute_kl_per_token — forward, reverse,
# todi, the JSD mixture — evaluates to exactly 0 there:
# exp(-1e9) is 0 in fp32, and the (target - input) factor is 0 as well.
# -inf would give 0 * inf = NaN instead, which is why this is finite.
EMPTY_SLOT_LOGPROB = -1e9

# Floor for the tail bucket. A student that puts essentially all its mass on
# the kept set leaves 1 - kept_mass at or below 0 in fp32; log() of that is
# -inf or NaN. exp(log(1e-12)) contributes ~1e-11 to the sum, far below the
# noise of the bf16 logits themselves.
_TAIL_EPS = 1e-12


@dataclass
class TeacherTopKCache:
    """Teacher's sparse next-token distributions for one generated sequence.

    All tensors live on CPU — they cross a process boundary (the vLLM
    ensemble's ``collective_rpc``) and a ``broadcast_object_list``, and they
    are only needed again inside the loss, one micro-batch at a time.

    Attributes:
        indices: (T, S) int64. Vocabulary ids of the kept entries. -1 marks an
            empty slot. S = 2*K; slots are empty where the student's top-K
            overlapped the teacher's and the duplicate was dropped.
        logits: (T, S) float32. Teacher's raw (pre-temperature) logits at
            ``indices``. Meaningless where ``indices`` is -1.
        lse: (T,) float32. logsumexp over the full vocabulary of
            ``teacher_logits / temperature``.
        temperature: the temperature ``lse`` was computed at. Must match the
            trainer's ``self.temperature`` when the cache is consumed.
    """

    indices: torch.Tensor
    logits: torch.Tensor
    lse: torch.Tensor
    temperature: float

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    @property
    def num_slots(self) -> int:
        return int(self.indices.shape[1])

    def to_cpu(self) -> "TeacherTopKCache":
        return TeacherTopKCache(
            indices=self.indices.detach().to("cpu", torch.int64),
            logits=self.logits.detach().to("cpu", torch.float32),
            lse=self.lse.detach().to("cpu", torch.float32),
            temperature=self.temperature,
        )


def capture_topk_rows(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    k: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Snapshot the teacher distribution for a batch of generation steps.

    Args:
        student_logits: (N, V) raw student logits for N rows (one per active
            sequence at this decode step).
        teacher_logits: (N, V) raw teacher logits, same rows, same order.
        k: top-K taken from each side; the union has at most 2*K entries.
        temperature: the *loss* temperature, which the trainer also uses for
            sampling. ``lse`` is computed at this temperature.

    Returns:
        (indices (N, 2K) int64, logits (N, 2K) float32, lse (N,) float32).
        Duplicate slots — student picks that the teacher's top-K already
        covers — carry index -1.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student/teacher logits shape mismatch: "
            f"{tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
        )
    if student_logits.ndim != 2:
        raise ValueError(f"expected (N, V) logits, got {tuple(student_logits.shape)}")

    vocab = student_logits.shape[-1]
    k = min(int(k), vocab)

    t_logits_f = teacher_logits.float()

    t_val, t_idx = t_logits_f.topk(k, dim=-1)
    s_idx = student_logits.float().topk(k, dim=-1).indices

    # Drop student picks the teacher's top-K already covers, so the union
    # counts every token once. (N, K, K) bools — at K=100 and the ~16 rows a
    # decode step carries, that is ~160 KB.
    dup = (s_idx.unsqueeze(-1) == t_idx.unsqueeze(-2)).any(dim=-1)

    # Gather before masking: s_idx is still a valid index everywhere, and the
    # masked-out slots are ignored downstream via their -1 index.
    s_val = t_logits_f.gather(-1, s_idx)
    s_idx = torch.where(dup, torch.full_like(s_idx, -1), s_idx)

    indices = torch.cat([t_idx, s_idx], dim=-1)
    logits = torch.cat([t_val, s_val], dim=-1)
    lse = torch.logsumexp(t_logits_f / temperature, dim=-1)

    return indices.to(torch.int64), logits, lse


def pack_steps(
    steps: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    temperature: float,
) -> TeacherTopKCache:
    """Stack per-step rows (each (S,), (S,), ()) into one sequence's cache."""
    if not steps:
        return empty_cache(num_slots=0, temperature=temperature)
    indices = torch.stack([s[0] for s in steps]).to("cpu", torch.int64)
    logits = torch.stack([s[1] for s in steps]).to("cpu", torch.float32)
    lse = torch.stack([s[2].reshape(()) for s in steps]).to("cpu", torch.float32)
    return TeacherTopKCache(indices=indices, logits=logits, lse=lse, temperature=temperature)


def empty_cache(num_slots: int, temperature: float) -> TeacherTopKCache:
    """A zero-length cache — for sequences that generated no tokens."""
    return TeacherTopKCache(
        indices=torch.zeros(0, num_slots, dtype=torch.int64),
        logits=torch.zeros(0, num_slots, dtype=torch.float32),
        lse=torch.zeros(0, dtype=torch.float32),
        temperature=temperature,
    )


def stack_caches(
    caches: list[TeacherTopKCache],
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad a micro-batch of per-sequence caches to a common length.

    Padding matches ``completion_ids``' right-padding, so index t of the
    stacked cache lines up with completion token t.

    Returns:
        (indices (B, L, S), logits (B, L, S), lse (B, L), valid (B, L, S) bool).
        Positions past a sequence's end are entirely invalid, which makes both
        the kept mass 0 and the tail mass 1 there — the KL contribution is 0
        without needing a separate mask.
    """
    if not caches:
        raise ValueError("stack_caches called with no caches")

    slots = max((c.num_slots for c in caches), default=0)
    batch = len(caches)

    indices = torch.full((batch, seq_len, slots), -1, dtype=torch.int64)
    logits = torch.zeros((batch, seq_len, slots), dtype=torch.float32)
    lse = torch.zeros((batch, seq_len), dtype=torch.float32)

    for b, cache in enumerate(caches):
        n = min(len(cache), seq_len)
        if n == 0:
            continue
        indices[b, :n] = cache.indices[:n]
        logits[b, :n] = cache.logits[:n]
        lse[b, :n] = cache.lse[:n]

    indices = indices.to(device, non_blocking=True)
    logits = logits.to(device, non_blocking=True)
    lse = lse.to(device, non_blocking=True)
    valid = indices >= 0

    return indices, logits, lse, valid


def teacher_logprobs_from_cache(
    logits: torch.Tensor,
    lse: torch.Tensor,
    valid: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Exact teacher log-probs on the kept indices: logits/T - lse."""
    logq = logits / temperature - lse.unsqueeze(-1)
    return torch.where(valid, logq, torch.full_like(logq, EMPTY_SLOT_LOGPROB))


def build_partition_logprobs(
    student_all_logps: torch.Tensor,
    indices: torch.Tensor,
    logits: torch.Tensor,
    lse: torch.Tensor,
    valid: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coarsen both distributions onto {kept tokens} ∪ {tail}.

    The result is a pair of log-prob tensors over an (S+1)-wide axis that the
    existing KL code consumes unchanged. KL over this partition lower-bounds
    the true full-vocabulary KL.

    Args:
        student_all_logps: (B, L, V) live student log-probs, already
            temperature-scaled by ``_get_per_token_logps_and_entropies``.
            Carries gradient.
        indices, logits, lse, valid: as returned by ``stack_caches``.
        temperature: must match the temperature ``lse`` was captured at.

    Returns:
        (student_part (B, L, S+1), teacher_part (B, L, S+1),
         student_kept_mass (B, L), teacher_kept_mass (B, L)).
    """
    logq_kept = teacher_logprobs_from_cache(logits, lse, valid, temperature)

    # clamp(min=0) keeps gather in range; the -1 slots are masked right after.
    logp_kept = student_all_logps.gather(-1, indices.clamp(min=0))
    logp_kept = torch.where(
        valid, logp_kept, torch.full_like(logp_kept, EMPTY_SLOT_LOGPROB)
    )

    q_kept_mass = (logq_kept.exp() * valid).sum(-1)
    p_kept_mass = (logp_kept.exp() * valid).sum(-1)

    logq_tail = (1.0 - q_kept_mass).clamp_min(_TAIL_EPS).log()
    logp_tail = (1.0 - p_kept_mass).clamp_min(_TAIL_EPS).log()

    teacher_part = torch.cat([logq_kept, logq_tail.unsqueeze(-1)], dim=-1)
    student_part = torch.cat([logp_kept, logp_tail.unsqueeze(-1)], dim=-1)

    return student_part, teacher_part, p_kept_mass, q_kept_mass


def cache_diagnostics(
    teacher_all_logps: torch.Tensor,
    indices: torch.Tensor,
    logits: torch.Tensor,
    lse: torch.Tensor,
    valid: torch.Tensor,
    temperature: float,
    token_mask: torch.Tensor,
) -> dict[str, float]:
    """Compare cached teacher log-probs against a fresh teacher forward.

    This is the check that the cache is the *same distribution* the loss would
    have recomputed — it isolates alignment and conditioning errors from the
    top-K truncation, which is measured separately via kept mass. A one-token
    misalignment shows up here as a large error; a prompt-conditioning
    difference shows up as a small but systematic one.

    Args:
        teacher_all_logps: (B, L, V) from the live ``ref_model`` forward.
        indices, logits, lse, valid: as returned by ``stack_caches``.
        temperature: the temperature the cache was captured at.
        token_mask: (B, L) float/bool, 1 at real completion positions.

    Returns:
        Scalar summary stats, all computed over valid slots at real tokens.
    """
    cached_logq = teacher_logprobs_from_cache(logits, lse, valid, temperature)
    fresh_logq = teacher_all_logps.gather(-1, indices.clamp(min=0))

    live = valid & token_mask.bool().unsqueeze(-1)
    n_live = live.sum()
    if int(n_live) == 0:
        return {}

    err = (cached_logq - fresh_logq).abs()
    err = torch.where(live, err, torch.zeros_like(err))

    # Probability-weighted error matters more than raw log-prob error: an
    # absolute discrepancy on a token the teacher never emits is irrelevant to
    # the KL, one on a high-probability token is not.
    weight = torch.where(live, fresh_logq.exp(), torch.zeros_like(fresh_logq))
    weighted_err = (err * weight).sum() / weight.sum().clamp_min(1e-12)

    fresh_kept_mass = weight.sum(-1)
    cached_kept_mass = (cached_logq.exp() * live).sum(-1)
    tok = token_mask.bool()

    return {
        "logprob_max_abs_err": float(err.max()),
        "logprob_mean_abs_err": float(err.sum() / n_live),
        "logprob_prob_weighted_err": float(weighted_err),
        "teacher_kept_mass": float(fresh_kept_mass[tok].mean()) if int(tok.sum()) else 0.0,
        "cached_kept_mass": float(cached_kept_mass[tok].mean()) if int(tok.sum()) else 0.0,
    }
