"""Routing strategies for speculative self-distillation (SSD).

A RoutingStrategy decides at each token position whether the student or teacher
contributes the next token. All strategies operate on raw logit vectors (pre-softmax)
from both models at the current decoding step.

The strategies form a hierarchy:
- PureStudent / PureTeacher are degenerate cases (on-policy / off-policy)
- KLSwitch / EntropySwitch / TopKIntersection are token-level switching strategies
- LogitFusion blends distributions (no clean provenance)
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class TokenDecision:
    """Result of a routing strategy decision at one token position."""
    source: str  # 'student' | 'teacher'
    # Diagnostic values recorded for visualization and debugging
    kl_value: float = None
    entropy_value: float = None
    overlap_value: float = None
    reason: str = ""


class RoutingStrategy(ABC):
    """Base class for all routing strategies."""

    @abstractmethod
    def decide(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> TokenDecision:
        """Decide which model contributes the next token.

        Args:
            student_logits: shape (vocab_size,) — raw logits from student
            teacher_logits: shape (vocab_size,) — raw logits from teacher

        Returns:
            TokenDecision with source='student' or source='teacher'
        """
        pass


class PureStudentStrategy(RoutingStrategy):
    """Always select student. Equivalent to on-policy distillation."""

    def decide(self, student_logits, teacher_logits):
        return TokenDecision(source="student", reason="pure_student")


class PureTeacherStrategy(RoutingStrategy):
    """Always select teacher. Equivalent to off-policy distillation."""

    def decide(self, student_logits, teacher_logits):
        return TokenDecision(source="teacher", reason="pure_teacher")


class KLSwitchStrategy(RoutingStrategy):
    """Switch to teacher when KL divergence between student and teacher exceeds threshold.

    direction='reverse': compute KL(S || T). High value means student assigns mass
        where teacher does not — student is overconfident in wrong places.
    direction='forward': compute KL(T || S). High value means teacher assigns mass
        where student does not — student is missing important modes.
    """

    def __init__(self, threshold: float, direction: str = "forward"):
        assert direction in ("forward", "reverse")
        self.threshold = threshold
        self.direction = direction

    def decide(self, student_logits, teacher_logits):
        s_logprobs = F.log_softmax(student_logits.float(), dim=-1)
        t_logprobs = F.log_softmax(teacher_logits.float(), dim=-1)

        if self.direction == "forward":
            # Forward KL: KL(T || S) = sum(T * (log T - log S))
            t_probs = t_logprobs.exp()
            kl = (t_probs * (t_logprobs - s_logprobs)).sum().item()
        else:
            # Reverse KL: KL(S || T) = sum(S * (log S - log T))
            s_probs = s_logprobs.exp()
            kl = (s_probs * (s_logprobs - t_logprobs)).sum().item()

        source = "teacher" if kl > self.threshold else "student"
        return TokenDecision(
            source=source,
            kl_value=kl,
            reason=f"kl_{self.direction}={kl:.4f}, threshold={self.threshold}",
        )


class JSDSwitchStrategy(RoutingStrategy):
    """Switch to teacher when Jensen-Shannon divergence between student and
    teacher exceeds threshold.

    JSD(S, T) = 0.5 * KL(S || M) + 0.5 * KL(T || M), where M = 0.5 * (S + T).
    Symmetric in (S, T); range [0, log 2] ≈ [0, 0.693] (nats).
    """

    def __init__(self, threshold: float):
        self.threshold = float(threshold)

    def decide(self, student_logits, teacher_logits):
        log_p_a = F.log_softmax(student_logits.float(), dim=-1)
        log_p_b = F.log_softmax(teacher_logits.float(), dim=-1)
        # Mixture log-probs, computed in log space for numerical stability.
        log_m = torch.logsumexp(
            torch.stack([log_p_a, log_p_b], dim=0), dim=0
        ) - math.log(2.0)
        p_a = log_p_a.exp()
        p_b = log_p_b.exp()
        kl_a_m = (p_a * (log_p_a - log_m)).sum().item()
        kl_b_m = (p_b * (log_p_b - log_m)).sum().item()
        jsd = 0.5 * (kl_a_m + kl_b_m)

        source = "teacher" if jsd > self.threshold else "student"
        return TokenDecision(
            source=source,
            kl_value=jsd,
            reason=f"jsd={jsd:.4f}, threshold={self.threshold}",
        )


class EntropySwitchStrategy(RoutingStrategy):
    """Switch to teacher when student entropy exceeds threshold.

    High student entropy = student is uncertain about the next token.
    The teacher (with privileged information) is likely more confident.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold

    def decide(self, student_logits, teacher_logits):
        s_logprobs = F.log_softmax(student_logits.float(), dim=-1)
        s_probs = s_logprobs.exp()
        entropy = -(s_probs * s_logprobs).sum().item()

        source = "teacher" if entropy > self.threshold else "student"
        return TokenDecision(
            source=source,
            entropy_value=entropy,
            reason=f"student_entropy={entropy:.4f}, threshold={self.threshold}",
        )


class TopKIntersectionStrategy(RoutingStrategy):
    """Switch to teacher when the overlap between student and teacher top-K tokens is low.

    Computes |topK(S) ∩ topK(T)| / K. If this ratio is below min_overlap,
    the models disagree on what tokens are most likely, so the teacher generates.
    """

    def __init__(self, k: int, min_overlap: float):
        self.k = k
        self.min_overlap = min_overlap

    def decide(self, student_logits, teacher_logits):
        s_topk = set(torch.topk(student_logits, self.k).indices.tolist())
        t_topk = set(torch.topk(teacher_logits, self.k).indices.tolist())
        overlap = len(s_topk & t_topk) / self.k

        source = "teacher" if overlap < self.min_overlap else "student"
        return TokenDecision(
            source=source,
            overlap_value=overlap,
            reason=f"topk_overlap={overlap:.2f}, k={self.k}, min={self.min_overlap}",
        )


class CoinFlipStrategy(RoutingStrategy):
    """Fair coin flip: randomly pick student or teacher at each token with p=0.5.

    No threshold tuning needed. Teacher tokens inject correct information
    at random positions; student learns to continue from teacher-injected facts.
    """

    def __init__(self, p_teacher: float = 0.5):
        self.p_teacher = p_teacher

    def decide(self, student_logits, teacher_logits):
        if torch.rand(1).item() < self.p_teacher:
            return TokenDecision(source="teacher", reason=f"coin_flip(p={self.p_teacher})")
        return TokenDecision(source="student", reason=f"coin_flip(p={self.p_teacher})")


class LogitFusionStrategy(RoutingStrategy):
    """Fuse student and teacher logits via linear interpolation.

    Unlike switching strategies, this blends distributions rather than selecting one.
    The 'source' is always 'fused' — there is no clean provenance.
    Used for training, NOT for the visualizer.
    """

    def __init__(self, alpha: float):
        self.alpha = alpha

    def decide(self, student_logits, teacher_logits):
        # For LogitFusion, the decision is always 'fused' — the actual fusion
        # happens in the generation loop which reads the alpha from this strategy.
        return TokenDecision(
            source="fused",
            reason=f"logit_fusion_alpha={self.alpha}",
        )

    def fuse(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Compute fused logits: alpha * teacher + (1-alpha) * student."""
        return self.alpha * teacher_logits + (1 - self.alpha) * student_logits
