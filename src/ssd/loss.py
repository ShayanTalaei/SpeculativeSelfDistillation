"""Loss computation for speculative self-distillation (SSD).

Supports:
- Forward KL: KL(teacher || student) — student tries to cover teacher's distribution
- Reverse KL: KL(student || teacher) — student focuses on teacher's peaks
- Token provenance-based masking: loss only on specific token types
- Directional loss: different KL directions for student vs teacher tokens
"""

import torch
import torch.nn.functional as F


def compute_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    direction: str = "forward",
) -> torch.Tensor:
    """Compute per-position KL divergence, summed over the vocab dimension.

    Args:
        student_logits: (batch, seq_len, vocab_size)
        teacher_logits: (batch, seq_len, vocab_size)
        direction: 'forward' for KL(T||S), 'reverse' for KL(S||T)

    Returns:
        (batch, seq_len) tensor of per-position KL values
    """
    s_logprobs = F.log_softmax(student_logits.float(), dim=-1)
    t_logprobs = F.log_softmax(teacher_logits.float(), dim=-1)

    if direction == "forward":
        # Forward KL: KL(T || S) = sum_v T(v) * (log T(v) - log S(v))
        t_probs = t_logprobs.exp()
        kl = (t_probs * (t_logprobs - s_logprobs)).sum(dim=-1)
    elif direction == "reverse":
        # Reverse KL: KL(S || T) = sum_v S(v) * (log S(v) - log T(v))
        s_probs = s_logprobs.exp()
        kl = (s_probs * (s_logprobs - t_logprobs)).sum(dim=-1)
    else:
        raise ValueError(f"direction must be 'forward' or 'reverse', got '{direction}'")

    return kl


def apply_provenance_mask(
    provenance: list[str],
    strategy: str,
) -> torch.Tensor:
    """Create a binary mask based on token provenance and masking strategy.

    Args:
        provenance: list of 'student' or 'teacher' per token position
        strategy: 'all' | 'teacher_only' | 'student_only'

    Returns:
        (seq_len,) float tensor, 1.0 for active positions, 0.0 for masked
    """
    seq_len = len(provenance)
    if strategy == "all":
        return torch.ones(seq_len, dtype=torch.float32)
    elif strategy == "teacher_only":
        return torch.tensor(
            [1.0 if p == "teacher" else 0.0 for p in provenance],
            dtype=torch.float32,
        )
    elif strategy == "student_only":
        return torch.tensor(
            [1.0 if p == "student" else 0.0 for p in provenance],
            dtype=torch.float32,
        )
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'all', 'teacher_only', or 'student_only'.")


def compute_masked_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    provenance: list[str],
    completion_mask: torch.Tensor,
    strategy: str = "all",
) -> torch.Tensor:
    """Compute per-token KL loss with provenance-based masking.

    For 'directional' strategy:
        - Student tokens: reverse KL (pushes student toward teacher's mode)
        - Teacher tokens: forward KL (ensures student covers teacher's distribution)

    For other strategies, forward KL is used everywhere, then masked by provenance.

    Args:
        student_logits: (batch, seq_len, vocab_size)
        teacher_logits: (batch, seq_len, vocab_size)
        provenance: list of 'student'/'teacher' per position (length = seq_len)
        completion_mask: (batch, seq_len) binary mask for valid tokens
        strategy: 'all' | 'teacher_only' | 'student_only' | 'directional'

    Returns:
        (batch, seq_len) per-token loss values (masked positions are 0)
    """
    B, L, V = student_logits.shape

    if strategy == "directional":
        fwd_kl = compute_kl_loss(student_logits, teacher_logits, direction="forward")
        rev_kl = compute_kl_loss(student_logits, teacher_logits, direction="reverse")

        # Build per-position selector: student tokens → reverse KL, teacher tokens → forward KL
        loss = torch.zeros(B, L, device=student_logits.device, dtype=student_logits.dtype)
        for i, p in enumerate(provenance):
            if p == "student":
                loss[:, i] = rev_kl[:, i]
            else:
                loss[:, i] = fwd_kl[:, i]

        return loss * completion_mask

    # Non-directional strategies: compute forward KL, then mask
    kl = compute_kl_loss(student_logits, teacher_logits, direction="forward")
    prov_mask = apply_provenance_mask(provenance, strategy).to(kl.device)

    # Broadcast provenance mask across batch dimension
    return kl * prov_mask.unsqueeze(0) * completion_mask
