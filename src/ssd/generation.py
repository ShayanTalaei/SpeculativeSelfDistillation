"""SSD generation loop.

Implements token-by-token generation where a RoutingStrategy decides at each step
whether the student or teacher contributes the next token. Both models are conditioned
on the exact same prefix (shared sequence of previously generated tokens).

This uses HuggingFace model.forward() directly, NOT vLLM, because SSD
generation requires per-token interleaving of student and teacher forward passes.

Uses KV cache: the prompt is encoded once (prefill), then each subsequent step
forwards only the last token with past_key_values for O(1) per-step compute.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ssd.strategies import RoutingStrategy, LogitFusionStrategy
from ssd.profiling import maybe_phase, get_profiler
from ssd.teacher_cache import TeacherTopKCache, capture_topk_rows, pack_steps


@dataclass
class GenerationOutput:
    """Complete output of a SSD generation run."""
    tokens: list[str]                # generated token strings
    token_ids: list[int]             # generated token ids
    provenance: list[str]            # 'student' or 'teacher' per token
    student_logprobs: list[float]    # student's logprob of the chosen token at each step
    teacher_logprobs: list[float]    # teacher's logprob of the chosen token at each step
    kl_forward: list[float]          # KL(S||T) at each step
    kl_reverse: list[float]          # KL(T||S) at each step
    student_entropy: list[float]     # student entropy at each step
    teacher_entropy: list[float]     # teacher entropy at each step
    decisions: list[dict]            # raw decision info per step
    shared_prefix_verified: bool = True  # always True in this implementation
    # Sparse snapshot of the teacher distribution at each generated token, so
    # the loss can skip re-running the teacher. None unless capture was
    # requested. See ssd.teacher_cache.
    teacher_cache: TeacherTopKCache | None = None
    full_teacher_cache: object | None = None


def _forward_last_logits(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    past_key_values=None,
    use_cache: bool = True,
) -> tuple[torch.Tensor, object]:
    """Run forward pass and return logits at the last position plus KV cache.

    Args:
        model: HuggingFace causal LM
        input_ids: (1, seq_len) token ids — full prompt for prefill, or (1, 1) for decode
        past_key_values: cached key/values from previous forward pass
        use_cache: whether to return updated KV cache

    Returns:
        (logits, past_key_values) — logits shape (vocab_size,), cache for next step
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
    return outputs.logits[0, -1, :], outputs.past_key_values


def _compute_per_step_stats(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    chosen_id: int,
) -> dict:
    """Compute KL divergences, entropies, and logprobs for one generation step."""
    s_logprobs = F.log_softmax(s_logits.float(), dim=-1)
    t_logprobs = F.log_softmax(t_logits.float(), dim=-1)
    s_probs = s_logprobs.exp()
    t_probs = t_logprobs.exp()

    s_entropy = -(s_probs * s_logprobs).sum().item()
    t_entropy = -(t_probs * t_logprobs).sum().item()
    kl_fwd = (t_probs * (t_logprobs - s_logprobs)).sum().item()  # KL(T||S)
    kl_rev = (s_probs * (s_logprobs - t_logprobs)).sum().item()  # KL(S||T)

    return {
        "student_logprob": s_logprobs[chosen_id].item(),
        "teacher_logprob": t_logprobs[chosen_id].item(),
        "kl_forward": kl_fwd,
        "kl_reverse": kl_rev,
        "student_entropy": s_entropy,
        "teacher_entropy": t_entropy,
    }


def ssd_generate(
    student_model: PreTrainedModel,
    teacher_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    student_prompt: str,
    teacher_prompt: str,
    strategy: RoutingStrategy,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int = 0,
    eos_token_id: int = None,
) -> GenerationOutput:
    """Generate a sequence using SSD token-by-token decoding.

    At each step:
    1. Both models forward pass on their respective prefix (prompt + shared generated tokens)
    2. Strategy decides which model contributes
    3. Token is sampled from the chosen distribution
    4. Token is appended to the shared generated sequence

    Both models see the exact same generated tokens as their continuation,
    but may have different prompt prefixes (student sees question only,
    teacher sees documents + question).

    Args:
        student_model: the trainable student model
        teacher_model: the frozen teacher model (possibly same arch, different prompt)
        tokenizer: shared tokenizer
        student_prompt: prompt for student (e.g., question only)
        teacher_prompt: prompt for teacher (e.g., documents + question)
        strategy: RoutingStrategy instance
        max_new_tokens: maximum tokens to generate
        temperature: sampling temperature (0.0 = greedy)
        eos_token_id: stop generation at this token (defaults to tokenizer.eos_token_id)
    """
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    # Tokenize prompts
    s_prompt_ids = tokenizer.encode(student_prompt, return_tensors="pt").to(student_model.device)
    t_prompt_ids = tokenizer.encode(teacher_prompt, return_tensors="pt").to(teacher_model.device)

    # Generated token ids (shared between both models)
    generated_ids = []

    output = GenerationOutput(
        tokens=[], token_ids=[], provenance=[],
        student_logprobs=[], teacher_logprobs=[],
        kl_forward=[], kl_reverse=[],
        student_entropy=[], teacher_entropy=[],
        decisions=[],
    )

    is_fusion = isinstance(strategy, LogitFusionStrategy)

    student_cache = None
    teacher_cache = None

    for step in range(max_new_tokens):
        if step == 0:
            # Prefill: forward full prompt, capture KV cache
            s_input = s_prompt_ids
            t_input = t_prompt_ids
        else:
            # Decode: forward only the last chosen token with cached KVs
            last_token = torch.tensor([[generated_ids[-1]]], dtype=torch.long)
            s_input = last_token.to(student_model.device)
            t_input = last_token.to(teacher_model.device)

        # Forward pass both models — move teacher logits to student device for comparison
        s_logits, student_cache = _forward_last_logits(
            student_model, s_input, past_key_values=student_cache,
        )
        t_logits_raw, teacher_cache = _forward_last_logits(
            teacher_model, t_input, past_key_values=teacher_cache,
        )
        t_logits = t_logits_raw.to(s_logits.device)

        # Strategy decides
        decision = strategy.decide(s_logits, t_logits)

        # Sample token from the chosen distribution
        if is_fusion:
            fused = strategy.fuse(s_logits, t_logits)
            chosen_id = _sample_token(fused, temperature, top_k)
            decision.source = "fused"
        elif decision.source == "student":
            chosen_id = _sample_token(s_logits, temperature, top_k)
        else:
            # Teacher logits may be on a different device — sample on teacher's device
            chosen_id = _sample_token(t_logits, temperature, top_k)

        # Check for EOS
        if chosen_id == eos_token_id:
            break

        # Record
        token_str = tokenizer.decode([chosen_id])
        generated_ids.append(chosen_id)

        stats = _compute_per_step_stats(s_logits, t_logits, chosen_id)

        output.tokens.append(token_str)
        output.token_ids.append(chosen_id)
        output.provenance.append(decision.source)
        output.student_logprobs.append(stats["student_logprob"])
        output.teacher_logprobs.append(stats["teacher_logprob"])
        output.kl_forward.append(stats["kl_forward"])
        output.kl_reverse.append(stats["kl_reverse"])
        output.student_entropy.append(stats["student_entropy"])
        output.teacher_entropy.append(stats["teacher_entropy"])
        output.decisions.append({
            "source": decision.source,
            "kl_value": decision.kl_value,
            "entropy_value": decision.entropy_value,
            "overlap_value": decision.overlap_value,
            "reason": decision.reason,
        })

    return output


def _forward_batch_last_logits(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values=None,
    use_cache: bool = True,
) -> tuple[torch.Tensor, object]:
    """Batched forward pass returning last-position logits for each sample.

    Args:
        model: HuggingFace causal LM
        input_ids: (B, seq_len) for prefill, or (B, 1) for decode
        attention_mask: (B, total_seq_len) including cached positions
        past_key_values: cached KVs from previous step
        use_cache: whether to return updated KV cache

    Returns:
        (logits, past_key_values) — logits shape (B, vocab_size)
    """
    # Compute position_ids from attention_mask so left-padding gets correct positions.
    # For prefill: positions = cumsum(mask) - 1, clamped to 0 for pad tokens.
    # For decode with cache: position = total attended tokens so far (per sample).
    if past_key_values is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
    else:
        # Decode step: position = number of 1s in attention_mask - 1
        position_ids = attention_mask.long().sum(-1, keepdim=True) - 1

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            logits_to_keep=1,
        )
    return outputs.logits[:, 0, :], outputs.past_key_values


def ssd_generate_batch(
    student_model: PreTrainedModel,
    teacher_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase, # assumes both student and teacher use the same tokenizer
    student_prompts: list[str],
    teacher_prompts: list[str],
    strategy: RoutingStrategy,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int = 0,
    eos_token_id: int = None,
    capture_teacher_topk: int = 0,
    cache_temperature: float | None = None,
) -> list[GenerationOutput]:
    """Batched SSD generation — one forward pass per model per step.

    Left-pads prompts so they can be batched. Strategy decisions and token
    sampling are still per-sample (cheap CPU ops), but GPU forward passes
    are batched across the batch dimension.

    Args:
        student_model: the trainable student model
        teacher_model: the frozen teacher model
        tokenizer: shared tokenizer (must have pad_token_id set)
        student_prompts: list of B student prompts
        teacher_prompts: list of B teacher prompts
        strategy: RoutingStrategy instance
        max_new_tokens: maximum tokens to generate per sample
        temperature: sampling temperature (0.0 = greedy)
        eos_token_id: stop token (defaults to tokenizer.eos_token_id)
        capture_teacher_topk: if > 0, snapshot the teacher's distribution at
            each generated token into ``GenerationOutput.teacher_cache`` so the
            loss can skip its own teacher forward. This is the K in top-K.
        cache_temperature: temperature the cache's logsumexp is computed at.
            Must be the temperature the *loss* scales logits by, which is not
            necessarily the sampling temperature (greedy sampling uses 0.0).
            Defaults to ``temperature``.

    Returns:
        list of B GenerationOutput, one per sample
    """
    B = len(student_prompts)
    assert len(teacher_prompts) == B

    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    is_fusion = isinstance(strategy, LogitFusionStrategy)

    capture = int(capture_teacher_topk) > 0
    if cache_temperature is None:
        cache_temperature = temperature
    if capture and cache_temperature <= 0:
        raise ValueError(
            f"cache_temperature must be > 0 to capture a teacher cache, got "
            f"{cache_temperature}. Pass the loss temperature explicitly when "
            f"sampling greedily (temperature=0)."
        )
    # Per-sample list of (indices, logits, lse) rows, one entry per emitted token.
    cache_steps: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = [
        [] for _ in range(B)
    ]

    # Tokenize and left-pad each set of prompts
    def _left_pad_encode(prompts: list[str], device: torch.device):
        encoded = [tokenizer.encode(p) for p in prompts]
        max_len = max(len(e) for e in encoded)
        padded = []
        mask = []
        for e in encoded:
            pad_len = max_len - len(e)
            padded.append([pad_id] * pad_len + e)
            mask.append([0] * pad_len + [1] * len(e))
        return (
            torch.tensor(padded, dtype=torch.long, device=device),
            torch.tensor(mask, dtype=torch.long, device=device),
        )

    s_ids, s_mask = _left_pad_encode(student_prompts, student_model.device)
    t_ids, t_mask = _left_pad_encode(teacher_prompts, teacher_model.device)

    # Per-sample state
    outputs = [
        GenerationOutput(
            tokens=[], token_ids=[], provenance=[],
            student_logprobs=[], teacher_logprobs=[],
            kl_forward=[], kl_reverse=[],
            student_entropy=[], teacher_entropy=[],
            decisions=[],
        )
        for _ in range(B)
    ]
    done = [False] * B

    student_cache = None
    teacher_cache = None

    for step in range(max_new_tokens):
        if all(done):
            break

        if step == 0:
            # Prefill
            s_input, t_input = s_ids, t_ids
        else:
            # Decode: (B, 1) of each sample's last chosen token
            s_input = torch.tensor(
                [[outputs[i].token_ids[-1] if not done[i] else pad_id] for i in range(B)],
                dtype=torch.long, device=student_model.device,
            )
            t_input = torch.tensor(
                [[outputs[i].token_ids[-1] if not done[i] else pad_id] for i in range(B)],
                dtype=torch.long, device=teacher_model.device,
            )
            # Extend attention masks by 1 for the new token
            s_mask = torch.cat([s_mask, torch.ones(B, 1, dtype=torch.long, device=s_mask.device)], dim=1)
            t_mask = torch.cat([t_mask, torch.ones(B, 1, dtype=torch.long, device=t_mask.device)], dim=1)

        # Batched forward
        prefill_or_decode = "prefill" if step == 0 else "decode"
        with maybe_phase(f"generation/{prefill_or_decode}_student_forward"):
            s_logits_batch, student_cache = _forward_batch_last_logits(
                student_model, s_input, s_mask, past_key_values=student_cache,
            )
        with maybe_phase(f"generation/{prefill_or_decode}_teacher_forward"):
            t_logits_batch, teacher_cache = _forward_batch_last_logits(
                teacher_model, t_input, t_mask, past_key_values=teacher_cache,
            )

        # Snapshot the teacher distribution for the whole step in one shot —
        # cheaper than a per-sample topk, and rows for samples that turn out
        # to be finished are simply never appended below.
        if capture:
            with maybe_phase("generation/teacher_cache_capture"):
                step_idx, step_val, step_lse = capture_topk_rows(
                    s_logits_batch,
                    t_logits_batch.to(s_logits_batch.device),
                    capture_teacher_topk,
                    cache_temperature,
                )
                step_idx = step_idx.cpu()
                step_val = step_val.cpu()
                step_lse = step_lse.cpu()

        # Per-sample: decide, sample, record
        with maybe_phase("generation/strategy_and_sample"):
         for i in range(B):
            if done[i]:
                continue

            s_logits = s_logits_batch[i]
            t_logits = t_logits_batch[i].to(s_logits.device)

            decision = strategy.decide(s_logits, t_logits)

            if is_fusion:
                fused = strategy.fuse(s_logits, t_logits)
                chosen_id = _sample_token(fused, temperature, top_k)
                decision.source = "fused"
            elif decision.source == "student":
                chosen_id = _sample_token(s_logits, temperature, top_k)
            else:
                chosen_id = _sample_token(t_logits, temperature, top_k)

            if chosen_id == eos_token_id:
                done[i] = True
                continue

            token_str = tokenizer.decode([chosen_id])
            stats = _compute_per_step_stats(s_logits, t_logits, chosen_id)

            # Appended in lockstep with token_ids — this is what keeps the
            # cache 1:1 with the completion, including the EOS step above,
            # which returns before appending anything.
            if capture:
                cache_steps[i].append((step_idx[i], step_val[i], step_lse[i]))

            outputs[i].tokens.append(token_str)
            outputs[i].token_ids.append(chosen_id)
            outputs[i].provenance.append(decision.source)
            outputs[i].student_logprobs.append(stats["student_logprob"])
            outputs[i].teacher_logprobs.append(stats["teacher_logprob"])
            outputs[i].kl_forward.append(stats["kl_forward"])
            outputs[i].kl_reverse.append(stats["kl_reverse"])
            outputs[i].student_entropy.append(stats["student_entropy"])
            outputs[i].teacher_entropy.append(stats["teacher_entropy"])
            outputs[i].decisions.append({
                "source": decision.source,
                "kl_value": decision.kl_value,
                "entropy_value": decision.entropy_value,
                "overlap_value": decision.overlap_value,
                "reason": decision.reason,
            })

    if capture:
        for i in range(B):
            outputs[i].teacher_cache = pack_steps(cache_steps[i], cache_temperature)

    return outputs


def _sample_token(logits: torch.Tensor, temperature: float, top_k: int = 0) -> int:
    """Sample a single token from logits.

    temperature=0.0 → greedy (argmax)
    temperature>0.0 → multinomial sampling with temperature scaling
    top_k>0 → restrict sampling to top-k tokens
    """
    if temperature == 0.0 or temperature < 1e-8:
        return logits.argmax().item()
    scaled = logits / temperature
    if top_k > 0:
        topk_vals, topk_idx = torch.topk(scaled, top_k)
        scaled = torch.full_like(scaled, float('-inf'))
        scaled.scatter_(0, topk_idx, topk_vals)
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()
