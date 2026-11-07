"""End-to-end check that the generation-time teacher cache reproduces the
teacher forward the loss would otherwise run.

This is the property the optimisation rests on: generation step ``t`` and loss
position ``t`` describe the same distribution — the one that produced
completion token ``t``. If that is off by even one position the loss trains
against shifted targets, and nothing downstream would notice.

Runs on a small randomly-initialised Qwen3 on CPU. Random weights are fine and
in fact preferable here: alignment is a plumbing property, and untrained
logits are less peaked than trained ones, so any misalignment shows up as a
*larger* discrepancy than it would in a real run.

The loss-side recompute here is deliberately fed the same token ids generation
used, one sequence at a time. That isolates the question being asked —
alignment — from two known differences in the production path: the trainer
tokenizes prompts with ``add_special_tokens=False`` plus truncation, where
generation uses neither, and it batches with left-padding without passing
``position_ids``. Those are pre-existing discrepancies between the two teacher
evaluations, measured at runtime by ``teacher_cache='verify'``.
"""

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from ssd.generation import ssd_generate_batch
from ssd.strategies import KLSwitchStrategy
from ssd.teacher_cache import stack_caches, teacher_logprobs_from_cache

MODEL_ID = "Qwen/Qwen3-0.6B"
K = 100
IMPOSSIBLE_EOS = -1  # never equals a sampled id, so nothing stops early


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)


@pytest.fixture(scope="module")
def models(tokenizer):
    """Two small random Qwen3s — same architecture as the real runs."""
    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
    )
    student = Qwen3ForCausalLM(cfg).eval()
    torch.manual_seed(1)
    teacher = Qwen3ForCausalLM(cfg).eval()
    return student, teacher


PROMPTS = [
    ("Question: what is 2+2?", "Context: arithmetic. Question: what is 2+2?"),
    ("Name a colour.", "Context: the sky is blue. Name a colour."),
    ("Count:", "Context: numbers go 1 2 3. Count:"),
]


def _generate(student, teacher, tokenizer, *, eos_token_id, max_new_tokens=8):
    return ssd_generate_batch(
        student_model=student,
        teacher_model=teacher,
        tokenizer=tokenizer,
        student_prompts=[p[0] for p in PROMPTS],
        teacher_prompts=[p[1] for p in PROMPTS],
        strategy=KLSwitchStrategy(threshold=0.5, direction="forward"),
        max_new_tokens=max_new_tokens,
        temperature=0.0,  # greedy, so the run is reproducible
        eos_token_id=eos_token_id,
        capture_teacher_topk=K,
        cache_temperature=1.0,  # the loss temperature, not the sampling one
    )


def _loss_side_teacher_logps(teacher, tokenizer, teacher_prompt, token_ids, temperature=1.0):
    """Recompute exactly as _get_per_token_logps_and_entropies does.

    Full sequence in, drop the final position, keep the last
    ``logits_to_keep``, divide by temperature, log_softmax.
    """
    prompt_ids = tokenizer.encode(teacher_prompt)
    input_ids = torch.tensor([prompt_ids + list(token_ids)])
    logits_to_keep = len(token_ids)

    with torch.no_grad():
        logits = teacher(input_ids=input_ids, use_cache=False).logits

    logits = logits[:, :-1, :]
    logits = logits[:, -logits_to_keep:, :]
    return F.log_softmax(logits.float() / temperature, dim=-1)


class TestAlignment:
    def test_one_cache_row_per_generated_token(self, models, tokenizer):
        student, teacher = models
        outs = _generate(student, teacher, tokenizer, eos_token_id=IMPOSSIBLE_EOS)

        for i, gen in enumerate(outs):
            assert gen.teacher_cache is not None, f"sample {i} came back without a cache"
            assert len(gen.teacher_cache) == len(gen.token_ids), (
                f"sample {i}: {len(gen.teacher_cache)} cached rows vs "
                f"{len(gen.token_ids)} tokens"
            )

    def test_cached_logprobs_match_a_fresh_teacher_forward(self, models, tokenizer):
        student, teacher = models
        outs = _generate(student, teacher, tokenizer, eos_token_id=IMPOSSIBLE_EOS)

        worst = 0.0
        for i, gen in enumerate(outs):
            n = len(gen.token_ids)
            fresh = _loss_side_teacher_logps(
                teacher, tokenizer, PROMPTS[i][1], gen.token_ids
            )
            idx, val, lse, valid = stack_caches([gen.teacher_cache], n, torch.device("cpu"))
            cached = teacher_logprobs_from_cache(val, lse, valid, 1.0)
            gathered = fresh.gather(-1, idx.clamp(min=0))

            err = float(((cached - gathered).abs() * valid).max())
            worst = max(worst, err)
            assert err < 1e-4, f"sample {i}: cached teacher log-probs off by {err}"

        print(f"\nworst cached-vs-recomputed log-prob error: {worst:.3e}")

    def test_a_shifted_comparison_would_fail(self, models, tokenizer):
        """Guard on the guard: the check above must be able to detect a shift."""
        student, teacher = models
        gen = _generate(student, teacher, tokenizer, eos_token_id=IMPOSSIBLE_EOS)[0]
        n = len(gen.token_ids)

        fresh = _loss_side_teacher_logps(teacher, tokenizer, PROMPTS[0][1], gen.token_ids)
        idx, val, lse, valid = stack_caches([gen.teacher_cache], n, torch.device("cpu"))
        cached = teacher_logprobs_from_cache(val, lse, valid, 1.0)
        shifted = fresh.roll(1, dims=1).gather(-1, idx.clamp(min=0))

        err = float(((cached - shifted).abs() * valid).max())
        assert err > 1e-2, f"a one-position shift was indistinguishable ({err})"


class TestEarlyStop:
    """The EOS break is where an off-by-one would most plausibly creep in:
    generation returns *before* appending, so the cache must not gain a row."""

    def test_alignment_holds_when_a_sequence_stops_early(self, models, tokenizer):
        student, teacher = models

        # Discover a token the greedy run actually emits, then make it EOS so
        # the second run provably stops early at a known position.
        baseline = _generate(student, teacher, tokenizer, eos_token_id=IMPOSSIBLE_EOS)
        stop_at = 3
        eos_id = baseline[0].token_ids[stop_at]

        outs = _generate(student, teacher, tokenizer, eos_token_id=eos_id)

        assert len(outs[0].token_ids) == stop_at, (
            f"expected the run to stop at {stop_at} tokens, got "
            f"{len(outs[0].token_ids)}"
        )
        for i, gen in enumerate(outs):
            assert len(gen.teacher_cache) == len(gen.token_ids), (
                f"sample {i}: EOS left the cache and the completion out of step "
                f"({len(gen.teacher_cache)} vs {len(gen.token_ids)})"
            )
            if not gen.token_ids:
                continue
            fresh = _loss_side_teacher_logps(
                teacher, tokenizer, PROMPTS[i][1], gen.token_ids
            )
            idx, val, lse, valid = stack_caches(
                [gen.teacher_cache], len(gen.token_ids), torch.device("cpu")
            )
            cached = teacher_logprobs_from_cache(val, lse, valid, 1.0)
            err = float(((cached - fresh.gather(-1, idx.clamp(min=0))).abs() * valid).max())
            assert err < 1e-4, f"sample {i} misaligned after early stop: {err}"


class TestGenerationConsistency:
    def test_chosen_token_logprob_agrees_with_generation_stats(self, models, tokenizer):
        """Cross-check against the teacher log-prob generation recorded itself.

        Independent of any model rerun: if the cache were shifted relative to
        the emitted tokens, these two would disagree.
        """
        student, teacher = models
        outs = _generate(student, teacher, tokenizer, eos_token_id=IMPOSSIBLE_EOS)

        for i, gen in enumerate(outs):
            n = len(gen.token_ids)
            idx, val, lse, valid = stack_caches([gen.teacher_cache], n, torch.device("cpu"))
            cached = teacher_logprobs_from_cache(val, lse, valid, 1.0)

            for t, tok in enumerate(gen.token_ids):
                hit = (idx[0, t] == tok).nonzero()
                # The emitted token is by construction in one of the two
                # top-Ks, so it must be present in the union.
                assert hit.numel() > 0, f"sample {i} token {t} missing from the union"
                lp = float(cached[0, t, hit[0, 0]])
                assert abs(lp - gen.teacher_logprobs[t]) < 1e-4, (
                    f"sample {i} token {t}: cache says {lp}, generation "
                    f"recorded {gen.teacher_logprobs[t]}"
                )

    def test_capture_is_off_by_default(self, models, tokenizer):
        student, teacher = models
        outs = ssd_generate_batch(
            student_model=student,
            teacher_model=teacher,
            tokenizer=tokenizer,
            student_prompts=[PROMPTS[0][0]],
            teacher_prompts=[PROMPTS[0][1]],
            strategy=KLSwitchStrategy(threshold=0.5, direction="forward"),
            max_new_tokens=4,
            temperature=0.0,
            eos_token_id=IMPOSSIBLE_EOS,
        )
        assert outs[0].teacher_cache is None

    def test_greedy_sampling_requires_an_explicit_cache_temperature(self, models, tokenizer):
        """temperature=0 is a valid sampling mode but an invalid divisor."""
        student, teacher = models
        with pytest.raises(ValueError, match="cache_temperature"):
            ssd_generate_batch(
                student_model=student,
                teacher_model=teacher,
                tokenizer=tokenizer,
                student_prompts=[PROMPTS[0][0]],
                teacher_prompts=[PROMPTS[0][1]],
                strategy=KLSwitchStrategy(threshold=0.5, direction="forward"),
                max_new_tokens=2,
                temperature=0.0,
                eos_token_id=IMPOSSIBLE_EOS,
                capture_teacher_topk=K,
            )
