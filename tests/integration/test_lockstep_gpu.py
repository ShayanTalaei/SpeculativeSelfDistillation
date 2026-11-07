"""GPU feedback, distinct KV layouts, bounded admission and EOS publication."""

import os

import pytest

from tests.integration.test_async_gpu import reference

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.getenv("PAIRED_GPU_TESTS") != "1", reason="requires two CUDA GPUs"
    ),
]


@pytest.mark.parametrize(
    "dtype,graphs,interval,temperature,threshold,transfer_logits,policy",
    [
        ("float32", False, 1, 0.8, 0.0108, False, "jsd"),
        ("float32", True, 8, 0.8, 0.0108, False, "jsd"),
        ("bfloat16", True, 4, 0.0, 0.0, False, "jsd"),
        ("float32", True, 8, 0.8, 0.0108, True, "jsd"),
        ("float32", True, 8, 0.8, 0.0432, True, "kl_reverse"),
        ("float32", True, 8, 0.8, 1.0, True, "kl_reverse"),
    ],
)
def test_lockstep_oracle_eos_capacity_and_id_reuse(
    tmp_path, dtype, graphs, interval, temperature, threshold, transfer_logits, policy
):
    import torch
    from transformers import AutoModelForCausalLM

    from paired_vllm.async_core import RolloutRequest
    from paired_vllm.core import PairConfig
    from paired_vllm.engine import RuntimeConfig
    from paired_vllm.lockstep_engine import LockstepPairedEngine
    from paired_vllm.validation import make_model

    torch.set_num_threads(1)
    paths = [
        make_model(tmp_path / name, seed)
        for name, seed in (("teacher", 42), ("student", 43))
    ]
    models = [
        AutoModelForCausalLM.from_pretrained(
            p,
            dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float32,
            attn_implementation="eager",
        ).eval()
        for p in paths
    ]
    if dtype == "bfloat16":
        # BF16 attention can change near-tied argmax decisions relative to
        # full-history eager attention. Check numerical agreement along the
        # actual prefix below; FP32 retains strict token/source equality.
        models = [model.to("cuda:0") for model in models]
    requests = [
        RolloutRequest(str(i), [3, 7, 11] * (11 + i), [5] * (25 - i), 21 + i)
        for i in range(8)
    ]
    expected = {}
    for i, request in enumerate(requests):
        value = reference(models, request, threshold, temperature, 20, policy)
        if i % 3 == 0:
            request.stop_tokens = (value[0][5],)
            value = reference(models, request, threshold, temperature, 20, policy)
        expected[request.request_id] = (value[0], value[1], value[2].cpu())
    runtime = RuntimeConfig(
        *paths,
        PairConfig(capacity=256, token_budget=64, max_requests=3),
        teacher_gpu="1",
        student_gpu="0",
        policy=f"{policy}:{threshold}",
        max_model_len=128,
        dtype=dtype,
        graphs=graphs,
        capture_targets=True,
        temperature=temperature,
        top_k=20,
    )
    with LockstepPairedEngine(
        runtime, readback_interval=interval, transfer_logits=transfer_logits,
        fused_jsd=policy != "jsd",
    ) as engine:
        with pytest.raises(ValueError, match="overlong"):
            engine.generate([RolloutRequest("too-long", [3] * 100, [5], 50)])
        for order in (requests, list(reversed(requests))):
            rows = engine.generate(order)
            targets = engine.targets()
            assert engine.last_metrics["admission_batches"] > 1
            assert engine.last_metrics["max_unverified_tokens"] == 0
            assert engine.last_metrics["corrections"] == 0
            assert (
                engine.last_metrics["model_positions"]
                == engine.last_metrics["verified_tokens"]
                + engine.last_metrics["positions_after_eos"]
            )
            for row in rows:
                tokens, sources, hidden = expected[row["request_id"]]
                if dtype == "bfloat16":
                    request = next(
                        r for r in requests if r.request_id == row["request_id"]
                    )
                    tokens = row["token_ids"]
                    assert row["sources"] == ["teacher"] * len(tokens)
                    assert not any(t in request.stop_tokens for t in tokens[:-1])
                    assert (
                        len(tokens) == request.max_tokens
                        or tokens[-1] in request.stop_tokens
                    )
                    with torch.inference_mode():
                        ids = torch.tensor(
                            [request.teacher_prompt + tokens[:-1]],
                            device=models[0].device,
                        )
                        hidden = (
                            models[0]
                            .model(ids, use_cache=False)
                            .last_hidden_state[0, len(request.teacher_prompt) - 1 :]
                        )
                        logits = models[0].lm_head(hidden).float()
                        selected = logits.gather(
                            1, torch.tensor(tokens, device=logits.device)[:, None]
                        )[:, 0]
                        # Eight millilogits bounds the observed BF16 kernel
                        # rounding, while rejecting a stale/wrong prefix.
                        assert (logits.max(-1).values - selected).max().item() <= 0.008
                    hidden = hidden.cpu()
                else:
                    assert row["token_ids"] == tokens
                    assert row["sources"] == sources
                tolerance = 0.04 if dtype == "bfloat16" else 0.004
                torch.testing.assert_close(
                    targets[row["request_id"]].float(),
                    hidden.float(),
                    atol=tolerance,
                    rtol=tolerance,
                )


def _worker_with_injected_oom(*args):
    """Fail one peer after prefill while its partner enters GPU receive."""
    from paired_vllm.lockstep_engine import LockstepWorker, process_main

    original = LockstepWorker.prefill

    def prefill(self, payload):
        result = original(self, payload)
        if self.role == 0:
            raise self.torch.cuda.OutOfMemoryError(
                "injected teacher allocation failure"
            )
        return result

    LockstepWorker.prefill = prefill
    process_main(*args)


def test_lockstep_allocation_failure_closes_both_workers(tmp_path, monkeypatch):
    from paired_vllm.async_core import RolloutRequest
    from paired_vllm.core import PairConfig
    from paired_vllm.engine import RuntimeConfig
    from paired_vllm.lockstep_engine import LockstepPairedEngine
    from paired_vllm.validation import make_model

    model = make_model(tmp_path / "model", 42)
    runtime = RuntimeConfig(
        model,
        model,
        PairConfig(capacity=128, token_budget=64, max_requests=1),
        teacher_gpu="1",
        student_gpu="0",
        policy="jsd:0.25",
        max_model_len=128,
        dtype="float32",
        graphs=False,
        timeout=30,
    )
    monkeypatch.setattr(
        "paired_vllm.lockstep_engine.process_main", _worker_with_injected_oom
    )
    engine = LockstepPairedEngine(runtime)
    with engine:
        with pytest.raises(RuntimeError, match="injected teacher allocation failure"):
            engine.generate([RolloutRequest("r", [3] * 17, [5] * 9, 16)])
        assert engine.closed
        assert all(not process.is_alive() for process in engine.workers.values())
        with pytest.raises(RuntimeError, match="engine is closed"):
            engine.generate([RolloutRequest("r", [3], [5], 1)])
