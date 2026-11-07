"""Independent HF oracle for the asynchronous paged-cache implementation."""

import os

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.getenv("PAIRED_GPU_TESTS") != "1", reason="requires two paired CUDA GPUs"
    ),
]


def reference(models, request, threshold, temperature=0, top_k=0, policy="jsd"):
    import torch

    from paired_vllm.async_core import uniform

    tokens, sources, hidden = [], [], []
    for position in range(request.max_tokens):
        logits = []
        for model, prompt in zip(
            models, (request.teacher_prompt, request.student_prompt)
        ):
            with torch.inference_mode():
                states = model.model(
                    torch.tensor([prompt + tokens], device=model.device), use_cache=False
                ).last_hidden_state
                logits.append(model.lm_head(states[:, -1]).float()[0])
            if model is models[0]:
                hidden.append(states[0, -1])
        t, s = [x.softmax(-1) for x in logits]
        m = (t + s) / 2
        if policy == "kl_reverse":
            score = (s * (s.log() - t.log())).sum()
        elif policy == "kl_forward":
            score = (t * (t.log() - s.log())).sum()
        else:
            assert policy == "jsd"
            score = ((t * (t.log() - m.log())).sum() + (s * (s.log() - m.log())).sum()) / 2
        source = "teacher" if score > threshold else "student"
        selected = logits[0 if source == "teacher" else 1]
        if temperature == 0:
            token = selected.argmax().item()
        else:
            values, indices = selected.topk(top_k)
            cumulative = (values / temperature).softmax(-1).cumsum(-1)
            index = min(
                int(
                    (
                        cumulative.double() < uniform(42, request.request_id, position)
                    ).sum()
                ),
                top_k - 1,
            )
            token = indices[index].item()
        tokens.append(token)
        sources.append(source)
        if token in request.stop_tokens:
            break
    return tokens, sources, torch.stack(hidden)


@pytest.mark.parametrize("chunk", [1, 2, 4, 8, 10, 16])
def test_async_matches_full_history_oracle(tmp_path, chunk):
    import torch
    from transformers import AutoModelForCausalLM

    from paired_vllm.async_core import RolloutRequest
    from paired_vllm.async_engine import AsyncConfig, AsyncPairedEngine
    from paired_vllm.core import PairConfig
    from paired_vllm.engine import RuntimeConfig
    from paired_vllm.validation import make_model

    torch.set_num_threads(1)
    teacher = make_model(tmp_path / "teacher", 42)
    student = make_model(tmp_path / "student", 43)
    models = [
        AutoModelForCausalLM.from_pretrained(
            p, dtype=torch.float32, attn_implementation="eager"
        ).eval()
        for p in (teacher, student)
    ]
    requests = [
        RolloutRequest(str(i), [3, 7, 11] * (6 + i), [5] * (15 + i), 37 + i)
        for i in range(3)
    ]
    config = AsyncConfig(
        RuntimeConfig(
            teacher,
            student,
            PairConfig(capacity=1024, token_budget=256, max_requests=4),
            teacher_gpu="1",
            student_gpu="0",
            policy="jsd:0.0108",
            max_model_len=128,
            dtype="float32",
            graphs=chunk in (2, 16),
            capture_targets=True,
        ),
        chunk_size=chunk,
        verify_body_graphs=chunk == 2,
        adaptive_lookahead=chunk == 2,
        max_ahead_chunks=6 if chunk == 2 else 3,
    )
    with AsyncPairedEngine(config) as engine:
        for temperature in (0.0,):
            expected = [
                reference(models, r, config.threshold, temperature) for r in requests
            ]
            actual = engine.generate(requests)
            targets = engine.targets()
            for request, row, (tokens, sources, hidden) in zip(
                requests, actual, expected
            ):
                for i, (actual_token, expected_token) in enumerate(
                    zip(row["token_ids"], tokens)
                ):
                    assert actual_token == expected_token, (
                        i,
                        row["sources"][i],
                        sources[i],
                    )
                assert row["token_ids"] == tokens
                assert row["sources"] == sources
                torch.testing.assert_close(
                    # Native attention and HF eager use different accumulation
                    # paths; token/source equality above remains exact.
                    targets[request.request_id].float(),
                    hidden,
                    atol=0.004,
                    rtol=0.004,
                )
            assert engine.last_metrics["corrections"] > 0
            assert (
                engine.last_metrics["max_unverified_tokens"]
                <= chunk * config.max_ahead_chunks
            )
            # Reusing IDs must release native state and reset both histories.
            assert engine.generate(requests) == actual


def test_async_sampled_eos_and_same_token_teacher(tmp_path):
    import torch
    from transformers import AutoModelForCausalLM

    from paired_vllm.async_core import RolloutRequest
    from paired_vllm.async_engine import AsyncConfig, AsyncPairedEngine
    from paired_vllm.core import PairConfig
    from paired_vllm.engine import RuntimeConfig
    from paired_vllm.validation import make_model

    torch.set_num_threads(1)
    path = make_model(tmp_path / "model", 42, tied=True)
    models = [
        AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.float32, attn_implementation="eager"
        ).eval()
        for _ in range(2)
    ]
    request = RolloutRequest("eos", [3, 7] * 9, [5] * 15, 45)
    expected = reference(models, request, 0.0, 0.8, 20)
    request.stop_tokens = (expected[0][6],)
    expected = reference(models, request, 0.0, 0.8, 20)
    config = AsyncConfig(
        RuntimeConfig(
            path,
            path,
            PairConfig(capacity=512, token_budget=128, max_requests=2),
            teacher_gpu="1",
            student_gpu="0",
            policy="jsd:0",
            max_model_len=128,
            dtype="float32",
            temperature=0.8,
            top_k=20,
            graphs=True,
        ),
        chunk_size=4,
        verify_body_graphs=True,
        adaptive_lookahead=True,
        max_ahead_chunks=4,
    )
    with AsyncPairedEngine(config) as engine:
        row = engine.generate([request])[0]
        assert row["token_ids"] == expected[0]
        assert row["sources"] == expected[1]


@pytest.mark.parametrize("backend", ["FLASH_ATTN", "TRITON_ATTN"])
def test_async_bf16_flashattention_rollbacks(tmp_path, backend):
    import torch
    from transformers import AutoModelForCausalLM

    from paired_vllm.async_core import RolloutRequest
    from paired_vllm.async_engine import AsyncConfig, AsyncPairedEngine
    from paired_vllm.core import PairConfig
    from paired_vllm.engine import RuntimeConfig
    from paired_vllm.validation import make_model

    torch.set_num_threads(1)
    teacher = make_model(tmp_path / "teacher", 42)
    student = make_model(tmp_path / "student", 43)
    models = [
        AutoModelForCausalLM.from_pretrained(
            p, dtype=torch.float32, attn_implementation="eager"
        ).eval()
        for p in (teacher, student)
    ]
    request = RolloutRequest("bf16", [3, 7, 11] * 6, [5] * 15, 33)
    expected = reference(models, request, 0)
    config = AsyncConfig(
        RuntimeConfig(
            teacher,
            student,
            PairConfig(capacity=512, token_budget=128, max_requests=2),
            teacher_gpu="1",
            student_gpu="0",
            policy="jsd:0",
            max_model_len=128,
            dtype="bfloat16",
            graphs=True,
            capture_targets=True,
        ),
        chunk_size=10,
        verify_backend=backend,
        verify_body_graphs=backend == "TRITON_ATTN",
    )
    with AsyncPairedEngine(config) as engine:
        row = engine.generate([request])[0]
        assert row["token_ids"] == expected[0]
        assert row["sources"] == expected[1]
        assert engine.last_metrics["corrections"] > 0
        torch.testing.assert_close(
            engine.targets()["bf16"].float(), expected[2], atol=0.04, rtol=0.04
        )


def test_async_queue_pressure_chunked_prefill_and_kv_reuse(tmp_path):
    import torch
    from transformers import AutoModelForCausalLM

    from paired_vllm.async_core import RolloutRequest
    from paired_vllm.async_engine import AsyncConfig, AsyncPairedEngine
    from paired_vllm.core import PairConfig
    from paired_vllm.engine import RuntimeConfig
    from paired_vllm.validation import make_model

    torch.set_num_threads(1)
    teacher = make_model(tmp_path / "teacher", 42)
    student = make_model(tmp_path / "student", 43)
    models = [
        AutoModelForCausalLM.from_pretrained(
            p, dtype=torch.float32, attn_implementation="eager"
        ).eval()
        for p in (teacher, student)
    ]
    lengths = [1, 15, 16, 17, 63, 64, 65, 97]
    requests = [
        RolloutRequest(
            str(i),
            [3 + (j * 7 + i) % 200 for j in range(lengths[i % 8])],
            [4 + (j * 3 + i) % 200 for j in range(lengths[(i + 3) % 8])],
            33 + i % 4,
        )
        for i in range(64)
    ]
    config = AsyncConfig(
        RuntimeConfig(
            teacher,
            student,
            PairConfig(capacity=256, token_budget=64, max_requests=4),
            teacher_gpu="1",
            student_gpu="0",
            policy="jsd:0",
            max_model_len=256,
            dtype="float32",
            capture_targets=True,
        ),
        chunk_size=16,
        max_ahead_chunks=2,
    )
    expected = {r.request_id: reference(models, r, 0) for r in requests}
    with AsyncPairedEngine(config) as engine:
        for batch in (requests, list(reversed(requests))):
            actual = engine.generate(batch)
            assert [r["request_id"] for r in actual] == [r.request_id for r in batch]
            assert engine.last_metrics["admission_batches"] > 16
            assert engine.last_metrics["corrections"] > 500
            assert engine.last_metrics["max_unverified_tokens"] <= 32
            for row in actual:
                tokens, sources, hidden = expected[row["request_id"]]
                assert row["token_ids"] == tokens
                assert row["sources"] == sources
                torch.testing.assert_close(
                    engine.targets()[row["request_id"]].float(),
                    hidden,
                    atol=0.004,
                    rtol=0.004,
                )
        # Capacity validation must fail before dispatch and leave the engine usable.
        with pytest.raises(ValueError, match="overlong|KV capacity"):
            engine.generate([RolloutRequest("too-big", [3] * 129, [4], 128)])
        assert engine.generate(requests[:1])[0]["token_ids"] == expected["0"][0]
