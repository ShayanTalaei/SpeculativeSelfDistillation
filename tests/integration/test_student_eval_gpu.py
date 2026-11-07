"""Reuse student allocations for continuous evaluation, then resume rollouts."""

import os

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.getenv("PAIRED_GPU_TESTS") != "1", reason="requires two GPUs"
    ),
]


def test_student_only_continuous_eval_matches_greedy_and_resumes(tmp_path, monkeypatch):
    from paired_vllm.async_core import RolloutRequest
    from paired_vllm.core import PairConfig
    from paired_vllm.engine import RuntimeConfig
    from paired_vllm.lockstep_engine import LockstepPairedEngine
    from paired_vllm.validation import make_model

    teacher = make_model(tmp_path / "teacher", 42)
    student = make_model(tmp_path / "student", 43)
    runtime = RuntimeConfig(
        teacher,
        student,
        PairConfig(capacity=256, token_budget=128, max_requests=4),
        teacher_gpu="1",
        student_gpu="0",
        policy="jsd:0.25",
        max_model_len=64,
        dtype="float32",
        graphs=False,
        capture_targets=False,
        temperature=0.0,
        top_k=20,
    )
    prompts = [[3, 7, 11] * (1 + i % 5) for i in range(19)]
    requests = [RolloutRequest(str(i), p, p, 8) for i, p in enumerate(prompts)]
    with LockstepPairedEngine(runtime, transfer_logits=True) as engine:
        engine.training_command(
            "SAMPLING",
            {"policy": "student", "temperature": 0.0, "top_k": 20, "capture": False},
        )
        expected = engine.generate(requests)
        calls = []
        send = engine._send

        def tracked(side, command, payload=None):
            calls.append((side, command))
            return send(side, command, payload)

        monkeypatch.setattr(engine, "_send", tracked)
        actual = engine.evaluate_student(
            {
                "student": prompts,
                "max_tokens": 8,
                "temperature": 0.0,
                "top_k": 20,
                "stop_tokens": [],
            }
        )
        assert calls == [(1, "EVALUATE")]
        assert [row["token_ids"] for row in actual] == [
            row["token_ids"] for row in expected
        ]
        timings = actual[0]["timings"]
        assert timings["teacher_forward_steps"] == 0
        assert timings["max_batch"] <= 4
        assert timings["continuous_refills"] >= 15
        # All later requests still honor EOS, and callback request IDs can be reused.
        stop = actual[0]["token_ids"][0]
        stopped = engine.evaluate_student(
            {
                "student": prompts,
                "max_tokens": 8,
                "temperature": 0.0,
                "top_k": 20,
                "stop_tokens": [stop],
            }
        )
        for row in stopped:
            assert stop not in row["token_ids"][:-1]
        assert len(stopped[0]["token_ids"]) == 1
        # Both the stock scheduler and lockstep metadata must be fully retired.
        resumed = engine.generate(requests[:2])
        assert [r["token_ids"] for r in resumed] == [
            r["token_ids"] for r in expected[:2]
        ]
