"""Opt-in real MRV2/Hugging Face equivalence, including different batch orders.

Run with PAIRED_GPU_TESTS=1 in the isolated paired environment.
"""

import os

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.getenv("PAIRED_GPU_TESTS") != "1",
        reason="requires two GPUs and the paired environment",
    ),
]


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("optimized", [False, True])
def test_training_cache_weight_refresh_and_sleep_match_fresh_teacher(
    tmp_path, tied, optimized
):
    import torch
    from transformers import AutoModelForCausalLM

    from paired_vllm.core import PairConfig
    from paired_vllm.engine import PairedEngine, RuntimeConfig
    from paired_vllm.training_ops import write_weights
    from paired_vllm.validation import make_model

    model_path = make_model(tmp_path / "model", 42, tied=tied)
    reference = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, attn_implementation="eager"
    ).eval()
    state = write_weights(tmp_path / "current", reference.state_dict())
    config = RuntimeConfig(
        model_path,
        model_path,
        PairConfig(capacity=256, token_budget=16, max_requests=4),
        max_model_len=128,
        capture_targets=True,
        enable_sleep_mode=True,
        graphs=optimized,
        compile_model=optimized,
        gpu_target_cache=optimized,
        fused_dispatch=optimized,
    )
    with PairedEngine(config) as engine:
        for version in ("v0", "v1"):
            if version == "v1":
                with torch.no_grad():
                    reference.lm_head.weight.mul_(-1)
                    for name, value in reference.state_dict().items():
                        state[name].copy_(value)
                engine.training_command("SLEEP")
                engine.training_command("WAKE")
            engine.training_command(
                "SYNC",
                {
                    "teacher": str(tmp_path / "current"),
                    "student": str(tmp_path / "current"),
                    "version": version,
                    "seed": 42,
                },
            )
            engine.add("r", [3] * 17, [4] * 5, max_tokens=3)
            engine.run()
            req = engine.core.requests["r"]
            targets = engine.training_command(
                "TARGETS", [("r", req.incarnation, 3, version)]
            )[-1]["r"]
            with torch.no_grad():
                fresh = reference.model(
                    torch.tensor([[3] * 17 + req.committed[:-1]]), use_cache=False
                ).last_hidden_state[0, 16:]
                cached_logits = targets @ reference.lm_head.weight.T
                fresh_logits = fresh @ reference.lm_head.weight.T
            torch.testing.assert_close(
                cached_logits, fresh_logits, atol=0.004, rtol=0.015
            )
            engine.take_finished()


@pytest.mark.parametrize(
    "graphs,optimized", [(False, False), (True, False), (True, True)]
)
def test_frozen_teacher_reuses_hidden_states_across_student_sleep(
    tmp_path, graphs, optimized
):
    from copy import deepcopy

    import torch
    from transformers import AutoModelForCausalLM

    from paired_vllm.core import PairConfig
    from paired_vllm.engine import PairedEngine, RuntimeConfig
    from paired_vllm.training_ops import write_weights
    from paired_vllm.validation import make_model

    model_path = make_model(tmp_path / "model", 42, tied=True)
    reference = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, attn_implementation="eager"
    ).eval()
    state = write_weights(tmp_path / "current", reference.state_dict())
    student_reference = deepcopy(reference)
    with PairedEngine(
        RuntimeConfig(
            model_path,
            model_path,
            PairConfig(capacity=256, token_budget=16, max_requests=4),
            max_model_len=128,
            capture_targets=True,
            enable_sleep_mode=True,
            keep_teacher_resident=True,
            graphs=graphs,
            compile_model=optimized,
            gpu_target_cache=optimized,
            fused_dispatch=optimized,
        )
    ) as engine:
        for step in range(2):
            if step:
                # Change the upload checkpoint, but refresh only the student.
                # Teacher body AND student-side teacher head must stay frozen.
                seen = set()
                for value in state.values():
                    if value.data_ptr() not in seen:
                        seen.add(value.data_ptr())
                        value.neg_()
                student_reference.load_state_dict(state)
                engine.training_command("SLEEP")
                engine.training_command("WAKE")
            version = str(step)
            engine.training_command(
                "SYNC",
                {
                    "teacher": str(tmp_path / "current"),
                    "student": str(tmp_path / "current"),
                    "sync_teacher": step == 0,
                    "version": version,
                    "seed": 42,
                },
            )
            engine.add("r", [3] * 17, [4] * 5, max_tokens=3)
            engine.run()
            req = engine.core.requests["r"]
            targets = engine.training_command(
                "TARGETS", [("r", req.incarnation, 3, version)]
            )[-1]["r"]
            with torch.no_grad():
                fresh = reference.model(
                    torch.tensor([[3] * 17 + req.committed[:-1]]), use_cache=False
                ).last_hidden_state[0, 16:]
                fresh_logits = fresh @ reference.lm_head.weight.T
            torch.testing.assert_close(
                targets @ reference.lm_head.weight.T,
                fresh_logits,
                atol=0.004,
                rtol=0.015,
            )
            inspected = engine.inspect()[-1][f"r:{req.incarnation}"]
            routed = inspected["teacher_logits"]
            torch.testing.assert_close(
                torch.tensor(routed),
                fresh_logits[-1].float(),
                atol=0.004,
                rtol=0.015,
            )
            with torch.no_grad():
                student_logits = (
                    student_reference(
                        torch.tensor([[4] * 5 + req.committed[:-1]]), use_cache=False
                    )
                    .logits[0, -1]
                    .float()
                )
            torch.testing.assert_close(
                torch.tensor(inspected["logits"]),
                student_logits,
                atol=0.004,
                rtol=0.015,
            )
            engine.take_finished()


@pytest.mark.parametrize("graphs", [False, True])
def test_mrv2_common_commit_matches_independent_reference(tmp_path, graphs):
    from paired_vllm.validation import validate

    result = validate(
        tmp_path,
        gaps=(0, 5, 16, 2048),
        policies=("teacher", "student", "alternating"),
        graphs=graphs,
    )
    assert result["max_logit_error"] < 0.08
    assert result["tokens_match"]
    assert result["histories_match"]


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("failure_phase", ["RUN", "FINALIZE"])
def test_worker_epoch_recovery_forces_committed_tokens(tmp_path, failure_phase, fused):
    from paired_vllm.core import PairConfig
    from paired_vllm.engine import PairedEngine, RuntimeConfig
    from paired_vllm.validation import make_model

    teacher, student = (
        make_model(tmp_path / side, seed)
        for side, seed in (("teacher", 42), ("student", 43))
    )
    config = RuntimeConfig(
        teacher,
        student,
        PairConfig(capacity=256, token_budget=16, max_requests=4),
        max_model_len=128,
        fused_dispatch=fused,
    )
    with PairedEngine(config) as engine:
        engine.add("a", [3] * 35, [4] * 5, max_tokens=1)
        original_receive = engine._receive

        def fail_after_reply(command, timeout=None):
            result = original_receive(command, timeout)
            expected = (
                "PREPARE_RUN" if fused and failure_phase == "RUN" else failure_phase
            )
            if command == expected and engine.core.inflight.ready:
                engine.workers[0].terminate()
                engine.workers[0].join(timeout=2)
                raise RuntimeError("injected worker failure")
            return result

        engine._receive = fail_after_reply
        with pytest.raises(RuntimeError, match="injected"):
            engine.run()
        committed = engine.core.requests["a"].committed.copy()
        assert bool(committed) == (failure_phase == "FINALIZE")
        assert all(not p.is_alive() for p in engine.workers)
        engine._receive = original_receive
        engine.recover()
        events = engine.run()
        assert len(events) == 1
        assert engine.core.requests["a"].committed == [events[0].token_id]
        if committed:
            assert committed == [events[0].token_id]
            assert events[0].incarnation == 0  # Original commit provenance retained.
        states = engine.inspect()
        assert states[0]["a:1"]["tokens"] == [3] * 35 + [events[0].token_id]
        assert states[1]["a:1"]["tokens"] == [4] * 5 + [events[0].token_id]


@pytest.mark.parametrize("fused", [False, True])
def test_cancel_issued_transfer_drains_before_release(tmp_path, fused):
    from paired_vllm.core import PairConfig
    from paired_vllm.engine import PairedEngine, RuntimeConfig
    from paired_vllm.validation import make_model

    model = make_model(tmp_path / "model", 42)
    with PairedEngine(
        RuntimeConfig(
            model,
            model,
            PairConfig(capacity=256),
            max_model_len=128,
            fused_dispatch=fused,
        )
    ) as engine:
        engine.add("cancel", [3] * 17, [4] * 5, max_tokens=4)
        original_receive = engine._receive

        def cancel_during_run(command, timeout=None):
            if command == ("PREPARE_RUN" if fused else "RUN"):
                engine.cancel("cancel")
                assert all(p.reserved > 0 for p in engine.core.pools)
            return original_receive(command, timeout)

        engine._receive = cancel_during_run
        assert engine.run() == []
        assert all(p.reserved == 0 for p in engine.core.pools)
        assert engine.core.requests["cancel"].status == "cancelled"


@pytest.mark.parametrize("graphs", [False, True])
def test_head_gpu_matches_cpu_as_batch_rows_change(graphs):
    import torch

    from paired_vllm.head import DecisionHead

    torch.manual_seed(42)
    weight = torch.randn(64, 32, device="cuda")
    for policy in (
        "teacher",
        "student",
        "alternating",
        "kl_forward:0.5",
        "kl_reverse:0.5",
        "jsd:0.1",
    ):
        features = torch.zeros(4, 32, device="cuda")
        eager_features = torch.zeros_like(features, device="cpu")
        routes = torch.empty(4, dtype=torch.int32, device="cuda")
        tokens = torch.empty(4, dtype=torch.int64, device="cuda")
        eager_routes, eager_tokens = (
            torch.empty_like(routes, device="cpu"),
            torch.empty_like(tokens, device="cpu"),
        )
        gpu = DecisionHead(
            torch, weight, features, routes, tokens, policy, enable_graphs=graphs
        )
        eager = DecisionHead(
            torch, weight.cpu(), eager_features, eager_routes, eager_tokens, policy
        )
        address = features.data_ptr()
        for step, n in enumerate((3, 1, 4, 2, 3)):
            features.copy_(torch.randn_like(features))
            eager_features.copy_(features)
            own = torch.randn(n, 64, device="cuda")
            indices = list(range(step, step + n))
            torch.testing.assert_close(
                gpu(own, indices).cpu(), eager(own.cpu(), indices)
            )
            assert torch.equal(routes[:n].cpu(), eager_routes[:n])
            assert torch.equal(tokens[:n].cpu(), eager_tokens[:n])
            assert features.data_ptr() == address


@pytest.mark.parametrize("optimized", [False, True])
def test_queued_arrivals_retirement_and_slot_reuse_preserve_tokens_and_targets(
    tmp_path,
    optimized,
):
    """More queued rollouts than slots, late arrivals, cancellation and graph replay."""
    import torch
    from transformers import AutoModelForCausalLM

    from paired_vllm.core import PairConfig
    from paired_vllm.engine import PairedEngine, RuntimeConfig
    from paired_vllm.validation import make_model

    path = make_model(tmp_path / "model", 42, tied=True)
    reference = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    config = RuntimeConfig(
        path,
        path,
        PairConfig(capacity=256, token_budget=16, max_requests=2),
        max_model_len=128,
        policy="alternating",
        graphs=True,
        capture_targets=True,
        compile_model=optimized,
        gpu_target_cache=optimized,
        fused_dispatch=optimized,
    )
    specifications = {
        f"r{i}": ([3] * (17 + i * 7), [4] * (3 + i), i + 2) for i in range(6)
    }
    events = []
    with PairedEngine(config) as engine:
        for rid, (teacher, student, length) in specifications.items():
            engine.add(rid, teacher, student, length)
        saw_waiting = saw_replacement = False
        tick = 0
        while engine.core.unfinished:
            events.extend(engine.step())
            tick += 1
            saw_waiting |= any(
                r.status == "waiting" for r in engine.core.requests.values()
            )
            if tick == 2:
                specifications["late"] = ([5] * 35, [6] * 6, 4)
                engine.add("late", *specifications["late"])
                engine.cancel("r1")
            saw_replacement |= engine.core.requests["r0"].status == "finished" and any(
                r.status == "active"
                for rid, r in engine.core.requests.items()
                if rid != "r0"
            )
        assert saw_waiting and saw_replacement
        assert engine.core.requests["r1"].status == "cancelled"
        assert all(pool.reserved == 0 for pool in engine.core.pools)
        requests = engine.core.requests
        nonempty = [
            (rid, r.incarnation, len(r.committed), engine.core.config.teacher_revision)
            for rid, r in requests.items()
            if r.committed
        ]
        targets = engine.training_command("TARGETS", nonempty)[-1]
        for rid, request in requests.items():
            teacher, student, length = specifications[rid]
            if request.status != "cancelled":
                assert len(request.committed) == length
            emitted = [e for e in events if e.request_id == rid]
            assert [e.output_index for e in emitted] == list(
                range(len(request.committed))
            )
            assert [e.token_id for e in emitted] == request.committed
            for index, token in enumerate(request.committed):
                source = "teacher" if index % 2 == 0 else "student"
                assert emitted[index].source_model == source
                prompt = teacher if source == "teacher" else student
                with torch.no_grad():
                    expected = reference(
                        torch.tensor([prompt + request.committed[:index]])
                    ).logits[0, -1]
                    teacher_hidden = reference.model(
                        torch.tensor([teacher + request.committed[:index]])
                    ).last_hidden_state[0, -1]
                assert token == expected.argmax().item()
                torch.testing.assert_close(
                    targets[rid][index].float(), teacher_hidden, atol=0.008, rtol=0.025
                )
        engine.take_finished()


def test_fused_prepare_failure_closes_both_peers_without_committing(tmp_path):
    from dataclasses import replace

    from paired_vllm.core import PairConfig
    from paired_vllm.engine import PairedEngine, RuntimeConfig
    from paired_vllm.validation import make_model

    model = make_model(tmp_path / "model", 42)
    with PairedEngine(
        RuntimeConfig(
            model,
            model,
            PairConfig(capacity=256, token_budget=16, max_requests=2),
            max_model_len=128,
            fused_dispatch=True,
            timeout=15,
        )
    ) as engine:
        original_send = engine._send

        def corrupt_teacher(command, payload):
            if command == "PREPARE_RUN":
                bad = replace(
                    payload, contract=("wrong-teacher", *payload.contract[1:])
                )
                engine.connections[0].send((command, bad))
                engine.connections[1].send((command, payload))
            else:
                original_send(command, payload)

        engine._send = corrupt_teacher
        engine.add("r", [3] * 5, [4] * 5, 1)
        with pytest.raises(RuntimeError, match="checkpoint revision mismatch"):
            engine.run()
        assert not engine.core.requests["r"].committed
        assert all(not p.is_alive() for p in engine.workers)


def test_graphed_sampling_advances_rng_and_respects_topk():
    import torch

    from paired_vllm.head import DecisionHead

    weight = torch.zeros(17, 4, device="cuda")
    features = torch.zeros(4, 4, device="cuda")
    routes = torch.empty(4, dtype=torch.int32, device="cuda")
    tokens = torch.empty(4, dtype=torch.int64, device="cuda")
    head = DecisionHead(
        torch,
        weight,
        features,
        routes,
        tokens,
        "teacher",
        temperature=1.0,
        top_k=3,
        enable_graphs=True,
    )
    allowed = set(torch.zeros(17, device="cuda").topk(3).indices.tolist())
    observed = set()
    for n in (3, 1, 4, 2) * 10:
        head(torch.zeros(n, 17, device="cuda"), list(range(n)))
        observed.update(tokens[:n].tolist())
        assert routes[:n].eq(1).all()
    assert observed <= allowed
    assert len(observed) > 1, "graph replay must not repeat a frozen RNG draw"
