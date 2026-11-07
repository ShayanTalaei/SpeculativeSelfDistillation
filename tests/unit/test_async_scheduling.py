"""Exercise the coordinator ordering without CUDA or timing assumptions."""

from collections import deque
from types import SimpleNamespace

from paired_vllm.async_core import Draft, Lookahead, RolloutRequest
from paired_vllm.async_engine import AsyncConfig, AsyncPairedEngine
from paired_vllm.core import PairConfig
from paired_vllm.engine import RuntimeConfig


def test_adaptive_lead_grows_on_acceptance_and_shrinks_on_correction():
    lead = Lookahead(chunk=4, maximum=24, adaptive=True)
    assert lead.limit == 8
    for _ in range(16):
        lead.observe(4, corrected=False)
    assert lead.limit == 24
    lead.observe(1, corrected=True)
    assert lead.limit == 12
    assert lead.clean_tokens == 0
    lead.observe(1, corrected=True)
    assert lead.limit == 8
    lead.observe(1, corrected=True)
    assert lead.limit == 8
    for _ in range(40):
        lead.observe(4, corrected=False)
    assert lead.limit == 24


def test_fixed_lead_ignores_corrections():
    lead = Lookahead(chunk=4, maximum=24)
    lead.observe(1, corrected=True)
    assert lead.limit == 24


def test_student_drafts_before_teacher_prefill_finishes(monkeypatch):
    engine = AsyncPairedEngine.__new__(AsyncPairedEngine)
    engine.config = AsyncConfig(
        RuntimeConfig(
            "teacher",
            "student",
            PairConfig(capacity=128, token_budget=8, max_requests=1),
            policy="jsd:.25",
            max_model_len=128,
        ),
        chunk_size=2,
    )
    queues = {s: deque() for s in (0, 1)}
    engine.connections = {s: SimpleNamespace(side=s) for s in (0, 1)}
    engine.expected = {s: [] for s in (0, 1)}
    drafted = []
    events = []

    def send(side, command, payload):
        events.append(("send", side, command))
        engine.expected[side].append(command)
        result = True
        if command == "DRAFT":
            limits, _, _ = payload
            start = len(drafted)
            tokens = [7] * min(limits["r"], 6 - start)
            drafted.extend(tokens)
            result = {
                "generated_tokens": len(tokens),
                "interrupted_tokens": 0,
                "drafts": [Draft("r", 0, start, tokens)],
            }
        elif command == "VERIFY":
            result = {
                "seconds": 0.0,
                "positions": sum(len(d.tokens) for d in payload),
                "rows": [
                    (d, d.tokens, ["student"] * len(d.tokens), []) for d in payload
                ],
            }
        queues[side].append((command, result))

    def receive(side):
        command, result = queues[side].popleft()
        assert engine.expected[side].pop(0) == command
        events.append(("receive", side, command))
        return command, result

    def ready(connections, timeout):
        available = [
            c
            for c in connections
            if queues[c.side]
            and not (c.side == 0 and queues[0][0][0] == "PREFILL" and not drafted)
        ]
        assert available, "student blocked behind teacher prefill"
        return available

    engine._send = send
    engine._receive = receive
    monkeypatch.setattr("paired_vllm.async_engine.wait", ready)
    rows = engine._generate([RolloutRequest("r", [1] * 64, [2], 6)])
    assert rows[0]["token_ids"] == [7] * 6
    assert events.index(("send", 1, "DRAFT")) < events.index(("receive", 0, "PREFILL"))
    assert events.index(("receive", 0, "PREFILL")) < events.index(("send", 0, "VERIFY"))
    assert engine.last_metrics["student_chunks_during_teacher_prefill"] > 0
    assert engine.last_metrics["max_unverified_tokens"] <= 6
