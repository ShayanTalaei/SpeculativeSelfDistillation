"""Spec acceptance tests, independent of CUDA and the legacy training stack."""

from dataclasses import replace

import pytest

from paired_vllm.core import PairConfig, PairedEngineCore, ProtocolError


@pytest.mark.parametrize("gap", [0, 5, 16, 4096])
@pytest.mark.parametrize("mode", ["pools", "reservations"])
def test_compact_history_and_bounded_slack(gap, mode):
    core = PairedEngineCore(PairConfig(capacity=16384, reservation_mode=mode))
    core.add("r", list(range(17 + gap)), list(range(17)), max_tokens=3)
    for k in range(3):
        step = core.plan()
        while not step.ready:
            complete(core, step, {})
            step = core.plan()
        r = core.requests["r"]
        for side, rows in enumerate((step.teacher, step.student)):
            assert (
                r.computed[side] + sum(len(row.tokens) for row in rows)
                == 17 + (gap if side == 0 else 0) + k
            )
        assert len(core.pools[1].pages["r"]) * 16 - (17 + k) < 16
        complete(core, step, {"r": (100 + k, "teacher" if k % 2 == 0 else "student")})
    assert core.requests["r"].committed == [100, 101, 102]
    assert core.pools[0].reserved == core.pools[1].reserved == 0


def complete(core, step, choices):
    core.prepared(step.digest, step.digest)
    core.executed(step.digest, step.digest)
    core.propose(step.digest, choices)
    events = core.commit()
    core.retire(step.digest, step.digest)
    return events


def test_long_prefill_holds_final_student_state_without_resampling():
    core = PairedEngineCore(PairConfig(capacity=256, token_budget=16))
    core.add("r", [1] * 49, [2] * 5, max_tokens=2)
    steps = []
    while not core.requests["r"].committed:
        step = core.plan()
        steps.append(step)
        complete(core, step, {"r": (9, "student")} if step.ready else {})
    assert len(steps) == 4
    assert [len(s.student[0].tokens) if s.student else 0 for s in steps] == [5, 0, 0, 0]
    assert core.requests["r"].computed == [49, 5]


def test_one_sided_pressure_never_partially_admits():
    core = PairedEngineCore(PairConfig(capacity=64))
    core.add("a", [1] * 40, [2], max_tokens=8)
    core.add("b", [1] * 32, [2], max_tokens=8)
    step = core.plan()
    assert [r.request_id for r in step.teacher] == ["a"]
    assert core.pools[0].reserved == 48
    assert core.pools[1].reserved == 16
    assert core.requests["b"].status == "waiting"


def test_second_reservation_failure_rolls_back(monkeypatch):
    core = PairedEngineCore(PairConfig(capacity=64))
    core.add("a", [1], [2], max_tokens=8)

    def fail(*args):
        raise RuntimeError("injected allocation failure")

    monkeypatch.setattr(core.pools[1], "reserve", fail)
    with pytest.raises(RuntimeError, match="injected"):
        core.plan()
    assert core.pools[0].reserved == core.pools[1].reserved == 0
    assert core.requests["a"].status == "waiting"


def test_mode_b_credit_cannot_be_lent():
    core = PairedEngineCore(PairConfig(capacity=128, reservation_mode="reservations"))
    core.add("a", [1] * 100, [2] * 50, max_tokens=1)
    step = core.plan()
    assert [pool.reserved for pool in core.pools] == [112, 112]
    assert [len(step.teacher[0].blocks), len(step.student[0].blocks)] == [7, 4]
    assert core.pools[1].unmapped_credit == 48


def test_no_successor_before_both_commit_acks():
    core = PairedEngineCore(PairConfig(capacity=64))
    core.add("a", [1], [2], max_tokens=2)
    step = core.plan()
    core.prepared(step.digest, step.digest)
    core.executed(step.digest, step.digest)
    core.propose(step.digest, {"a": (8, "teacher")})
    events = core.commit()
    assert core.commit() == events  # Retry cannot append twice.
    with pytest.raises(ProtocolError):
        core.plan()
    with pytest.raises(ProtocolError):
        core.retire(step.digest, "stale")
    assert core.requests["a"].committed == [8]
    core.retire(step.digest, step.digest)
    assert core.plan().teacher[0].tokens == (8,)


@pytest.mark.parametrize("after_commit", [False, True])
def test_epoch_recovery_uses_committed_history_only(after_commit):
    core = PairedEngineCore(PairConfig(capacity=128))
    core.add("a", [1, 2], [3], max_tokens=3)
    first = core.plan()
    complete(core, first, {"a": (7, "teacher")})
    step = core.plan()
    core.prepared(step.digest, step.digest)
    core.executed(step.digest, step.digest)
    core.propose(step.digest, {"a": (8, "student")})
    if after_commit:
        core.commit()
    core.recover()
    rebuilt = core.plan()
    assert rebuilt.epoch == step.epoch + 1
    assert rebuilt.teacher[0].incarnation == 1
    assert (
        rebuilt.teacher[0].tokens == (1, 2, 7, 8)
        if after_commit
        else rebuilt.teacher[0].tokens == (1, 2, 7)
    )
    with pytest.raises(ProtocolError):
        core.prepared(step.digest, step.digest)
    assert [e.source_model for e in core.requests["a"].events] == (
        ["teacher", "student"] if after_commit else ["teacher"]
    )


def test_cancellation_does_not_release_inflight_pages():
    core = PairedEngineCore(PairConfig(capacity=64))
    core.add("a", [1], [2], max_tokens=2)
    step = core.plan()
    core.cancel("a")
    assert core.pools[0].reserved > 0
    events = complete(core, step, {"a": (9, "teacher")})
    assert events == []
    assert core.pools[0].reserved == core.pools[1].reserved == 0


def test_selected_eos_only_and_provenance_even_for_identical_token():
    core = PairedEngineCore(PairConfig(capacity=64))
    core.add("a", [1], [2], max_tokens=3, stop_tokens=(9,))
    events = complete(core, core.plan(), {"a": (8, "student")})
    assert events[0].source_model == "student"
    assert core.requests["a"].status == "active"
    events = complete(core, core.plan(), {"a": (9, "teacher")})
    assert events[0].output_index == 1
    assert core.requests["a"].status == "finished"


def test_manifest_digest_covers_row_identity_and_role_specific_work():
    core = PairedEngineCore(PairConfig(capacity=128))
    core.add("a", [1] * 17, [2] * 15, max_tokens=2)
    core.add("b", [3] * 16, [4] * 17, max_tokens=2)
    step = core.plan()
    assert replace(step, teacher=step.teacher[::-1]).digest != step.digest
    assert replace(step, student=step.teacher).digest != step.digest


@pytest.mark.parametrize(
    "kwargs",
    [
        {"block_size": 32},
        {"block_size": 16, "strict_pair_tail": 16},
        {"conditioning_site": "input"},
        {"prefix_caching": True},
    ],
)
def test_unsupported_contracts_fail_at_configuration(kwargs):
    with pytest.raises(ValueError):
        PairConfig(**kwargs)


def test_invalid_proposal_is_atomic():
    core = PairedEngineCore(PairConfig(capacity=128))
    for r in ("a", "b"):
        core.add(r, [1], [2], max_tokens=2)
    step = core.plan()
    core.prepared(step.digest, step.digest)
    core.executed(step.digest, step.digest)
    with pytest.raises(ProtocolError):
        core.propose(step.digest, {"a": (9, "teacher"), "b": (7, "unknown")})
    assert all(not r.committed for r in core.requests.values())


def test_final_committed_token_replays_without_sampling_an_extra_token():
    core = PairedEngineCore(PairConfig(capacity=128, token_budget=16))
    core.add("a", [1] * 35, [2] * 5, max_tokens=1)
    while True:
        step = core.plan()
        if step.ready:
            break
        complete(core, step, {})
    core.prepared(step.digest, step.digest)
    core.executed(step.digest, step.digest)
    core.propose(step.digest, {"a": (8, "teacher")})
    core.commit()
    core.recover()
    replay_steps = 0
    while core.unfinished:
        step = core.plan()
        assert not step.ready
        assert complete(core, step, {}) == []
        replay_steps += 1
    assert replay_steps == 3
    assert core.requests["a"].committed == [8]
    assert core.requests["a"].computed == [36, 6]


def test_completed_prefill_is_not_resent_in_decode_manifest():
    core = PairedEngineCore(PairConfig(capacity=128))
    core.add("a", [1] * 35, [2] * 5, max_tokens=2)
    complete(core, core.plan(), {"a": (8, "teacher")})
    step = core.plan()
    assert step.teacher[0].prompt == step.student[0].prompt == ()
    assert step.teacher[0].prompt_length == 35
    assert step.student[0].prompt_length == 5
    assert step.teacher[0].tokens == step.student[0].tokens == (8,)


def test_decode_transmits_constant_size_commitment_and_recovery_full_history():
    import hashlib

    core = PairedEngineCore(PairConfig(capacity=256))
    core.add("a", [1] * 35, [2] * 5, max_tokens=20)
    digest = hashlib.sha256(b"").hexdigest()
    for i in range(10):
        manifest = core.plan()
        complete(core, manifest, {"a": (i + 10, "teacher")})
        digest = hashlib.sha256(
            bytes.fromhex(digest) + (i + 10).to_bytes(8, "little")
        ).hexdigest()
    manifest = core.plan()
    for row in (manifest.teacher[0], manifest.student[0]):
        assert row.history == ()
        assert row.history_digest == digest

        assert row.tokens == (19,)
    core.recover()
    manifest = core.plan()
    for row in (manifest.teacher[0], manifest.student[0]):
        assert row.history == tuple(range(10, 20))
        assert row.history_digest == digest


def test_block_updates_are_incremental_and_integrity_checked():
    core = PairedEngineCore(PairConfig(capacity=8192, token_budget=4096))
    core.add("a", [1] * 2048, [2] * 17, max_tokens=20)
    first = core.plan()
    assert first.teacher[0].block_start == 0
    assert len(first.teacher[0].blocks) == 128
    complete(core, first, {"a": (3, "teacher")})
    second = core.plan()
    assert second.teacher[0].block_start == 128
    assert len(second.teacher[0].blocks) == 1
    assert second.teacher[0].block_digest != first.teacher[0].block_digest
    complete(core, second, {"a": (4, "student")})
    third = core.plan()
    assert third.teacher[0].block_start == 129
    assert third.teacher[0].blocks == ()
    assert third.teacher[0].block_digest == second.teacher[0].block_digest
