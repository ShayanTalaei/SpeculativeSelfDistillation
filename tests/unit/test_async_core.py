import pytest

from paired_vllm.async_core import (
    Draft,
    Ledger,
    RolloutRequest,
    admission_batches,
    uniform,
)


def ledger(stops=()):
    return Ledger(RolloutRequest("r", [1], [1], 12, stops))


def test_correction_invalidates_later_inflight_work():
    row = ledger()
    first = Draft("r", 0, 0, [3, 4, 5, 6])
    second = Draft("r", 0, 4, [7, 8, 9, 10])
    row.append(first)
    row.append(second)
    assert row.apply(first, [3, 99], ["student", "teacher"])
    assert row.tokens == row.drafted == [3, 99]
    assert row.version == 1
    assert row.discarded == 6
    assert not row.append(Draft("r", 0, 8, [11, 12]))
    assert not row.apply(second, second.tokens, ["student"] * 4)
    replacement = Draft("r", 1, 2, [20, 21])
    row.append(replacement)
    row.apply(replacement, replacement.tokens, ["student", "student"])
    assert row.tokens == [3, 99, 20, 21]


def test_teacher_same_token_preserves_pending_drafts_and_provenance():
    row = ledger()
    draft = Draft("r", 0, 0, [3, 4])
    row.append(draft)
    assert not row.apply(draft, [3, 4], ["teacher", "student"])
    assert row.version == 0
    assert row.sources == ["teacher", "student"]


def test_tentative_eos_can_be_replaced():
    row = ledger((2,))
    draft = Draft("r", 0, 0, [3, 2])
    row.append(draft)
    assert not row.done
    assert row.apply(draft, [3, 9], ["student", "teacher"])
    assert not row.done


def test_verified_eos_discards_suffix():
    row = ledger((2,))
    draft = Draft("r", 0, 0, [3, 4, 5])
    row.append(draft)
    row.apply(draft, [3, 2], ["student", "teacher"])
    assert row.done
    assert row.tokens == [3, 2]


def test_noncontiguous_verification_fails():
    row = ledger()
    with pytest.raises(ValueError, match="frontier"):
        row.apply(Draft("r", 0, 1, [4]), [4], ["student"])


def test_position_randomness_is_repeatable_and_request_specific():
    assert uniform(42, "r", 3) == uniform(42, "r", 3)
    assert len({uniform(42, r, i) for r in ("r", "s") for i in range(100)}) == 200
    assert all(0 < uniform(42, "r", i) < 1 for i in range(1000))


def test_admission_obeys_each_gpu_pool_and_preserves_order():
    rows = [RolloutRequest(str(i), [1] * 65, [1] * 15, 17) for i in range(19)]
    batches = admission_batches(rows, 8, {0: 256, 1: 128}, 16)
    assert [r for batch in batches for r in batch] == rows
    assert [len(batch) for batch in batches] == [2] * 9 + [1]


def test_oversized_request_is_rejected_before_any_admission():
    rows = [
        RolloutRequest("small", [1], [1], 1),
        RolloutRequest("large", [1] * 256, [1], 1),
    ]
    with pytest.raises(ValueError, match="large.*KV capacity"):
        admission_batches(rows, 8, {0: 256, 1: 256}, 16)
