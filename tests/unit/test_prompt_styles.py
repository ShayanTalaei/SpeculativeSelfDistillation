"""Tests for ``to_distil_dataset`` rendering — V3_QA_BEFORE format with the
four boolean / position knobs."""

from __future__ import annotations

import pytest

from data.core import (
    CONTEXT_LEAD_IN,
    STUDENT_SYSTEM,
    TEACHER_SYSTEM,
    DistillationRow,
    to_distil_dataset,
)

Q = "What year was Hearthwell & Sons founded?"
A = "1978; Fitchburg, Massachusetts."


def _row(**overrides) -> DistillationRow:
    return DistillationRow(question=Q, answer=A, **overrides)


def _ctx_block(teacher_user: str) -> str:
    """Text inside <context>...</context>."""
    return teacher_user.split("<context>\n", 1)[1].split("\n</context>", 1)[0]


# ── Default V3_QA_BEFORE rendering ───────────────────────────────


def test_v3_qa_before_exact_render():
    ds = to_distil_dataset([_row()])
    student, teacher = ds[0]["prompt"], ds[0]["teacher_prompt"]

    assert student == [
        {"role": "system", "content": STUDENT_SYSTEM},
        {"role": "user",   "content": f"Question: {Q}"},
    ]
    assert teacher[0] == {"role": "system", "content": TEACHER_SYSTEM}
    teacher_user = teacher[1]["content"]
    assert teacher_user.startswith(f"{CONTEXT_LEAD_IN}\n<context>\n")
    assert _ctx_block(teacher_user).split("\n") == [f"Question: {Q}", f"Answer: {A}"]
    assert teacher_user.endswith(f"\n\nQuestion: {Q}")


def test_universal_output_schema():
    ds = to_distil_dataset([_row()])
    assert set(ds.column_names) == {"prompt", "teacher_prompt", "answer", "chat_template_kwargs"}
    assert ds[0]["answer"] == A


# ── Position knob ────────────────────────────────────────────────


def test_position_after_swaps_order():
    ds = to_distil_dataset([_row()], context_position="after")
    teacher_user = ds[0]["teacher_prompt"][1]["content"]
    assert teacher_user.startswith(f"Question: {Q}")
    assert teacher_user.rstrip().endswith("</context>")


def test_invalid_position_raises():
    with pytest.raises(ValueError, match="context_position"):
        to_distil_dataset([_row()], context_position="middle")


# ── Three boolean knobs ──────────────────────────────────────────


def test_documents_only_inside_context():
    ds = to_distil_dataset(
        [_row(documents="Hearthwell & Sons is in Fitchburg.")],
        context_documents=True, context_question=False, context_answer=False,
    )
    assert _ctx_block(ds[0]["teacher_prompt"][1]["content"]) == "Hearthwell & Sons is in Fitchburg."


def test_answer_only_inside_context():
    ds = to_distil_dataset(
        [_row()], context_documents=False, context_question=False, context_answer=True,
    )
    assert _ctx_block(ds[0]["teacher_prompt"][1]["content"]) == f"Answer: {A}"


def test_full_doc_q_and_a_in_order():
    ds = to_distil_dataset(
        [_row(documents="DOC")],
        context_documents=True, context_question=True, context_answer=True,
    )
    block = _ctx_block(ds[0]["teacher_prompt"][1]["content"])
    assert block.split("\n") == ["DOC", f"Question: {Q}", f"Answer: {A}"]


def test_all_off_falls_back_to_student_shape():
    ds = to_distil_dataset(
        [_row()], context_documents=False, context_question=False, context_answer=False,
    )
    teacher_user = ds[0]["teacher_prompt"][1]["content"]
    assert "<context>" not in teacher_user
    assert CONTEXT_LEAD_IN not in teacher_user
    assert teacher_user == ds[0]["prompt"][1]["content"]


def test_answer_none_omits_answer_line():
    ds = to_distil_dataset(
        [DistillationRow(question="Q?", answer=None)],
        context_question=True, context_answer=True,
    )
    block = _ctx_block(ds[0]["teacher_prompt"][1]["content"])
    assert block == "Question: Q?"
    assert "None" not in block


# ── Per-row system override ──────────────────────────────────────


def test_row_system_prompt_overrides_canonical():
    custom = "You are a doctor."
    ds = to_distil_dataset([_row(system_prompt=custom)])
    assert ds[0]["prompt"][0]["content"] == custom
    assert ds[0]["teacher_prompt"][0]["content"] == custom
