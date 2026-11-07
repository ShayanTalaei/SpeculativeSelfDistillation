"""Core data types and conversion functions for distillation training.

Every benchmark produces ``list[DistillationRow]``. ``to_distil_dataset`` turns
those into the universal HuggingFace Dataset consumed by all training paths
(on-policy, off-policy FKL, SFT).

Teacher-prompt format (V3_QA_BEFORE)
------------------------------------
Student and teacher share the same user-turn shape ``Question: {question}``.
The teacher additionally has a ``<context>...</context>`` block built from any
subset of {documents, question, answer}, placed before or after the user-turn
question. Three booleans + one position knob give every variant we tested:

    context_documents   include row.documents inside <context>
    context_question    include "Question: {q}" inside <context>
    context_answer      include "Answer: {a}" inside <context>
    context_position    "before" | "after"  — relative to the user-turn question

Default ``(documents=False, question=True, answer=True, position="before")``
is the empirical winner.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from datasets import Dataset as HFDataset
from datasets import load_from_disk


@dataclass
class DistillationRow:
    """Universal data format for all distillation benchmarks.

    Attributes:
        question: What the student sees as input (always required).
        documents: Privileged context the teacher gets but the student does not.
        answer: Gold completion text. Required for off-policy/SFT, ignored
            on-policy.
        system_prompt: Optional per-row system prompt. If set, replaces the
            canonical system prompt for both student and teacher.
        metadata: Per-row provenance dict (doc_id, chunk_id, ...). Not passed
            to trainers — collected into metadata.json for analysis.
    """
    question: str
    documents: str | None = None
    answer: str | None = None
    system_prompt: str | None = None
    metadata: dict | None = None
    eval_type: str | None = None


# Canonical system prompts. The only per-side difference is the extra sentence
# in the teacher system that announces the upcoming knowledge context.
STUDENT_SYSTEM = "You are a helpful assistant. Answer the user's question."
TEACHER_SYSTEM = (
    "You are a helpful assistant. Answer the user's question.\n\n"
    "You will be provided some knowledge context relevant to the question. "
    "Use it when answering."
)
CONTEXT_LEAD_IN = "Here is context we know about the question:"

CONTEXT_KWARGS = (
    "context_documents", "context_question", "context_answer", "context_position",
)


def _render_row(
    row: DistillationRow,
    *,
    context_documents: bool,
    context_question: bool,
    context_answer: bool,
    context_position: str,
) -> tuple[list[dict], list[dict]]:
    """Render one row into ``(student_msgs, teacher_msgs)``."""
    if context_position not in ("before", "after"):
        raise ValueError(
            f"context_position must be 'before' or 'after', got {context_position!r}"
        )

    student_system = teacher_system = row.system_prompt
    if student_system is None:
        student_system, teacher_system = STUDENT_SYSTEM, TEACHER_SYSTEM

    student_user = f"Question: {row.question}"

    parts: list[str] = []
    if context_documents and row.documents:
        parts.append(row.documents)
    if context_question:
        parts.append(f"Question: {row.question}")
    if context_answer and row.answer is not None:
        parts.append(f"Answer: {row.answer}")

    if not parts:
        teacher_user = student_user
    else:
        ctx_chunk = (
            f"{CONTEXT_LEAD_IN}\n"
            f"<context>\n" + "\n".join(parts) + "\n</context>"
        )
        if context_position == "before":
            teacher_user = f"{ctx_chunk}\n\n{student_user}"
        else:
            teacher_user = f"{student_user}\n\n{ctx_chunk}"

    return (
        [{"role": "system", "content": student_system},
         {"role": "user",   "content": student_user}],
        [{"role": "system", "content": teacher_system},
         {"role": "user",   "content": teacher_user}],
    )


def to_distil_dataset(
    rows: list[DistillationRow],
    *,
    enable_thinking: bool = False,
    context_documents: bool = False,
    context_question: bool = True,
    context_answer: bool = True,
    context_position: str = "before",
) -> HFDataset:
    """Convert ``DistillationRow``s to the universal distil HFDataset.

    Output schema (consumed by every training path):
        - prompt:               list[dict]  student chat messages
        - teacher_prompt:       list[dict]  teacher chat messages
        - answer:               str | None  completion target (None if on-policy-only)
        - chat_template_kwargs: dict        passed to apply_chat_template at training time
    """
    chat_kwargs = {"enable_thinking": enable_thinking}
    hf_rows = []
    for row in rows:
        student_msgs, teacher_msgs = _render_row(
            row,
            context_documents=context_documents,
            context_question=context_question,
            context_answer=context_answer,
            context_position=context_position,
        )
        hf_rows.append({
            "prompt":               student_msgs,
            "teacher_prompt":       teacher_msgs,
            "answer":               row.answer,
            "chat_template_kwargs": chat_kwargs,
        })
    return HFDataset.from_list(hf_rows)


# ── Save / load built datasets ───────────────────────────────────


def save_built_dataset(
    dataset: HFDataset,
    output_dir: str,
    metadata: dict,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved {len(dataset)} rows to {output_dir}/", flush=True)
    print(f"  columns: {dataset.column_names}", flush=True)
    print(f"  metadata: {meta_path}", flush=True)


def load_built_dataset(path: str) -> tuple[HFDataset, dict]:
    dataset = load_from_disk(path)
    meta_path = os.path.join(path, "metadata.json")
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
    return dataset, metadata
