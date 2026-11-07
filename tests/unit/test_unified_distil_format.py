"""Unified distil-format dataset contract.

to_distil_dataset's output must be usable by:
  - on-policy trainer (reads prompt, teacher_prompt; ignores answer)
  - off-policy FKL    (reads prompt, teacher_prompt, answer)
  - off-policy SFT    (reads prompt + answer, builds messages on the fly)

These tests cover the dataset shape and the tokenize_for_fkl contract.
No GPU / no training — just schema and string assembly.
"""

from data.core import DistillationRow, to_distil_dataset


def test_to_distil_dataset_has_answer_column():
    rows = [DistillationRow(question="Q", documents="D", answer="A")]
    ds = to_distil_dataset(rows)
    assert set(ds.column_names) == {
        "prompt", "teacher_prompt", "answer", "chat_template_kwargs"
    }, ds.column_names
    assert ds[0]["answer"] == "A"


def test_to_distil_dataset_answer_passes_through_none_cleanly():
    # On-policy sources that don't know an answer yet: should still build.
    rows = [DistillationRow(question="Q", documents="D", answer=None)]
    ds = to_distil_dataset(rows)
    assert ds[0]["answer"] is None


def test_tokenize_for_fkl_reads_distil_format():
    """ssd.kl_utils.tokenize_for_fkl consumes a distil-format row."""
    from transformers import AutoTokenizer
    from ssd.kl_utils import tokenize_for_fkl

    # Skip if the tokenizer isn't already available (network-dependent test).
    try:
        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-4B-Instruct-2507", local_files_only=True,
        )
    except (OSError, ValueError):
        print("  SKIP test_tokenize_for_fkl_reads_distil_format "
              "(Qwen/Qwen3-4B-Instruct-2507 not cached locally)")
        return

    rows = [DistillationRow(question="Q", documents="D", answer="A")]
    ds = to_distil_dataset(rows)
    out = tokenize_for_fkl(ds[0], tok, max_student_len=1024, max_teacher_len=1024)
    assert set(out.keys()) == {
        "student_ids", "teacher_ids",
        "completion_start_student", "completion_start_teacher",
        "num_completion_tokens",
    }
    assert 0 < out["completion_start_student"] < len(out["student_ids"])
    assert 0 < out["completion_start_teacher"] < len(out["teacher_ids"])
    assert out["num_completion_tokens"] > 0
