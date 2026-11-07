"""Unit tests for EvalGroup scoring and per-group metric aggregation.

These tests cover only the pure-Python score->metrics code path. No
generation — we feed canned responses directly to the callback's
aggregation helper.
"""

from data.core import DistillationRow
from evals.online_eval import (
    EvalGroup,
    aggregate_group_metrics,
    exact_match_scorer,
)


def _recall_scorer(question, reference, response, row_metadata):
    """Fake recall scorer: 1.0 if every reference name is in the response."""
    names = reference.split(",")
    found = sum(1 for n in names if n.strip() in response)
    return {
        "score": found / len(names),
        "n_found": found,
        "n_total": len(names),
    }


def test_exact_match_flat_group():
    rows = [
        DistillationRow(question="q1", answer="yes"),
        DistillationRow(question="q2", answer="no"),
    ]
    responses = ["yes it is", "definitely yes"]
    group = EvalGroup(
        rows=rows,
        scorer=exact_match_scorer,
        metrics_prefix="eval",
    )
    metrics, records = aggregate_group_metrics(group, responses)
    assert metrics["eval/mean_score"] == 0.5
    assert metrics["eval/n_total"] == 2
    assert len(records) == 2


def test_group_by_category_and_split():
    rows = [
        DistillationRow(question="q1", answer="A",
                        metadata={"category": "mcq", "split": "in_dist"}),
        DistillationRow(question="q2", answer="A",
                        metadata={"category": "mcq", "split": "ood"}),
        DistillationRow(question="q3", answer="B",
                        metadata={"category": "tf", "split": "in_dist"}),
    ]
    responses = ["A", "A", "B"]
    group = EvalGroup(
        rows=rows,
        scorer=exact_match_scorer,
        metrics_prefix="generalization",
        group_by=["category", "split"],
    )
    metrics, _ = aggregate_group_metrics(group, responses)
    assert metrics["generalization/mean_score"] == 1.0
    assert metrics["generalization/mcq/mean_score"] == 1.0
    assert metrics["generalization/tf/mean_score"] == 1.0
    assert metrics["generalization/in_dist/mean_score"] == 1.0
    assert metrics["generalization/ood/mean_score"] == 1.0


def test_nested_group_by_combines_keys():
    rows = [
        DistillationRow(question="q1", answer="A",
                        metadata={"category": "mcq", "split": "in_dist"}),
        DistillationRow(question="q2", answer="A",
                        metadata={"category": "mcq", "split": "ood"}),
    ]
    responses = ["A", "wrong"]
    group = EvalGroup(
        rows=rows,
        scorer=exact_match_scorer,
        metrics_prefix="generalization",
        group_by=["split", "category"],
    )
    metrics, _ = aggregate_group_metrics(group, responses)
    assert metrics["generalization/in_dist/mcq/mean_score"] == 1.0
    assert metrics["generalization/ood/mcq/mean_score"] == 0.0


def test_scorer_tags_are_propagated_to_records():
    rows = [
        DistillationRow(question="q1", answer="Alice,Bob"),
    ]
    responses = ["Alice but not the other one"]
    group = EvalGroup(
        rows=rows,
        scorer=_recall_scorer,
        metrics_prefix="recall",
    )
    metrics, records = aggregate_group_metrics(group, responses)
    assert metrics["recall/mean_score"] == 0.5
    assert records[0]["n_found"] == 1
    assert records[0]["n_total"] == 2


def test_eval_logging_after_profiler_steps_preserves_wandb_and_local_results(
    monkeypatch, tmp_path,
):
    """Eval at optimizer step 50 must survive W&B already being at step 1800."""
    import json
    import sys
    from types import SimpleNamespace

    from evals.online_eval import OnlineEvalCallback

    history, definitions = [], []

    def log(data, step=None):
        # Model W&B's monotonic internal step counter, advanced by profiling.
        if step is None or step >= 1800 + len(history):
            history.append(data)

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        run=object(), log=log,
        define_metric=lambda name, **kwargs: definitions.append((name, kwargs)),
    ))
    callback = OnlineEvalCallback.__new__(OnlineEvalCallback)
    callback.output_dir = str(tmp_path)
    callback._file = callback._rollouts_file = None
    state = SimpleNamespace(global_step=50, epoch=0.3)
    try:
        for prefix in ("eval", "heldout"):
            group = EvalGroup(rows=[], scorer=exact_match_scorer, metrics_prefix=prefix)
            callback._log_group(state, group, {
                f"{prefix}/mean_score": 0.5, f"{prefix}/n_total": 2,
            }, [{"question": "q", "reference": "A", "response": "A", "score": 1}])
    finally:
        callback.on_train_end(None, state, None)

    assert len(history) == 2
    assert all(row["train/global_step"] == 50 for row in history)
    assert history[0]["eval/mean_score"] == 0.5
    assert definitions == [
        (f"{prefix}/*", {"step_metric": "train/global_step"})
        for prefix in ("eval", "heldout")
    ]
    summaries = [json.loads(line) for line in (tmp_path / "eval_online.jsonl").read_text().splitlines()]
    rollouts = [json.loads(line) for line in (tmp_path / "eval_rollouts.jsonl").read_text().splitlines()]
    assert [row["step"] for row in summaries] == [50, 50]
    assert [row["prefix"] for row in summaries] == ["eval", "heldout"]
    assert summaries[0]["eval/mean_score"] == 0.5
    assert len(rollouts) == 2 and all(row["score"] == 1 for row in rollouts)
