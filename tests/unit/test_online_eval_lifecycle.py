import json
from types import SimpleNamespace

from data.core import DistillationRow
from evals.online_eval import EvalGroup, OnlineEvalCallback, exact_match_scorer
from ssd.checkpoint_utils import PruneCheckpointsBeforeSave
from transformers.trainer_utils import rotate_checkpoints


def test_best_evaluation_honors_disabled_saving(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "evals.online_eval.AutoTokenizer.from_pretrained", lambda _: None
    )
    monkeypatch.setattr("evals.online_eval.torch.cuda.empty_cache", lambda: None)
    callback = OnlineEvalCallback(
        groups=[
            EvalGroup(
                rows=[DistillationRow(question="q", answer="yes")],
                scorer=exact_match_scorer,
                metrics_prefix="eval",
            )
        ],
        vllm_instance=SimpleNamespace(wake_up=lambda: None, sleep=lambda **kw: None),
        model_name="fake",
        weight_sync_fn=lambda m: None,
        output_dir=str(tmp_path),
    )
    monkeypatch.setattr(callback, "_generate_for_group", lambda g: ["yes"])
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        metric_for_best_model="eval/mean_score",
        greater_is_better=True,
        save_strategy="no",
    )
    state = SimpleNamespace(
        global_step=3,
        epoch=0.1,
        is_world_process_zero=True,
        best_metric=None,
        best_model_checkpoint=None,
    )
    control = SimpleNamespace(should_training_stop=True, should_save=False)
    callback.on_step_end(args, state, control, model=object())
    callback.on_train_end(args, state, control)
    assert state.best_metric == 1.0
    assert not control.should_save
    assert state.best_model_checkpoint is None
    assert not list(tmp_path.glob("checkpoint-*"))


def test_final_eval_and_best_survive_both_checkpoint_rotations(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "evals.online_eval.AutoTokenizer.from_pretrained", lambda _: None
    )
    monkeypatch.setattr("evals.online_eval.torch.cuda.empty_cache", lambda: None)
    group = EvalGroup(
        rows=[DistillationRow(question="q", answer="yes")],
        scorer=exact_match_scorer,
        metrics_prefix="eval",
    )
    engine = SimpleNamespace(wake_up=lambda: None, sleep=lambda **kw: None)
    callback = OnlineEvalCallback(
        groups=[group],
        vllm_instance=engine,
        model_name="fake",
        weight_sync_fn=lambda m: None,
        output_dir=str(tmp_path),
    )
    calls = []
    monkeypatch.setattr(
        callback, "_generate_for_group", lambda g: calls.append(1) or ["yes"]
    )
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        metric_for_best_model="eval/mean_score",
        greater_is_better=True,
        save_total_limit=2,
        should_save=True,
    )
    state = SimpleNamespace(
        global_step=300,
        epoch=1.8,
        is_world_process_zero=True,
        best_metric=None,
        best_model_checkpoint=None,
    )
    control = SimpleNamespace(should_training_stop=False, should_save=False)
    callback.on_step_end(args, state, control, model=object())
    assert control.should_save and state.best_global_step == 300
    for step in (300, 325):
        path = tmp_path / f"checkpoint-{step}"
        path.mkdir()
        (path / "trainer_state.json").write_text("{}")
    state.global_step = 325
    callback.on_step_end(args, state, control, model=object())
    assert len(calls) == 1
    state.global_step = 334
    control.should_training_stop = True
    callback.on_step_end(args, state, control, model=object())
    callback.evaluate_now(args, state, control, model=object())
    assert len(calls) == 2  # off-cadence final evaluation, without duplicates
    callback.on_train_end(args, state, control)
    assert '"step": 334' in (tmp_path / "eval_online.jsonl").read_text()
    timings = [
        json.loads(row)
        for row in (tmp_path / "io-timings.jsonl").read_text().splitlines()
    ]
    cycles = [row for row in timings if row["kind"] == "evaluation_cycle"]
    timings = [row for row in timings if row["kind"] == "evaluation"]
    assert [row["step"] for row in cycles] == [300, 334]
    for cycle, detail in zip(cycles, timings):
        assert cycle["started_at_unix"] <= detail["started_at_unix"]
        assert cycle["finished_at_unix"] >= detail["finished_at_unix"]
    assert [row["step"] for row in timings] == [300, 334]
    for row in timings:
        assert row["kind"] == "evaluation"
        assert row["seconds"] >= sum(
            row[key]
            for key in (
                "wake_seconds",
                "sync_seconds",
                "generation_seconds",
                "scoring_seconds",
                "logging_seconds",
                "sleep_seconds",
            )
        )
    PruneCheckpointsBeforeSave(str(tmp_path)).on_step_end(args, state, control)
    assert (tmp_path / "checkpoint-300").exists()
    (tmp_path / "checkpoint-334").mkdir()
    rotate_checkpoints(
        output_dir=str(tmp_path),
        save_total_limit=2,
        best_model_checkpoint=state.best_model_checkpoint,
    )
    assert {p.name for p in tmp_path.glob("checkpoint-*")} == {
        "checkpoint-300",
        "checkpoint-334",
    }


def test_best_checkpoint_is_protected_even_under_disk_pressure(monkeypatch, tmp_path):
    best = tmp_path / "checkpoint-300"
    best.mkdir()
    (best / "weights").write_bytes(b"weights")
    monkeypatch.setattr(
        "ssd.checkpoint_utils.shutil.disk_usage", lambda _: SimpleNamespace(free=0)
    )
    PruneCheckpointsBeforeSave(str(tmp_path)).on_step_end(
        None,
        SimpleNamespace(is_world_process_zero=True, best_model_checkpoint=str(best)),
        SimpleNamespace(should_save=True),
    )
    assert best.exists()


def test_provenance_padding_and_labels():
    import torch
    from ssd.distil_trainer import _encode_provenance

    actual = _encode_provenance(
        [["student", "teacher", "fused"], ["teacher"], []], 4, "cpu"
    )
    assert actual.dtype == torch.long
    assert actual.tolist() == [[1, 2, 3, 0], [2, 0, 0, 0], [0, 0, 0, 0]]
