"""Direct GPU weight loading preserves fresh weights and evaluation residency."""

from types import SimpleNamespace

import pytest
import torch
from data.core import DistillationRow
from distil_trainer import DistilTrainer
from evals.online_eval import EvalGroup, OnlineEvalCallback, exact_match_scorer
from ssd.weight_sync_audit import verify_colocated_weights


@pytest.mark.parametrize("batched", [False, True])
def test_colocated_sync_copies_current_weights_and_invalidates_prefixes(batched):
    source = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    calls, resets = [], []

    def load(weights):
        weights = list(weights)
        calls.append(len(weights))
        for name, value in weights:
            assert value.data_ptr() == dict(source.named_parameters())[name].data_ptr()
            with torch.no_grad():
                dict(target.named_parameters())[name].copy_(value)

    engine = SimpleNamespace(
        model_executor=SimpleNamespace(
            driver_worker=SimpleNamespace(
                model_runner=SimpleNamespace(model=SimpleNamespace(load_weights=load))
            )
        )
    )
    trainer = SimpleNamespace(
        model=source,
        generate_from_teacher=False,
        is_fsdp_enabled=False,
        accelerator=SimpleNamespace(state=SimpleNamespace(deepspeed_plugin=None)),
        args=SimpleNamespace(
            vllm_batch_weight_sync=batched, vllm_verify_weight_sync=False
        ),
        vllm_mode="colocate",
        llm=SimpleNamespace(
            llm_engine=engine, reset_prefix_cache=lambda: resets.append(1)
        ),
        _fix_param_name_to_vllm=lambda name: name,
    )
    for _ in range(2):
        with torch.no_grad():
            source.weight.add_(1)
        DistilTrainer._move_model_to_vllm.__wrapped__(trainer)
        assert all(
            torch.equal(a, b) for a, b in zip(source.parameters(), target.parameters())
        )
    assert calls == ([2, 2] if batched else [1, 1, 1, 1])
    assert resets == [1, 1]


def test_resident_evaluation_never_sleeps_and_refreshes_each_step(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "evals.online_eval.AutoTokenizer.from_pretrained", lambda _: None
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Resident evaluation must not sleep, wake, or empty cache")

    monkeypatch.setattr(torch.cuda, "empty_cache", forbidden)
    syncs = []
    callback = OnlineEvalCallback(
        groups=[
            EvalGroup(
                rows=[DistillationRow(question="q", answer="yes")],
                scorer=exact_match_scorer,
                metrics_prefix="eval",
            )
        ],
        vllm_instance=SimpleNamespace(wake_up=forbidden, sleep=forbidden),
        model_name="fake",
        manage_sleep=False,
        weight_sync_fn=lambda model: syncs.append(model),
        output_dir=str(tmp_path),
    )
    monkeypatch.setattr(callback, "_generate_for_group", lambda _: ["yes"])
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        metric_for_best_model="eval/mean_score",
        greater_is_better=True,
        save_strategy="no",
    )
    state = SimpleNamespace(
        global_step=25,
        epoch=0.1,
        is_world_process_zero=True,
        best_metric=None,
        best_model_checkpoint=None,
    )
    control = SimpleNamespace(should_save=False)
    model = object()
    callback.evaluate_now(args, state, control, model=model)
    state.global_step = 50
    callback.evaluate_now(args, state, control, model=model)
    callback.on_train_end(args, state, control)
    assert syncs == [model, model]
    assert state.best_metric == 1.0 and not control.should_save


def test_offpolicy_evaluation_batches_live_parameter_views():
    source = torch.nn.Linear(3, 2)
    calls = []
    callback = OnlineEvalCallback.__new__(OnlineEvalCallback)
    callback.weight_sync_fn = None
    callback.batch_weight_sync = True
    callback.llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            reset_prefix_cache=lambda: None,
            model_executor=SimpleNamespace(
                driver_worker=SimpleNamespace(
                    model_runner=SimpleNamespace(
                        model=SimpleNamespace(
                            load_weights=lambda weights: calls.append(list(weights))
                        )
                    )
                )
            ),
        )
    )
    callback._sync_weights(source)
    assert len(calls) == 1 and len(calls[0]) == 2
    for name, value in calls[0]:
        assert value.data_ptr() == dict(source.named_parameters())[name].data_ptr()


def test_audit_checks_packed_weights_tied_aliases_and_detects_stale_updates():
    source = torch.nn.Module()
    source.block = torch.nn.ModuleDict(
        {
            name: torch.nn.Linear(2, 2, bias=False)
            for name in ("q_proj", "k_proj", "v_proj")
        }
    )
    source.alias = source.block.q_proj
    target = torch.nn.Module()
    target.block = torch.nn.ModuleDict({"qkv_proj": torch.nn.Linear(2, 6, bias=False)})
    target.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    with torch.no_grad():
        target.block.qkv_proj.weight.copy_(
            torch.cat(
                [source.block[name].weight for name in ("q_proj", "k_proj", "v_proj")]
            )
        )
    assert verify_colocated_weights(source, target) == {
        "source_tensors": 4,
        "inference_tensors": 1,
        "status": "passed",
    }
    with torch.no_grad():
        source.block.k_proj.weight.add_(1)
    with pytest.raises(ValueError, match="differ from current student"):
        verify_colocated_weights(source, target)
