"""Phase instrumentation must preserve updates and separate active time from I/O."""

import json
import time
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from ssd.experiment_timing import ExperimentTiming
from ssd.io_timing import record_io_timing


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(16, 16)

    def forward(self, inputs):
        return self.head(inputs)

    def get_output_embeddings(self):
        return self.head


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_timing_preserves_updates_and_excludes_between_step_work(
    tmp_path, monkeypatch, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "cpu":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(41)
    model = TinyModel().to(device)
    baseline = TinyModel().to(device)
    baseline.load_state_dict(model.state_dict())
    teacher = TinyModel().to(device).requires_grad_(False)
    teacher_initial = {k: v.clone() for k, v in teacher.state_dict().items()}
    inputs = torch.randn(8, 16, device=device)
    trainer = SimpleNamespace(
        model=model,
        ref_model=teacher,
        accelerator=SimpleNamespace(backward=lambda loss: loss.backward()),
        _metrics={"train": defaultdict(list)},
        _cum_num_tokens=0,
        compute_loss=lambda student, x: (student(x) - teacher(x)).square().mean(),
        _clip_grad_norm=lambda student: torch.nn.utils.clip_grad_norm_(
            student.parameters(), 1.0
        ),
    )
    original_forward = model.forward
    timing = ExperimentTiming(trainer)
    args, state = (
        SimpleNamespace(output_dir=str(tmp_path)),
        SimpleNamespace(global_step=0),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.01)
    for _ in range(2):
        # Same math and inputs without the wrappers.
        (baseline(inputs) - teacher(inputs)).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1.0)
        baseline_optimizer.step()
        baseline_optimizer.zero_grad()
        timing.on_step_begin(args, state, None)
        trainer.accelerator.backward(trainer.compute_loss(model, inputs))
        trainer._clip_grad_norm(model)
        timing.on_pre_optimizer_step(args, state, None)
        optimizer.step()
        timing.on_optimizer_step(args, state, None)
        optimizer.zero_grad()
        state.global_step += 1
        trainer._cum_num_tokens += 8
        timing.on_step_end(args, state, None)
        time.sleep(0.03)  # Stands for evaluation/checkpoint work between steps.
    timing.on_train_end(args, state, None)
    assert model.forward == original_forward
    for key, value in model.state_dict().items():
        assert torch.equal(value, baseline.state_dict()[key])
    for key, value in teacher.state_dict().items():
        assert torch.equal(value, teacher_initial[key])
    rows = [
        json.loads(line)
        for line in (tmp_path / "step-timings.jsonl").read_text().splitlines()
    ]
    assert state.active_training_seconds == pytest.approx(
        sum(r["step_seconds"] for r in rows)
    )
    assert rows[1]["between_steps_seconds"] >= 0.03
    for row in rows:
        for phase in [
            "student_forward",
            "teacher_forward",
            "loss_total",
            "backward",
            "gradient_clipping",
            "optimizer",
        ]:
            assert row[f"{phase}_calls"] == 1
            assert row[f"{phase}_wall_seconds"] > 0
            if device == "cuda":
                assert row[f"{phase}_cuda_seconds"] > 0
        assert (
            "teacher_projection_calls" not in row
        )  # No double-counting the teacher head.
        assert row["completion_tokens"] == 8
        assert row["finished_at_unix"] >= row["started_at_unix"]


def test_explicit_io_intervals_survive_delayed_logging(tmp_path):
    row = record_io_timing(
        tmp_path,
        3,
        "checkpoint_publication",
        started_at_unix=10.0,
        finished_at_unix=12.0,
        seconds=2.0,
    )
    assert row["started_at_unix"] == 10.0
    assert row["finished_at_unix"] == row["timestamp"] == 12.0
