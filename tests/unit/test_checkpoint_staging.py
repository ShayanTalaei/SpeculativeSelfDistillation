import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from ssd.checkpoint_staging import (
    PENDING,
    CheckpointPublisher,
    publish_checkpoint,
    staging_directory,
)
from ssd.checkpoint_utils import PruneCheckpointsBeforeSave, latest_checkpoint


def test_publication_exposes_state_last_and_preserves_contents(monkeypatch, tmp_path):
    from ssd import checkpoint_staging

    source, target = tmp_path / "source", tmp_path / "checkpoint-1"
    source.mkdir()
    for name in (
        "model.safetensors",
        "optimizer.pt",
        "teacher_state.pt",
        "trainer_state.json",
    ):
        (source / name).write_text(name)
    copied = []
    original = checkpoint_staging.shutil.copyfile

    def copy(src, dst):
        assert (target / PENDING).exists()
        assert latest_checkpoint(str(tmp_path)) is None
        copied.append(Path(src).name)
        return original(src, dst)

    monkeypatch.setattr(checkpoint_staging.shutil, "copyfile", copy)
    publish_checkpoint(source, target)
    assert copied[-1] == "trainer_state.json"
    assert not (target / PENDING).exists()
    assert latest_checkpoint(str(tmp_path)) == str(target)
    for file in source.iterdir():
        assert (target / file.name).read_bytes() == file.read_bytes()


def test_failed_copy_stays_unresumable_and_snapshot_is_recoverable(
    monkeypatch, tmp_path
):
    from ssd import checkpoint_staging

    stage = tmp_path / "stage"
    (stage / "checkpoint-2").mkdir(parents=True)
    (stage / "checkpoint-2/trainer_state.json").write_text("{}")
    target = tmp_path / "run/checkpoint-2"
    monkeypatch.setattr(
        checkpoint_staging.shutil,
        "copyfile",
        lambda *_: (_ for _ in ()).throw(OSError("storage failed")),
    )
    publisher = CheckpointPublisher(tmp_path / "run")
    publisher.submit(stage, target, 2)
    with pytest.raises(RuntimeError, match="snapshot retained"):
        publisher.close()
    assert (stage / "checkpoint-2/trainer_state.json").exists()
    assert (target / PENDING).exists()
    assert latest_checkpoint(str(target.parent)) is None


def test_pending_publication_is_not_pruned(tmp_path):
    pending = tmp_path / "checkpoint-1"
    pending.mkdir()
    (pending / PENDING).touch()
    (pending / "weights").write_bytes(b"in flight")
    (tmp_path / "checkpoint-2").mkdir()
    PruneCheckpointsBeforeSave(str(tmp_path)).on_step_end(
        None,
        SimpleNamespace(is_world_process_zero=True, best_model_checkpoint=None),
        SimpleNamespace(should_save=True),
    )
    assert pending.exists()


def test_resume_replaces_only_incomplete_publication(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "trainer_state.json").write_text("{}")
    target = tmp_path / "checkpoint-1"
    target.mkdir()
    (target / PENDING).touch()
    (target / "partial").write_bytes(b"old partial snapshot")
    publish_checkpoint(source, target)
    archived = list(tmp_path.glob(".incomplete-checkpoint-1-*"))
    assert len(archived) == 1 and (archived[0] / "partial").exists()
    assert not (target / "partial").exists()
    with pytest.raises(FileExistsError):
        publish_checkpoint(source, target)


def test_explicit_staging_validates_capacity(monkeypatch, tmp_path):
    from ssd import checkpoint_staging

    monkeypatch.setattr(
        checkpoint_staging.shutil, "disk_usage", lambda _: SimpleNamespace(free=0)
    )
    assert staging_directory("off", 1024) is None
    assert staging_directory("auto", 1024) is None
    with pytest.raises(ValueError, match="lacks space"):
        staging_directory(str(tmp_path), 1024)


def test_storage_lock_excludes_another_process_and_releases_on_error(
    monkeypatch, tmp_path
):
    import subprocess
    import sys

    from ssd.checkpoint_staging import checkpoint_io_lock

    lock = tmp_path / "storage.lock"
    monkeypatch.setenv("SSD_CHECKPOINT_IO_LOCK", str(lock))
    probe = """
import fcntl, sys
with open(sys.argv[1], 'a') as stream:
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(2)
"""

    def check():
        return subprocess.run(
            [sys.executable, "-c", probe, str(lock)], timeout=10, check=False
        ).returncode

    with (
        pytest.raises(RuntimeError, match="failed write"),
        checkpoint_io_lock() as waited,
    ):
        assert waited >= 0
        assert check() == 2
        raise RuntimeError("failed write")
    assert check() == 0


def test_storage_lock_is_optional(monkeypatch):
    from ssd.checkpoint_staging import checkpoint_io_lock

    monkeypatch.delenv("SSD_CHECKPOINT_IO_LOCK", raising=False)
    with checkpoint_io_lock() as waited:
        assert waited == 0


def test_actual_trainer_checkpoints_publish_resumable_state(monkeypatch, tmp_path):
    import copy

    import torch
    from distil_config import DistilConfig
    from distil_trainer import BaseTrainer, DistilTrainer
    from transformers import Qwen3Config, Qwen3ForCausalLM, Trainer, TrainerCallback

    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setenv("SSD_CHECKPOINT_STAGING", str(staging))

    class SnapshotTrainer(DistilTrainer):
        def __init__(self, model, args, data):
            BaseTrainer.__init__(self, model=model, args=args, train_dataset=data)
            self.ref_model = copy.deepcopy(model)

        get_train_dataloader = Trainer.get_train_dataloader
        _get_train_sampler = Trainer._get_train_sampler
        _prepare_inputs = Trainer._prepare_inputs
        _set_signature_columns_if_needed = Trainer._set_signature_columns_if_needed
        log = Trainer.log

        def compute_loss(self, model, inputs, **kwargs):
            return model(**inputs).loss

        def create_model_card(self, **kwargs):
            pass

    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
        )
    )
    args = DistilConfig(
        output_dir=str(tmp_path / "run"),
        use_cpu=True,
        bf16=False,
        report_to="none",
        max_steps=3,
        save_steps=1,
        save_total_limit=2,
        per_device_train_batch_size=1,
        num_generations=1,
        persist_ref_model=True,
        gradient_checkpointing=False,
    )
    data = [
        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])}
    ] * 2
    trainer = SnapshotTrainer(model, args, data)

    class BestAtOne(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1:
                state.best_global_step = 1
                state.best_metric = 0.5
                state.best_model_checkpoint = str(
                    Path(args.output_dir) / "checkpoint-1"
                )
            return control

    trainer.add_callback(BestAtOne())
    trainer.add_callback(PruneCheckpointsBeforeSave(args.output_dir))
    trainer.train()
    assert {p.name for p in Path(args.output_dir).glob("checkpoint-*")} == {
        "checkpoint-1",
        "checkpoint-3",
    }
    for step in (1, 3):
        checkpoint = Path(args.output_dir) / f"checkpoint-{step}"
        for name in (
            "model.safetensors",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "teacher_state.pt",
            "trainer_state.json",
        ):
            assert (checkpoint / name).is_file(), name
        assert not (checkpoint / PENDING).exists()
        assert (
            json.loads((checkpoint / "trainer_state.json").read_text())["global_step"]
            == step
        )
        assert json.loads((checkpoint / "trainer_state.json").read_text())[
            "best_model_checkpoint"
        ] == str(Path(args.output_dir) / "checkpoint-1")
        teacher = torch.load(checkpoint / "teacher_state.pt", weights_only=True)
        for name, value in trainer.ref_model.state_dict().items():
            torch.testing.assert_close(teacher[name], value)
    assert not list(staging.iterdir())
    assert latest_checkpoint(args.output_dir).endswith("checkpoint-3")
    records = [
        json.loads(row)
        for row in (Path(args.output_dir) / "io-timings.jsonl").read_text().splitlines()
    ]
    assert sum(row["kind"] == "checkpoint_publication" for row in records) == 3
    assert (
        sum(row["kind"] == "checkpoint_foreground" and row["staged"] for row in records)
        == 3
    )
    # Exercise the real Trainer restore path, including teacher and optimizer,
    # rather than treating the mere presence of serialized files as sufficient.
    args.max_steps = 4
    resumed = SnapshotTrainer(Qwen3ForCausalLM(model.config), args, data)
    resumed.train(resume_from_checkpoint=str(Path(args.output_dir) / "checkpoint-3"))
    assert resumed.state.global_step == 4
    assert resumed.lr_scheduler.last_epoch == 4
    assert all(int(s["step"]) == 4 for s in resumed.optimizer.state.values())
    for name, value in trainer.ref_model.state_dict().items():
        torch.testing.assert_close(resumed.ref_model.state_dict()[name], value)
    assert latest_checkpoint(args.output_dir).endswith("checkpoint-4")
    assert not list(staging.iterdir())
