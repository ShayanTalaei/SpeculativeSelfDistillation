"""Publish immutable local checkpoints without blocking subsequent GPU steps.

Only one checkpoint may be in flight. A pending marker excludes partial copies
from resume/pruning, and trainer_state.json is copied last. Publication errors
propagate on the next callback; train end waits for all files to reach storage.
"""

import fcntl
import os
import shutil
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from transformers import TrainerCallback

from ssd.io_timing import record_io_timing

PENDING = ".publication-pending"


@contextmanager
def checkpoint_io_lock():
    """Optionally serialize storage writes across independent local trainers."""
    path = os.environ.get("SSD_CHECKPOINT_IO_LOCK")
    if not path:
        yield 0.0
        return
    # Use a host-local file, not an object-store mount with emulated locking.
    with open(path, "a") as stream:
        started = time.perf_counter()
        fcntl.flock(stream, fcntl.LOCK_EX)
        waited = time.perf_counter() - started
        try:
            yield waited
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def staging_directory(setting, required_bytes):
    if setting in (None, "", "off"):
        return None
    root = Path("/dev/shm" if setting == "auto" else setting)
    if (
        not root.is_dir()
        or shutil.disk_usage(root).free < required_bytes * 1.25 + 256 * 1024**2
    ):
        if setting != "auto":
            raise ValueError(
                f"Checkpoint staging directory is missing or lacks space: {root}"
            )
        return None
    return Path(tempfile.mkdtemp(prefix="distil-checkpoint-", dir=root))


def publish_checkpoint(source, destination):
    """The input snapshot is complete and immutable before this function runs."""
    source, destination = Path(source), Path(destination)
    if destination.exists() and (
        (destination / PENDING).exists()
        or not (destination / "trainer_state.json").exists()
    ):
        # A resumed job can revisit a save interrupted in the prior process.
        # Preserve that partial directory instead of mixing two snapshots.
        destination.rename(
            destination.with_name(f".incomplete-{destination.name}-{uuid.uuid4().hex}")
        )
    destination.mkdir(parents=True, exist_ok=False)
    (destination / PENDING).write_text("Checkpoint publication is in progress.\n")
    files = sorted(p for p in source.rglob("*") if p.is_file())
    state = source / "trainer_state.json"
    if state not in files:
        raise ValueError("Checkpoint snapshot lacks trainer_state.json")
    files.remove(state)
    files.append(state)
    copied = 0
    for file in files:
        target = destination / file.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
        with target.open("rb") as stream:
            os.fsync(stream.fileno())
        if file.stat().st_size != target.stat().st_size:
            raise OSError(f"Incomplete checkpoint copy: {target}")
        copied += target.stat().st_size
    (destination / PENDING).unlink()
    return copied


class CheckpointPublisher:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="checkpoint")
        self.future = None

    def submit(self, staging_parent, destination, step):
        self.wait()
        source = Path(staging_parent) / Path(destination).name

        def publish():
            started = time.perf_counter()
            started_unix = time.time()
            # Keep the complete RAM snapshot if publication fails: it remains
            # recoverable until node shutdown, and its path is in the error.
            try:
                with checkpoint_io_lock() as storage_wait:
                    size = publish_checkpoint(source, destination)
            except Exception as exc:
                raise RuntimeError(
                    f"Checkpoint publication failed; snapshot retained at {source}"
                ) from exc
            seconds = time.perf_counter() - started
            finished_unix = time.time()
            shutil.rmtree(staging_parent)
            return record_io_timing(
                self.output_dir,
                step,
                "checkpoint_publication",
                started_at_unix=started_unix,
                finished_at_unix=finished_unix,
                seconds=seconds,
                storage_wait_seconds=storage_wait,
                storage_active_seconds=seconds - storage_wait,
                bytes=size,
                bytes_per_second=size / max(seconds, 1e-9),
                checkpoint=str(destination),
            )

        self.future = self.pool.submit(publish)

    def poll(self):
        if self.future is not None and self.future.done():
            return self.wait()
        return None

    def wait(self):
        if self.future is None:
            return None
        result = self.future.result()
        self.future = None
        return result

    def close(self):
        try:
            return self.wait()
        finally:
            self.pool.shutdown(wait=True)


class CheckpointPublicationCallback(TrainerCallback):
    def __init__(self, trainer, publisher):
        self.trainer, self.publisher = trainer, publisher

    def report(self, result):
        if result is not None:
            from transformers.trainer_utils import rotate_checkpoints

            best = self.trainer.state.best_model_checkpoint
            if best is None or (Path(best) / "trainer_state.json").is_file():
                rotate_checkpoints(
                    output_dir=str(self.publisher.output_dir),
                    save_total_limit=self.trainer.args.save_total_limit,
                    best_model_checkpoint=best,
                )
            self.trainer.log(
                {
                    "checkpoint/published_step": result["step"],
                    "checkpoint/publication_seconds": result["seconds"],
                    "checkpoint/storage_wait_seconds": result["storage_wait_seconds"],
                    "checkpoint/publication_bytes_per_second": result[
                        "bytes_per_second"
                    ],
                }
            )

    def on_step_begin(self, args, state, control, **kwargs):
        self.report(self.publisher.poll())
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.report(self.publisher.close())
        return control
