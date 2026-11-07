"""Exercise direct evaluation logging against the installed W&B SDK."""

import json
import struct
import zlib
from pathlib import Path
from types import SimpleNamespace

from evals.online_eval import EvalGroup, OnlineEvalCallback, exact_match_scorer
from wandb.proto import wandb_internal_pb2 as pb

import wandb


def records(path):
    # W&B's documented LevelDB framing; SDK 0.30 removed its Python reader.
    # https://github.com/wandb/wandb/blob/v0.22.2/wandb/sdk/internal/datastore.py
    with path.open("rb") as stream:
        assert struct.unpack("<4sHB", stream.read(7)) == (b":W&B", 0xBEE1, 0)
        parts = []
        while True:
            remaining = 32768 - stream.tell() % 32768
            if remaining < 7:
                assert stream.read(remaining) == bytes(remaining)
            header = stream.read(7)
            if not header:
                break
            checksum, length, kind = struct.unpack("<IHB", header)
            data = stream.read(length)
            assert len(data) == length
            assert zlib.crc32(bytes([kind]) + data) & 0xFFFFFFFF == checksum
            if kind in (1, 2):
                assert not parts
            else:
                assert kind in (3, 4) and parts
            parts.append(data)
            if kind in (1, 4):
                record = pb.Record()
                record.ParseFromString(b"".join(parts))
                yield record
                parts = []
        assert not parts


def test_evaluations_survive_ahead_of_optimizer_history(tmp_path):
    callback = OnlineEvalCallback.__new__(OnlineEvalCallback)
    callback.output_dir = str(tmp_path)
    callback._file = callback._rollouts_file = None
    group = EvalGroup(rows=[], scorer=exact_match_scorer, metrics_prefix="eval")
    run = wandb.init(project="online-eval-regression", mode="offline", dir=str(tmp_path),
                     settings=wandb.Settings(console="off", x_disable_stats=True))
    transaction_log = Path(run.dir).parent / f"run-{run.id}.wandb"
    try:
        for step in (3, 6):
            for marker in range(20):
                run.log({"profiling/marker": marker})
            callback._log_group(SimpleNamespace(global_step=step, epoch=step / 334),
                                group, {"eval/mean_score": step / 10, "eval/n_total": 1},
                                [{"question": "q", "reference": "A", "response": "A", "score": 1}])
            run.log({"train/global_step": step, "train/loss": 0.1})
    finally:
        callback.on_train_end(None, None, None)
        run.finish()
    rows = []
    for record in records(transaction_log):
        if record.WhichOneof("record_type") == "history":
            rows.append({item.key or "/".join(item.nested_key): json.loads(item.value_json)
                         for item in record.history.item})
    evaluations = [row for row in rows if "eval/mean_score" in row]
    assert [row["train/global_step"] for row in evaluations] == [3, 6]
    assert [row["eval/mean_score"] for row in evaluations] == [0.3, 0.6]
    assert [row["_step"] for row in rows] == list(range(44))
    assert all(row["_step"] > row["train/global_step"] for row in evaluations)
    local = [json.loads(line) for line in (tmp_path / "eval_online.jsonl").read_text().splitlines()]
    assert [row["eval/mean_score"] for row in local] == [0.3, 0.6]
