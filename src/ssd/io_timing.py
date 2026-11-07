"""Direct wall-clock records for work outside the optimizer-step timer."""

import json
import threading
import time
from pathlib import Path

_WRITE_LOCK = threading.Lock()


def record_io_timing(output_dir, step, kind, **values):
    finished = values.pop("finished_at_unix", time.time())
    started = values.pop("started_at_unix", None)
    if started is None and "seconds" in values:
        started = finished - values["seconds"]
    row = {"step": step, "kind": kind, "timestamp": finished,
           "started_at_unix": started, "finished_at_unix": finished, **values}
    if output_dir:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        with _WRITE_LOCK, (directory / "io-timings.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
    return row
