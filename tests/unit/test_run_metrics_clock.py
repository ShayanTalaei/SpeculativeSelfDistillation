import json
from types import SimpleNamespace

import pytest
from ssd.run_metrics import RunMetricsCallback


@pytest.mark.parametrize("finished,expected", [(False, 60), (True, 10)])
def test_cost_clock_includes_active_launcher_startup(monkeypatch, tmp_path, finished, expected):
    lifecycle = {"launcher_started_at_unix": 100}
    if finished:
        lifecycle["finished_at_unix"] = 140
    (tmp_path / "run-lifecycle.json").write_text(json.dumps(lifecycle))
    monkeypatch.setattr("ssd.run_metrics.time.time", lambda: 150)
    callback = RunMetricsCallback(str(tmp_path), "onpolicy", "model", 1)
    monkeypatch.setattr("ssd.run_metrics.time.time", lambda: 160)
    callback.on_log(None, SimpleNamespace(global_step=1, is_world_process_zero=True), None,
                    logs={"loss": 0.1, "num_tokens": 100})
    record = json.loads((tmp_path / "run_metrics.jsonl").read_text())
    assert record["job_wall_s"] == expected
    assert record["cum_gpu_hours"] == expected / 3600
    assert record["callback_started_at_unix"] == 150
