"""Byte-exact isolated-process weight transfer, including tied and sliced tensors."""

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(os.getenv("PAIRED_GPU_TESTS") != "1", reason="requires CUDA")
def test_cuda_weight_transfer_between_processes():
    import torch

    from paired_vllm.cuda_weights import export_cuda_weights

    root = Path(__file__).resolve().parents[2]
    source = torch.arange(128, device="cuda", dtype=torch.bfloat16).reshape(16, 8)
    state = {"weight": source, "tied": source, "slice": source[4:8]}
    script = """
import json, pickle, sys, torch
from paired_vllm.cuda_weights import import_cuda_weights
payload = pickle.load(sys.stdin.buffer)
with import_cuda_weights(payload) as state:
    print(json.dumps({k: v.float().cpu().tolist() for k, v in state.items()}))
"""
    for _ in range(2):
        payload = export_cuda_weights(state)
        result = subprocess.run(
            [sys.executable, "-c", script],
            input=pickle.dumps(payload),
            capture_output=True,
            timeout=60,
            check=False,
            cwd=root,
        )
        assert result.returncode == 0, result.stderr.decode()
        values = json.loads(result.stdout)
        assert values == {k: v.float().cpu().tolist() for k, v in state.items()}
        source.add_(1)


@pytest.mark.skipif(os.getenv("PAIRED_GPU_TESTS") != "1", reason="requires CUDA")
def test_persistent_cuda_mapping_observes_updates_and_reopens_changed_allocations():
    import torch

    from paired_vllm.cuda_weights import export_cuda_weights

    script = """
import json, pickle, sys, torch
from paired_vllm.cuda_weights import CudaWeightMappingCache
cache = CudaWeightMappingCache()
while True:
    payload = pickle.load(sys.stdin.buffer)
    if payload is None:
        cache.close()
        print('closed', flush=True)
        break
    state, hit = cache.get(payload)
    values = {k: v.float().cpu().tolist() for k, v in state.items()}
    print(json.dumps({'hit': hit, 'values': values}), flush=True)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    sources = []  # Exported allocations must survive until the consumer closes them.
    try:
        source = torch.arange(128, device="cuda", dtype=torch.bfloat16).reshape(16, 8)
        sources.append(source)
        for step in range(3):
            if step == 2:
                source = source.clone() + 7
                sources.append(source)
            else:
                source.add_(1)
            state = {"weight": source, "tied": source, "slice": source[4:8]}
            pickle.dump(export_cuda_weights(state), child.stdin)
            child.stdin.flush()
            response = json.loads(child.stdout.readline())
            assert response["hit"] == (step == 1)
            assert response["values"] == {
                k: v.float().cpu().tolist() for k, v in state.items()
            }
        pickle.dump(None, child.stdin)
        child.stdin.flush()
        assert child.stdout.readline().strip() == b"closed"
        assert child.wait(timeout=30) == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


@pytest.mark.skipif(os.getenv("PAIRED_GPU_TESTS") != "1", reason="requires two CUDA GPUs")
def test_peer_gpu_pull_with_hidden_exporter_device():
    """A consumer that cannot even enumerate the exporter's GPU pulls byte-exact."""
    import torch

    from paired_vllm.cuda_weights import export_cuda_weights

    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA GPUs")
    root = Path(__file__).resolve().parents[2]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical = visible.split(",") if visible else [str(i) for i in range(torch.cuda.device_count())]
    source = torch.arange(4096, device="cuda:0", dtype=torch.bfloat16).reshape(64, 64)
    state = {"weight": source, "tied": source, "slice": source[8:24]}
    script = """
import json, pickle, sys, torch
from paired_vllm.cuda_weights import CudaWeightPuller
puller = CudaWeightPuller()
assert torch.cuda.device_count() == 1
while True:
    payload = pickle.load(sys.stdin.buffer)
    if payload is None:
        puller.close()
        print('closed', flush=True)
        break
    state, timings = puller.pull(payload, cache=True)
    assert all(t.device == torch.device('cuda:0') for t in state.values())
    assert state['weight'].data_ptr() == state['tied'].data_ptr()
    print(json.dumps({'values': {k: v.float().cpu().tolist() for k, v in state.items()},
                      'reopened': timings['weight_handles_reopened'],
                      'bytes': timings['weight_peer_bytes']}), flush=True)
"""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": physical[1]}
    child = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=env,
        cwd=root,
    )
    sources = [source]
    try:
        for step in range(3):
            if step == 2:
                source = source.clone() + 7
                sources.append(source)
            else:
                source.add_(1)
            state = {"weight": source, "tied": source, "slice": source[8:24]}
            pickle.dump(export_cuda_weights(state), child.stdin)
            child.stdin.flush()
            response = json.loads(child.stdout.readline())
            assert response["reopened"] == (step != 1)
            assert response["bytes"] == source.numel() * 2 + 16 * 64 * 2
            assert response["values"] == {k: v.float().cpu().tolist() for k, v in state.items()}
        pickle.dump(None, child.stdin)
        child.stdin.flush()
        assert child.stdout.readline().strip() == b"closed"
        assert child.wait(timeout=30) == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
