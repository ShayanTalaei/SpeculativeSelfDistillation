"""Small real CUDA-graph engine verifies sleep/teardown without a training run."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(os.environ.get("ONPOLICY_GPU_TESTS") != "1", reason="opt-in GPU test")
def test_sleeping_colocated_engine_shuts_down_cleanly(tmp_path):
    script = r'''
import os, socket, sys
from types import SimpleNamespace
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from distil_trainer import DistilTrainer
from vllm import LLM

config = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                    num_hidden_layers=2, num_attention_heads=2,
                    num_key_value_heads=1, head_dim=32)
Qwen3ForCausalLM(config).to(torch.bfloat16).save_pretrained(sys.argv[1])
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
os.environ.update(RANK='0', LOCAL_RANK='0', WORLD_SIZE='1',
                  MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
llm = LLM(model=sys.argv[1], skip_tokenizer_init=True, dtype='bfloat16',
          enable_sleep_mode=True, enforce_eager=False,
          gpu_memory_utilization=0.1, max_model_len=32, max_num_seqs=2,
          max_num_batched_tokens=32, distributed_executor_backend='external_launcher',
          logprobs_mode='processed_logprobs',
          compilation_config={'cudagraph_capture_sizes': [1, 2]})
llm.sleep(level=2)
llm.wake_up()
llm.sleep(level=1)  # Online evaluation leaves the engine in this state.
trainer = SimpleNamespace(use_vllm=True, vllm_mode='colocate', llm=llm,
                          args=SimpleNamespace(vllm_enable_sleep_mode=True))
DistilTrainer.close_vllm(trainer)
if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()
print('CLEAN_SHUTDOWN')
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "model")],
        capture_output=True, text=True, timeout=240, check=False,
        env={**os.environ, "VLLM_PLUGINS": "", "OMP_NUM_THREADS": "4",
             "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")},
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "CLEAN_SHUTDOWN" in output
    assert "CUDA Error" not in output
