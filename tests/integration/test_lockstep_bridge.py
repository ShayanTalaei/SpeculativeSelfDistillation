"""Exercise isolated trainer/vLLM workers, CUDA weight refresh and hidden reuse.

Run using .venv; workers use the same current environment.
"""

import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.getenv("PAIRED_GPU_TESTS") != "1", reason="requires two CUDA GPUs"
    ),
]


def test_runner_keeps_hidden_targets_on_student_gpu(tmp_path):
    import torch
    from transformers import AutoModelForCausalLM

    from paired_vllm.validation import make_model

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    from ssd.paired_cache import full_teacher_logps
    from ssd.paired_runner import PairedRunner

    torch.set_num_threads(1)
    paths = [
        make_model(tmp_path / name, seed)
        for name, seed in (("teacher", 42), ("student", 43))
    ]
    models = [
        AutoModelForCausalLM.from_pretrained(
            p, torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
        .eval()
        .to("cuda:0")
        for p in paths
    ]

    class Tokenizer:
        eos_token_id = 2

        def __call__(self, prompts, **kwargs):
            return {"input_ids": [[int(t) for t in p.split()] for p in prompts]}

        def decode(self, ids, **kwargs):
            return " ".join(map(str, ids))

    args = SimpleNamespace(
        steps_per_generation=1,
        gradient_accumulation_steps=1,
        num_iterations=1,
        routing_strategy="jsd_switch",
        jsd_threshold=0.0108,
        paired_teacher_cache="hidden",
        paired_teacher_gpu="1",
        paired_student_gpu="0",
        paired_capacity=256,
        temperature=0.8,
        top_k=20,
        sync_ref_model=False,
        paired_compile_model=False,
        paired_token_budget=64,
        paired_transfer_logits=True,
        per_device_train_batch_size=2,
        output_dir=str(tmp_path),
        paired_python=str(root / ".venv/bin/python"),
        seed=42,
    )
    trainer = SimpleNamespace(
        args=args,
        accelerator=SimpleNamespace(num_processes=1, device=torch.device("cuda:0")),
        is_fsdp_enabled=False,
        is_deepspeed_enabled=False,
        processing_class=Tokenizer(),
        model=models[1],
        ref_model=models[0],
        optimizer=None,
        state=SimpleNamespace(global_step=0),
        max_prompt_length=64,
        _metrics=defaultdict(lambda: defaultdict(list)),
    )
    teachers = ["3 7 11 " * 8, "5 7 " * 15, "4 8 " * 9]
    students = ["5 " * 20, "3 9 " * 6, "7 " * 14]
    runner = PairedRunner(
        trainer, paths[1], paths[0], str(tmp_path / "bridge"), 128, 0.8
    )
    try:
        for version in (0, 1):
            trainer.state.global_step = version
            if version:
                with torch.no_grad():
                    trainer.model.model.norm.weight.mul_(0.8)
            runner.wake_up()
            runner.update_named_params("a", trainer.model.state_dict().items())
            runner.update_named_params("b", trainer.ref_model.state_dict().items())
            if version:
                assert "a" in runner.cuda_weights
            outputs = runner.generate_batch(students, teachers, 20, 0.8, 20)
            runner.sleep()
            assert next(trainer.ref_model.parameters()).is_cuda
            for prompt, output in zip(teachers, outputs):
                cache = output.full_teacher_cache
                assert cache.values.device == torch.device("cuda:0")
                assert cache.version == version
                ids = torch.tensor([output.token_ids], device="cuda:0")
                actual = full_teacher_logps(
                    [cache], trainer.ref_model.lm_head, ids, version, 0.8, "cuda:0"
                )
                prefix = [int(x) for x in prompt.split()]
                with torch.no_grad():
                    hidden = trainer.ref_model.model(
                        torch.tensor([prefix + output.token_ids[:-1]], device="cuda:0"),
                        use_cache=False,
                    ).last_hidden_state[0, len(prefix) - 1 :]
                    expected = (
                        trainer.ref_model.lm_head(hidden).float() / 0.8
                    ).log_softmax(-1)
                torch.testing.assert_close(actual[0], expected, atol=0.04, rtol=0.04)
                logits = torch.randn_like(actual, requires_grad=True)
                loss = torch.nn.functional.kl_div(
                    logits.log_softmax(-1),
                    actual,
                    reduction="batchmean",
                    log_target=True,
                )
                loss.backward()
                assert torch.isfinite(logits.grad).all()
            assert not list(runner.scratch.glob("targets-*.pt"))
        # Evaluation switches graph policy without leaving target leases behind.
        runner.wake_up()
        runner.update_named_params("a", trainer.model.state_dict().items())
        assert (
            len(
                runner.generate(
                    students, SimpleNamespace(max_tokens=5, temperature=0.0, top_k=0)
                )
            )
            == 3
        )
        runner.sleep()
    finally:
        runner.close()
