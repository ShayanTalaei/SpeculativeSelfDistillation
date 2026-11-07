"""Resident targets survive inference sleep and preserve the teacher snapshot."""

import os
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.getenv("PAIRED_GPU_TESTS") != "1", reason="requires two CUDA GPUs"
    ),
]


@pytest.mark.parametrize("transfer_logits", [True, False])
def test_resident_hidden_sleep_refresh_and_projection(tmp_path, transfer_logits):
    import torch
    from transformers import AutoModelForCausalLM

    from paired_vllm.async_core import RolloutRequest
    from paired_vllm.core import PairConfig
    from paired_vllm.cuda_cache import copy_cuda_target
    from paired_vllm.engine import RuntimeConfig
    from paired_vllm.lockstep_engine import LockstepPairedEngine
    from paired_vllm.training_ops import write_weights
    from paired_vllm.validation import make_model
    from tests.integration.test_async_gpu import reference

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Self-Distillation"))
    from ssd.paired_cache import FullTeacherCache, full_teacher_logps

    torch.set_num_threads(1)
    paths = [
        make_model(tmp_path / name, seed)
        for name, seed in (("teacher", 42), ("student", 43))
    ]
    models = [
        AutoModelForCausalLM.from_pretrained(
            p, dtype=torch.float32, attn_implementation="eager"
        ).eval()
        for p in paths
    ]
    maps = [str(tmp_path / (name + "-weights")) for name in ("teacher", "student")]
    for path, model in zip(maps, models):
        write_weights(path, model.state_dict())
    requests = [
        RolloutRequest(str(i), [3, 7, 11] * (8 + i), [5] * (23 - i), 20 + i)
        for i in range(4)
    ]
    runtime = RuntimeConfig(
        *paths,
        PairConfig(capacity=256, token_budget=64, max_requests=2),
        teacher_gpu="1",
        student_gpu="0",
        policy="jsd:.0108",
        max_model_len=128,
        dtype="float32",
        graphs=True,
        capture_targets=True,
        enable_sleep_mode=True,
        keep_teacher_resident=True,
        temperature=0.8,
        top_k=20,
    )
    with LockstepPairedEngine(
        runtime, transfer_logits=transfer_logits, fused_jsd=True, resident_targets=True
    ) as engine:
        for version in (0, 1):
            if version:
                with torch.no_grad():
                    models[0].model.norm.weight.mul_(0.7)
                write_weights(maps[0], models[0].state_dict())
                engine.training_command("WAKE")
            engine.training_command(
                "SYNC",
                {
                    "teacher": maps[0],
                    "student": maps[1],
                    "version": str(version),
                    "seed": 42,
                    "sync_teacher": True,
                },
            )
            expected = {
                r.request_id: reference(models, r, 0.0108, 0.8, 20) for r in requests
            }
            outputs = engine.generate(requests)
            descriptors = engine.targets()
            assert engine.last_metrics["admission_batches"] == 2
            with pytest.raises(RuntimeError, match="release"):
                engine.training_command("SYNC", {})
            assert set(engine.training_command("SLEEP")) == {1}
            with pytest.raises(RuntimeError, match="refresh"):
                engine.generate(requests)
            cached = {}
            for output in outputs:
                rid, tokens = output["request_id"], output["token_ids"]
                assert tokens == expected[rid][0]
                assert output["sources"] == expected[rid][1]
                with pytest.raises(ValueError, match="version"):
                    copy_cuda_target(
                        descriptors[rid],
                        device="cuda:0",
                        version=version + 9,
                        token_ids=tokens,
                    )
                hidden = copy_cuda_target(
                    descriptors[rid], device="cuda:0", version=version, token_ids=tokens
                )
                assert hidden.device == torch.device("cuda:0")
                torch.testing.assert_close(
                    hidden.cpu(), expected[rid][2], atol=0.004, rtol=0.004
                )
                cached[rid] = hidden
            engine.release_targets(list(engine.target_leases))
            # Loss needs just the matching projection, with inference asleep
            # and producer leases already released. Shuffle accumulation order.
            head = torch.nn.Linear(
                models[0].config.hidden_size,
                models[0].config.vocab_size,
                bias=False,
                device="cuda:0",
            )
            head.load_state_dict(models[0].lm_head.state_dict())
            for output in reversed(outputs):
                rid, tokens = output["request_id"], output["token_ids"]
                cache = FullTeacherCache("hidden", cached[rid], version, tokens)
                ids = torch.tensor([tokens], device="cuda:0")
                actual = full_teacher_logps([cache], head, ids, version, 0.8, "cuda:0")
                expected_logps = (
                    head(expected[rid][2].to("cuda:0")) / 0.8
                ).log_softmax(-1)
                torch.testing.assert_close(
                    actual[0], expected_logps, atol=0.004, rtol=0.004
                )
                student_logits = torch.randn_like(actual, requires_grad=True)
                loss = torch.nn.functional.kl_div(
                    student_logits.log_softmax(-1),
                    actual,
                    log_target=True,
                    reduction="batchmean",
                )
                loss.backward()
                assert torch.isfinite(student_logits.grad).all()
            del cached, hidden, cache, actual, head
        # A resident teacher must also work without another teacher upload.
        assert set(engine.training_command("WAKE")) == {1}
        engine.training_command(
            "SYNC",
            {
                "teacher": str(tmp_path / "must-not-read-frozen-teacher"),
                "student": maps[1],
                "version": "1",
                "seed": 42,
                "sync_teacher": False,
            },
        )
        for output in engine.generate(requests):
            rid = output["request_id"]
            assert output["token_ids"] == expected[rid][0]
            copy_cuda_target(
                engine.targets()[rid],
                device="cuda:0",
                version=1,
                token_ids=output["token_ids"],
            )
        engine.release_targets(list(engine.target_leases))
