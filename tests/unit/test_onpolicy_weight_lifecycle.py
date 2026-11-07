from collections import defaultdict
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("frozen_teacher_generation", [False, True])
def test_discarded_weights_reload_even_at_same_step(monkeypatch, frozen_teacher_generation):
    import torch
    from distil_trainer import DistilTrainer

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    class Engine:
        ready = False
        syncs = 0

        def wake_up(self):
            pass

        def generate(self, *args, **kwargs):
            assert self.ready, "Generation used discarded inference weights"
            assert kwargs["tokenization_kwargs"] == {
                "truncate_prompt_tokens": 4, "truncation_side": "left",
            }
            return [SimpleNamespace(prompt_token_ids=[1], outputs=[SimpleNamespace(
                token_ids=[2], logprobs=[{2: SimpleNamespace(logprob=-0.5)}],
            )])]

        def sleep(self, level):
            assert level == 2
            self.ready = False

        def sync(self):
            self.ready = True
            self.syncs += 1

    engine = Engine()
    trainer = SimpleNamespace(
        accelerator=SimpleNamespace(device="cpu", is_main_process=True),
        processing_class=None, model=SimpleNamespace(training=True),
        use_vllm=True, vllm_mode="colocate", vllm_tensor_parallel_size=1,
        args=SimpleNamespace(vllm_enable_sleep_mode=True, sync_ref_model=False, generation_kwargs=None, report_to=[]),
        generate_from_teacher=frozen_teacher_generation,
        state=SimpleNamespace(global_step=7), _last_loaded_step=7,
        _vllm_weights_discarded=True, llm=engine,
        _move_model_to_vllm=engine.sync,
        _metrics=defaultdict(lambda: defaultdict(list)),
        repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=20,
        min_p=None, max_completion_length=3, max_prompt_length=4,
    )
    for _ in range(2):
        prompt_ids, completion_ids, logps, _ = DistilTrainer._generate_single_turn(trainer, ["question"], None)
        assert prompt_ids == [[1]] and completion_ids == [[2]]
        assert logps == [[-0.5]]
        assert trainer._vllm_weights_discarded
    assert engine.syncs == 2
