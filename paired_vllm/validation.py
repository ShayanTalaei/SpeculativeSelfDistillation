"""Independent full-history eager oracle; no paired scheduler/cache code reused."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def make_model(path, seed, tied=False):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    path = Path(path)
    if (path / "config.json").exists():
        return str(path)
    torch.manual_seed(seed)
    config = Qwen3Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=8192,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=tied,
    )
    model = Qwen3ForCausalLM(config)
    model.save_pretrained(path)
    tokenizer = Tokenizer(WordLevel({f"t{i}": i for i in range(256)}, unk_token="t0"))
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="t0", bos_token="t1", eos_token="t2"
    ).save_pretrained(path)
    return str(path)


def validate(
    directory,
    gaps=(0, 5, 16, 2048),
    policies=("teacher", "student", "alternating"),
    graphs=False,
):
    import torch
    from transformers import AutoModelForCausalLM

    from paired_vllm.core import PairConfig
    from paired_vllm.engine import PairedEngine, RuntimeConfig

    torch.set_num_threads(1)
    directory = Path(directory)
    teacher = make_model(directory / "teacher", 42)
    student = make_model(directory / "student", 43)
    # CPU float32 reference has no paged cache, no shared worker adapter.
    models = [
        AutoModelForCausalLM.from_pretrained(
            p, dtype=torch.float32, attn_implementation="eager"
        ).eval()
        for p in (teacher, student)
    ]
    max_error, histories_match, tokens_match = 0.0, True, True
    comparisons = 0
    for policy in policies:
        config = RuntimeConfig(
            teacher,
            student,
            PairConfig(capacity=16384, token_budget=128, max_requests=8),
            policy=policy,
            max_model_len=4096,
            graphs=graphs,
        )
        with PairedEngine(config) as engine:
            for i, gap in enumerate(gaps):
                # Deliberately different 15/16/17 boundaries and row reorderings.
                student_prompt = [3 + (j * 7 + i) % 200 for j in range(15 + i % 3)]
                teacher_prompt = [
                    3 + (j * 11 + i) % 200 for j in range(len(student_prompt) + gap)
                ]
                engine.add(str(i), teacher_prompt, student_prompt, max_tokens=4)
            while engine.core.unfinished:
                events = engine.step()
                states = engine.inspect()
                for event in events:
                    req = engine.core.requests[event.request_id]
                    expected_logits = []
                    for side, model in enumerate(models):
                        prefix = list(req.prompts[side]) + req.committed[:-1]
                        with torch.inference_mode():
                            logits = (
                                model(torch.tensor([prefix]), use_cache=False)
                                .logits[0, -1]
                                .float()
                            )
                        actual = states[side][f"{event.request_id}:{event.incarnation}"]
                        max_error = max(
                            max_error,
                            (logits - torch.tensor(actual["logits"]))
                            .abs()
                            .max()
                            .item(),
                        )
                        if side == 0:
                            reconstructed = states[1][
                                f"{event.request_id}:{event.incarnation}"
                            ]["teacher_logits"]
                            max_error = max(
                                max_error,
                                (
                                    torch.tensor(actual["logits"])
                                    - torch.tensor(reconstructed)
                                )
                                .abs()
                                .max()
                                .item(),
                            )
                        histories_match &= (
                            actual["tokens"] == list(req.prompts[side]) + req.committed
                        )
                        histories_match &= actual["computed"] == len(prefix)
                        expected_logits.append(logits)
                        comparisons += 1
                    source = (
                        0
                        if policy == "teacher"
                        or (policy == "alternating" and event.output_index % 2 == 0)
                        else 1
                    )
                    tokens_match &= event.token_id == int(
                        expected_logits[source].argmax()
                    )
                    tokens_match &= event.source_model == (
                        "teacher" if source == 0 else "student"
                    )
    result = {
        "max_logit_error": max_error,
        "tokens_match": tokens_match,
        "histories_match": histories_match,
        "comparisons": comparisons,
        "gaps": list(gaps),
        "policies": list(policies),
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="scratch/paired-validation")
    parser.add_argument("--graphs", action="store_true")
    args = parser.parse_args()
    result = validate(args.directory, graphs=args.graphs)
    print(json.dumps(result, indent=2))
    if (
        not result["tokens_match"]
        or not result["histories_match"]
        or result["max_logit_error"] >= 0.08
    ):
        raise SystemExit(1)
