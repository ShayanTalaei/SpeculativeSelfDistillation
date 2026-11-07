"""Generic online evaluation callback with pluggable scoring.

Works with both on-policy (train.py) and off-policy (train_off_policy.py)
training scripts. Uses the trainer's colocated vLLM instance for generation.

Usage::

    from evals.online_eval import OnlineEvalCallback, make_llm_judge_scorer

    callback = OnlineEvalCallback(
        eval_data=eval_rows,           # list[DistillationRow]
        scoring_fn=judge_scorer,       # callable(q, ref, resp) -> dict
        vllm_instance=llm,             # colocated vLLM
        model_name="Qwen/Qwen3-4B",
        eval_every_n_steps=50,
    )
    trainer.add_callback(callback)
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

import torch
from openai import OpenAI
from transformers import AutoTokenizer, TrainerCallback
from vllm import LLM, SamplingParams


# ── EvalGroup abstraction ──────────────────────────────────────


@dataclass
class EvalGroup:
    """One logical eval slice: rows, how to score them, and where to log.

    Attributes:
        rows: Questions + references. DistillationRow.metadata is used for
            per-question grouping (see ``group_by``).
        scorer: Callable(question, reference, response, row_metadata) -> dict
            with at minimum ``{"score": float}``. May return additional
            fields; they are passed through to the per-question record and
            dumped to the eval JSONL.
        metrics_prefix: wandb / JSONL namespace (e.g. ``"eval"``,
            ``"recall"``, ``"generalization"``).
        group_by: Row-metadata keys to pivot scores on. ``["category"]``
            emits ``{prefix}/{category}/mean_score``; ``["split",
            "category"]`` emits ``{prefix}/{split}/{category}/mean_score``
            AND the per-axis aggregates ``{prefix}/{split}/mean_score``
            and ``{prefix}/{category}/mean_score``.
        chat_template_kwargs: Passed to ``tokenizer.apply_chat_template``
            for this group's prompts (e.g. ``{"enable_thinking": False}``).
        generation_kwargs: Per-group generation overrides (``temperature``,
            ``top_k``, ``max_new_tokens``). Missing keys fall back to the
            callback-level defaults.
    """
    rows: list
    scorer: Callable
    metrics_prefix: str = "eval"
    group_by: list = field(default_factory=list)
    chat_template_kwargs: dict = field(default_factory=dict)
    generation_kwargs: dict = field(default_factory=dict)


def _row_metadata(row) -> dict:
    """Return a row's metadata dict, or {} if it has none."""
    md = getattr(row, "metadata", None) or {}
    # eval_type from legacy flat-callback usage is also a grouping axis.
    et = getattr(row, "eval_type", None)
    if et is not None and "eval_type" not in md:
        md = {**md, "eval_type": et}
    return md


def aggregate_group_metrics(
    group: "EvalGroup",
    responses: list,
    *,
    max_workers: int = 128,
) -> tuple[dict, list]:
    """Score each (row, response) and compute flat + pivoted metrics.

    Scorers run in a ``ThreadPoolExecutor`` — ours are either pure-Python
    (fast, the pool overhead is negligible) or I/O-bound LLM-judge calls
    (benefit from parallelism).

    Returns:
        (metrics, records)
        metrics: flat dict of wandb-loggable keys, all prefixed by
            ``group.metrics_prefix`` and optionally nested by ``group_by``
            values.
        records: per-row dicts with {question, reference, response, score,
            verdict (if present), metadata, + any extra scorer fields}.
    """
    # Score (threaded) — order preserved by the index map.
    scored: list = [None] * len(responses)
    metadatas: list = [_row_metadata(r) for r in group.rows]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for i, (row, resp, md) in enumerate(
            zip(group.rows, responses, metadatas)
        ):
            futures[pool.submit(
                group.scorer, row.question, row.answer, resp, md,
            )] = i
        for fut in as_completed(futures):
            scored[futures[fut]] = fut.result()

    records: list = []
    scores: list = []
    buckets: dict = {}

    for row, resp, md, sc in zip(group.rows, responses, metadatas, scored):
        score = float(sc["score"])
        scores.append(score)
        records.append({
            "question": row.question,
            "reference": row.answer,
            "response": resp,
            "metadata": md,
            **sc,
        })

        # Bucket into every subset: each axis singly, plus the full
        # nested tuple if more than one axis.
        for axis in group.group_by:
            if axis in md:
                buckets.setdefault((axis, md[axis]), []).append(score)
        if len(group.group_by) > 1:
            key = tuple((axis, md.get(axis, "_missing_")) for axis in group.group_by)
            buckets.setdefault(key, []).append(score)

    prefix = group.metrics_prefix
    metrics: dict = {}
    if scores:
        metrics[f"{prefix}/mean_score"] = sum(scores) / len(scores)
        metrics[f"{prefix}/n_total"] = len(scores)

    for key, vals in buckets.items():
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        if isinstance(key, tuple) and isinstance(key[0], tuple):
            # Nested key: tuple of (axis, value) pairs
            suffix = "/".join(str(v) for (_, v) in key)
        else:
            # Single-axis key: (axis, value)
            suffix = str(key[1])
        metrics[f"{prefix}/{suffix}/mean_score"] = mean
        metrics[f"{prefix}/{suffix}/n_total"] = len(vals)

    return metrics, records


# ── Scoring functions ──────────────────────────────────────────


JUDGE_PROMPT = """\
You are an expert evaluator assessing whether a model's response \
correctly answers a question.

Your task is to compare the model's response to the reference answer \
and determine if the model's response is:
1. CORRECT: The response contains the key information from the \
reference answer, even if phrased differently or includes additional \
correct details.
2. PARTIALLY_CORRECT: The response contains most of the key \
information from the reference answer but misses some details.
3. INCORRECT: The response is wrong, misses the main point, or \
provides incorrect information.

Focus on factual accuracy and completeness, not on writing style or \
verbosity.

[Question]
{question}

[Reference Answer]
{reference_answer}

[Model Response]
{model_response}

Evaluate the model's response. Output ONLY one of: \
"CORRECT", "PARTIALLY_CORRECT", or "INCORRECT"."""


def make_llm_judge_scorer(
    judge_url: str,
    judge_model: str,
    api_key: str = "token-abc123",
) -> Callable:
    """Create a scorer that calls an external vLLM judge server.

    The judge server must be started separately (different GPU):
        vllm serve <model> --port <port>

    Returns:
        Callable(question, reference, response) -> dict with
        "score" (float) and "verdict" (str).
    """
    client = OpenAI(base_url=judge_url, api_key=api_key)

    def scorer(
        question: str,
        reference_answer: str,
        model_response: str,
        row_metadata: dict | None = None,
    ) -> dict:
        prompt = JUDGE_PROMPT.format(
            question=question,
            reference_answer=reference_answer,
            model_response=model_response,
        )
        resp = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20,
        )
        verdict = (resp.choices[0].message.content or "").strip().upper()

        if "PARTIALLY" in verdict:
            return {"score": 0.5, "verdict": "PARTIALLY_CORRECT"}
        elif "INCORRECT" in verdict:
            return {"score": 0.0, "verdict": "INCORRECT"}
        elif "CORRECT" in verdict:
            return {"score": 1.0, "verdict": "CORRECT"}
        return {"score": 0.0, "verdict": "UNKNOWN"}

    return scorer


def exact_match_scorer(
    question: str,
    reference_answer: str,
    model_response: str,
    row_metadata: dict | None = None,
) -> dict:
    """Score by checking if reference answer appears in model response."""
    match = reference_answer.strip().lower() in model_response.strip().lower()
    return {"score": 1.0 if match else 0.0}


# ── Eval data loading ──────────────────────────────────────────


def _load_single_eval_file(path: str, eval_type: str | None = None):
    """Load eval rows from a single path, tagging each with eval_type."""
    from data.core import DistillationRow

    if path.endswith(".json"):
        import json
        with open(path) as f:
            raw = json.load(f)
        if "test" in raw:
            # SDFT JSON — load test split directly (single file = single article)
            questions = raw["test"]["question"]
            answers = raw["test"]["answer"]
            return [
                DistillationRow(question=q, answer=a, eval_type=eval_type)
                for q, a in zip(questions, answers)
            ]
        elif "question" in raw and "answer" in raw and isinstance(raw["question"], list):
            # Flat parallel-lists format: {"question": [...], "answer": [...]}
            return [
                DistillationRow(question=q, answer=a, eval_type=eval_type)
                for q, a in zip(raw["question"], raw["answer"])
            ]
        else:
            # Generated rows.json
            return [
                DistillationRow(
                    question=r["question"],
                    answer=r.get("answer"),
                    documents=r.get("documents"),
                    system_prompt=r.get("system_prompt"),
                    eval_type=eval_type,
                )
                for r in raw
            ]
    raise ValueError(f"eval_data must be a .json file, got {path!r}")


def load_eval_data(eval_data_path: str):
    """Load eval data from one or more paths.

    Supports multiple sources via comma-separated ``type:path`` entries::

        "id:/path/to/test.json,ood:/path/to/ood.json"

    Each entry tags its rows with the given type (used for per-type W&B logging).
    A bare path (no colon prefix) gets eval_type=None.

    Returns list[DistillationRow].
    """
    parts = [p.strip() for p in eval_data_path.split(",") if p.strip()]
    all_rows = []
    for part in parts:
        if ":" in part and not part[0] == "/":
            eval_type, path = part.split(":", 1)
        else:
            eval_type, path = None, part
        all_rows.extend(_load_single_eval_file(path, eval_type=eval_type))
    return all_rows


# ── Scorer resolution ──────────────────────────────────────────


def _resolve_scorer(dotted_path: str, kwargs: dict | None = None) -> Callable:
    """Resolve a dotted import path to a scoring function.

    If the resolved object is a callable that returns a callable (factory),
    call it with kwargs to get the actual scorer.

    Examples:
        _resolve_scorer("evals.online_eval:exact_match_scorer")
        _resolve_scorer("evals.online_eval:make_llm_judge_scorer",
                        {"judge_url": "http://...", "judge_model": "Qwen/..."})
    """
    import importlib

    module_path, func_name = dotted_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    obj = getattr(module, func_name)

    # If kwargs provided, assume it's a factory function
    if kwargs:
        return obj(**kwargs)
    return obj


# ── Callback ───────────────────────────────────────────────────


class OnlineEvalCallback(TrainerCallback):
    """Generic online eval: generate via vLLM, score, log.

    Args:
        eval_data: List of DistillationRows with question + answer.
        scoring_fn: Callable(question, reference, response) -> dict
            with at least a "score" key (float 0-1).
        vllm_instance: Colocated vLLM LLM instance (shared with trainer
            for on-policy, or standalone for off-policy).
        model_name: Model name for tokenizer loading.
        eval_every_n_steps: Run eval every N training steps.
        max_new_tokens: Max generation length.
        temperature: Generation temperature (0.0 for greedy).
        output_dir: Directory to write eval_online.jsonl.
    """

    def __init__(
        self,
        *,
        groups: list | None = None,
        vllm_instance: LLM | None,
        model_name: str,
        # Multi-rank: non-main ranks may hold no engine at all (pass
        # vllm_instance=None) yet must still be given the callback, because the
        # weight sync below is a COLLECTIVE under FSDP/ZeRO-3.
        weight_sync_fn: Callable | None = None,
        # True when the engine is tensor-parallel across ranks: then generate()
        # is itself a collective and every rank must call it with identical
        # inputs. Only rank 0 logs the result.
        engine_is_collective: bool = False,
        manage_sleep: bool = True,
        batch_weight_sync: bool = False,
        eval_every_n_steps: int = 50,
        max_new_tokens: int = 1024,  # default unified with training caps
        temperature: float = 0.0,
        top_k: int | None = None,
        output_dir: str | None = None,
        # ── Back-compat shim (legacy single-group usage) ────────
        eval_data=None,
        scoring_fn: Callable | None = None,
        chat_template_kwargs: dict | None = None,
    ):
        if groups is None:
            if eval_data is None or scoring_fn is None:
                raise ValueError(
                    "OnlineEvalCallback requires either `groups=[...]` or both "
                    "`eval_data` and `scoring_fn` (legacy single-group usage)."
                )
            # Wrap legacy args into a single EvalGroup. Legacy scorers take
            # 3 args, so adapt to the 4-arg contract.
            def _legacy_shim(q, ref, resp, row_metadata):
                return scoring_fn(q, ref, resp)
            groups = [EvalGroup(
                rows=list(eval_data),
                scorer=_legacy_shim,
                metrics_prefix="eval",
                group_by=["eval_type"],
                chat_template_kwargs=chat_template_kwargs or {},
            )]

        self.groups = groups
        self.llm = vllm_instance
        self.weight_sync_fn = weight_sync_fn
        self.engine_is_collective = engine_is_collective
        self.manage_sleep = manage_sleep
        self.batch_weight_sync = batch_weight_sync
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.eval_every_n_steps = eval_every_n_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.output_dir = output_dir
        self._file = None
        self._rollouts_file = None
        self._last_evaluated_step = None

    def _sync_weights(self, model):
        """Sync training model weights to the colocated vLLM instance.

        When ``weight_sync_fn`` is supplied the caller owns this entirely —
        that is how the ensemble path gets in, because under FSDP/DeepSpeed the
        naive loop below is wrong: ``model.named_parameters()`` yields each
        rank's *shard*, and writing a shard into a full parameter corrupts the
        model silently. The trainer's ``_sync_module_to_side`` re-forms the
        full tensors first, and is a collective, so every rank must call this.

        The fallback path assumes unsharded parameters and
        ``distributed_executor_backend="external_launcher"``, so
        ``model_executor.driver_worker`` is available.
        """
        if self.weight_sync_fn is not None:
            self.weight_sync_fn(model)
            return
        if self.llm is None:
            return
        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        if self.batch_weight_sync:
            llm_model.load_weights(
                (name, param.detach()) for name, param in model.named_parameters()
            )
        else:
            for name, param in model.named_parameters():
                llm_model.load_weights([(name, param.data)])
        try:
            self.llm.llm_engine.reset_prefix_cache()
        except Exception:
            pass

        device = next(model.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _generate_for_group(self, group: "EvalGroup") -> list:
        """Render group prompts and generate via vLLM."""
        prompts = []
        ck = group.chat_template_kwargs or {}
        for row in group.rows:
            messages = []
            if row.system_prompt:
                messages.append({"role": "system", "content": row.system_prompt})
            messages.append({"role": "user", "content": row.question})
            prompts.append(self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **ck,
            ))

        gk = group.generation_kwargs or {}
        params = SamplingParams(
            temperature=gk.get("temperature", self.temperature),
            top_k=gk.get("top_k", self.top_k) or -1,
            max_tokens=gk.get("max_new_tokens", self.max_new_tokens),
        )
        outputs = self.llm.generate(prompts, sampling_params=params, use_tqdm=False)
        return [o.outputs[0].text.strip() for o in outputs]

    def _log_group(self, state, group: "EvalGroup", metrics: dict, records: list):
        """Log one group's metrics to stdout, wandb, and JSONL."""
        if hasattr(state, "active_training_seconds"):
            metrics["timing/active_training_seconds"] = state.active_training_seconds
        metrics["timing/evaluation_started_at_unix"] = getattr(self, "_evaluation_started_at_unix", None)
        metrics["timing/score_recorded_at_unix"] = time.time()
        prefix = group.metrics_prefix
        mean = metrics.get(f"{prefix}/mean_score", 0.0)
        n_total = metrics.get(f"{prefix}/n_total", len(records))
        print(
            f"\n[eval step={state.global_step} {prefix}] "
            f"mean={mean:.1%} n={n_total}",
            flush=True,
        )
        # Per-axis summary lines
        for key, val in sorted(metrics.items()):
            if key in (f"{prefix}/mean_score", f"{prefix}/n_total"):
                continue
            if key.endswith("/mean_score"):
                axis_label = key[len(prefix) + 1:-len("/mean_score")]
                print(f"  [{prefix}/{axis_label}] {val:.1%}", flush=True)

        try:
            import wandb
            if wandb.run is not None:
                # Profiling and trainer logs advance W&B's internal counter
                # independently of optimizer steps. An explicit step here can
                # therefore discard the evaluation as an out-of-order write.
                wandb.define_metric(f"{prefix}/*", step_metric="train/global_step")
                wandb.log({**metrics, "train/global_step": state.global_step})
        except ImportError:
            pass

        if self.output_dir:
            if self._file is None:
                os.makedirs(self.output_dir, exist_ok=True)
                path = os.path.join(self.output_dir, "eval_online.jsonl")
                self._file = open(path, "a")
            # One summary line per group per step.
            self._file.write(json.dumps({
                "step": state.global_step,
                "epoch": state.epoch,
                "prefix": prefix,
                **metrics,
            }) + "\n")
            self._file.flush()

            # Per-row rollouts (question, reference, response, score) — lets
            # us inspect how the model's answers evolve over training.
            # One file per run: eval_rollouts.jsonl with {step, prefix, idx,
            # question, reference, response, score, verdict?}.
            if self._rollouts_file is None:
                rpath = os.path.join(self.output_dir, "eval_rollouts.jsonl")
                self._rollouts_file = open(rpath, "a")
            for i, rec in enumerate(records):
                self._rollouts_file.write(json.dumps({
                    "step": state.global_step,
                    "epoch": state.epoch,
                    "prefix": prefix,
                    "idx": i,
                    "question": rec.get("question"),
                    "reference": rec.get("reference"),
                    "response": rec.get("response"),
                    "score": rec.get("score"),
                    "verdict": rec.get("verdict"),
                    "metadata": rec.get("metadata"),
                }) + "\n")
            self._rollouts_file.flush()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_every_n_steps == 0 or (
            control is not None and control.should_training_stop
        ):
            self.evaluate_now(args, state, control, model=model)
        return control

    def evaluate_now(self, args, state, control, model=None):
        """Evaluate this exact step once, including off-cadence final steps."""
        if model is None or self._last_evaluated_step == state.global_step:
            return control
        started_unix = time.time()
        started = time.perf_counter()
        try:
            return self._evaluate_now(args, state, control, model=model)
        finally:
            if getattr(state, "is_world_process_zero", True):
                from ssd.io_timing import record_io_timing

                record_io_timing(
                    self.output_dir, state.global_step, "evaluation_cycle",
                    started_at_unix=started_unix, finished_at_unix=time.time(),
                    seconds=time.perf_counter() - started,
                )

    def _evaluate_now(self, args, state, control, model=None):

        # Rank discipline: the engine is rank 0's, but the weight sync is a
        # collective under FSDP/ZeRO-3. So EVERY rank reaches _sync_weights,
        # and only rank 0 touches self.llm. Skipping the sync on non-main
        # ranks would hang the job on the first gather.
        has_engine = self.llm is not None
        is_main = getattr(state, "is_world_process_zero", True)
        evaluation_started = time.perf_counter()
        self._evaluation_started_at_unix = time.time()
        timings = {}

        # Wake vLLM + sync weights once per trigger (not per group).
        phase_started = time.perf_counter()
        if self.manage_sleep:
            torch.cuda.empty_cache()
        if has_engine and self.manage_sleep:
            try:
                getattr(self.llm, "wake_up_for_evaluation", self.llm.wake_up)()
            except Exception:
                pass
        timings["wake_seconds"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        self._sync_weights(model)
        timings["sync_seconds"] = time.perf_counter() - phase_started

        # When the engine is TP-sharded across ranks, generate() is collective:
        # every rank drives its own shard and skipping it elsewhere hangs the
        # job. Scoring and logging still happen once, on rank 0.
        all_metrics = {}
        timings.update(generation_seconds=0.0, scoring_seconds=0.0, logging_seconds=0.0)
        try:
            if has_engine and (is_main or self.engine_is_collective):
                for group in self.groups:
                    phase_started = time.perf_counter()
                    responses = self._generate_for_group(group)
                    generation_seconds = time.perf_counter() - phase_started
                    timings["generation_seconds"] += generation_seconds
                    if is_main:
                        phase_started = time.perf_counter()
                        metrics, records = aggregate_group_metrics(group, responses)
                        scoring_seconds = time.perf_counter() - phase_started
                        timings["scoring_seconds"] += scoring_seconds
                        prefix = group.metrics_prefix
                        metrics.update({f"{prefix}/generation_seconds": generation_seconds, f"{prefix}/scoring_seconds": scoring_seconds})
                        metrics.update({f"{prefix}/engine_{key}": value for key, value in getattr(self.llm, "last_eval_metrics", {}).items()})
                        phase_started = time.perf_counter()
                        self._log_group(state, group, metrics, records)
                        timings["logging_seconds"] += time.perf_counter() - phase_started
                        all_metrics.update(metrics)
        finally:
            phase_started = time.perf_counter()
            if has_engine and self.manage_sleep:
                try:
                    self.llm.sleep(level=1)
                except Exception:
                    pass
            timings["sleep_seconds"] = time.perf_counter() - phase_started
            timings["seconds"] = time.perf_counter() - evaluation_started
            if is_main:
                from ssd.io_timing import record_io_timing

                record_io_timing(self.output_dir, state.global_step, "evaluation",
                                 started_at_unix=self._evaluation_started_at_unix,
                                 finished_at_unix=time.time(), **timings)
                try:
                    import wandb

                    if wandb.run is not None:
                        wandb.log({"train/global_step": state.global_step, **{f"eval_timing/{key}": value for key, value in timings.items()}})
                except ImportError:
                    pass
        self._last_evaluated_step = state.global_step
        # HF's rotation already protects state.best_model_checkpoint. Online
        # evaluation bypasses Trainer.evaluate(), so register its result here.
        metric = getattr(args, "metric_for_best_model", None)
        if metric:
            from accelerate.utils import broadcast_object_list

            shared_metrics = [all_metrics]
            broadcast_object_list(shared_metrics, from_process=0)
            all_metrics = shared_metrics[0]
        if metric and metric in all_metrics:
            score = all_metrics[metric]
            best = getattr(state, "best_metric", None)
            greater = getattr(args, "greater_is_better", True)
            if best is None or (score > best if greater else score < best):
                state.best_metric = score
                state.best_global_step = state.global_step
                strategy = getattr(args, "save_strategy", "steps")
                if getattr(strategy, "value", strategy) != "no":
                    state.best_model_checkpoint = os.path.join(
                        args.output_dir, f"checkpoint-{state.global_step}"
                    )
                    control.should_save = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._rollouts_file is not None:
            self._rollouts_file.close()
            self._rollouts_file = None
