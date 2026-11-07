#!/usr/bin/env python3
"""Off-policy baselines: SFT and forward-KL distillation on frozen-teacher targets.

Both losses run in one manual training loop over a built dataset
(``src/data/build.py``). Before training, the frozen teacher samples one
completion for every prompt through the evaluation vLLM engine (loaded with
the teacher weights for the duration); every epoch then trains on that fixed
set of teacher completions.

* ``loss=sft``: token-mean cross-entropy on the completion.
* ``loss=fkl``: forward KL between the teacher's and the student's next-token
  distributions on every completion token (the teacher sees the gold answer in
  its prompt, the student does not).

Values are pydra ``key=value`` overrides. ``scripts/launch.py`` composes them
from the YAML files in ``configs/``; append ``--show`` to preview a config.
"""

import json
import os
import random
import time

import pydra
import torch
import torch.nn.functional as F
from data.core import load_built_dataset
from ssd.kl_utils import (
    batch_forward_slice as _batch_forward_slice,
)
from ssd.kl_utils import (
    compute_fkl_loss as _compute_fkl_loss,
)
from ssd.kl_utils import (
    tokenize_for_fkl as _tokenize_for_fkl,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler


def _setup_eval_vllm_and_callback(
    cfg,
    model_name: str,
    *,
    max_model_len: int | None = None,
):
    """Set up distributed env, colocated vLLM, and online eval callback.

    Builds the evaluation group from cfg.eval_data + cfg.eval_scoring.
    Returns (None, None) when no evaluation is configured.

    Must be called BEFORE any Trainer/Accelerator initialization so the
    env vars are picked up correctly.
    """
    from evals.online_eval import (
        EvalGroup,
        OnlineEvalCallback,
        _resolve_scorer,
        load_eval_data,
    )

    groups: list = []

    if cfg.eval_data and cfg.eval_scoring:
        generic_rows = load_eval_data(cfg.eval_data)
        generic_scorer = _resolve_scorer(cfg.eval_scoring, cfg.eval_scoring_kwargs)

        def _generic_shim(q, ref, resp, md):
            return generic_scorer(q, ref, resp)

        groups.append(EvalGroup(
            rows=generic_rows,
            scorer=_generic_shim,
            metrics_prefix="eval",
            group_by=["eval_type"],
            chat_template_kwargs=cfg.chat_template_kwargs or {},
        ))
        print(f"  Generic online eval: {len(generic_rows)} questions", flush=True)

    if not groups:
        return None, None

    from trl.trainer.utils import ensure_master_addr_port
    from vllm import LLM

    # Single-process rendezvous variables for vLLM's external launcher.
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    ensure_master_addr_port()

    eval_llm = LLM(
        model=model_name,
        gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        kv_cache_memory_bytes=cfg.vllm_kv_cache_memory_bytes,
        distributed_executor_backend="external_launcher",
        enable_sleep_mode=True,
        max_num_seqs=cfg.eval_max_num_seqs,
        max_num_batched_tokens=4096,
        max_model_len=max(max_model_len, cfg.eval_max_model_len or 0),
    )
    eval_llm.sleep(level=1)

    eval_cb = OnlineEvalCallback(
        groups=groups,
        vllm_instance=eval_llm,
        batch_weight_sync=cfg.vllm_batch_weight_sync,
        model_name=model_name,
        eval_every_n_steps=cfg.eval_every_n_steps,
        max_new_tokens=cfg.eval_max_new_tokens,
        output_dir=cfg.output_dir,
    )
    print(f"  Online eval ON: "
          f"{sum(len(g.rows) for g in groups)} rows across "
          f"{len(groups)} group(s) "
          f"({', '.join(g.metrics_prefix for g in groups)})",
          flush=True)
    return eval_cb, eval_llm


class OffPolicyConfig(pydra.Config):
    def __init__(self) -> None:
        super().__init__()

        # ── Data and model ───────────────────────────────────────
        self.dataset = pydra.REQUIRED  # built by src/data/build.py
        self.loss = "sft"  # "sft" | "fkl"
        self.model_name = "Qwen/Qwen3-4B-Instruct-2507"
        self.attn_implementation = "sdpa"
        self.output_dir = pydra.REQUIRED
        self.seed = 42
        self.shuffle_dataset = True

        # ── Optimization ─────────────────────────────────────────
        self.learning_rate = 5e-5
        self.warmup_ratio = 0.1
        self.lr_scheduler_type = "constant_with_warmup"
        self.max_grad_norm = 1.0
        self.weight_decay = 0.0
        self.num_train_epochs = 1
        self.per_device_train_batch_size = 4
        self.gradient_accumulation_steps = 2
        # Every epoch runs the same number of optimizer steps.
        self.drop_last_effective_batch = True
        self.gradient_checkpointing = False
        self.stop_after_steps = 0  # stop early without shortening the LR schedule

        # ── Sequence lengths and tokenization ────────────────────
        self.max_student_len = 4608  # student prompt + completion, tokens
        self.max_teacher_len = 4608  # teacher prompt + completion, tokens
        self.temperature = 1.0  # softmax temperature for the FKL
        self.chat_template_kwargs: dict = {}  # e.g. {"enable_thinking": False}

        # ── Teacher targets (sampled once, before training) ──────
        self.target_temperature = 1.0
        self.target_top_k = 20
        self.target_top_p = 1.0
        self.target_max_new_tokens = 512
        self.debug_max_rows = 0  # smoke tests only: truncate the dataset

        # ── Online evaluation (its vLLM engine also samples the targets) ──
        self.eval_data: str | None = None
        self.eval_scoring: str | None = None  # e.g. "evals.online_eval:make_llm_judge_scorer"
        self.eval_scoring_kwargs: dict = {}
        self.eval_every_n_steps = 25
        self.eval_max_new_tokens = 512
        self.eval_max_num_seqs = 64  # vLLM batch width for evaluation and target sampling
        self.eval_max_model_len: int | None = None  # defaults to max_teacher_len
        self.vllm_gpu_memory_utilization = 0.3
        self.vllm_kv_cache_memory_bytes: int | None = None
        self.vllm_batch_weight_sync = True

        # ── Checkpoints and run records ──────────────────────────
        self.save_strategy = "no"  # "no" | "epochs"
        self.save_epochs = 1
        self.save_final_model = False
        self.record_step_timing = False
        self.verify_frozen_teacher = False  # assert the teacher is unchanged at the end
        self.record_student_fingerprint = False

        # ── Logging ──────────────────────────────────────────────
        self.report_to = "wandb"
        self.run_name: str | None = None
        self.wandb_project = "speculative-self-distillation"
        self.wandb_group: str | None = None
        self.logging_steps = 1

    def finalize(self):
        if self.loss not in ("sft", "fkl"):
            raise ValueError(f"loss must be 'sft' or 'fkl', got {self.loss!r}")
        if not (self.eval_data and self.eval_scoring):
            raise ValueError("teacher targets are sampled through the evaluation engine: set eval_data and eval_scoring")


def _compute_nll_loss(student_logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Token-mean cross-entropy on the completion slice (SFT in the manual loop)."""
    return F.cross_entropy(student_logits.float(), target_ids)


def _completion_targets(rows: list[dict], device) -> torch.Tensor:
    ids = []
    for r in rows:
        cs, nc = r["completion_start_student"], r["num_completion_tokens"]
        ids.extend(r["student_ids"][cs: cs + nc])
    return torch.tensor(ids, dtype=torch.long, device=device)


def _tokenize_rows(dataset, answers, tokenizer, cfg) -> list[dict]:
    tokenized = []
    for row, answer in zip(dataset, answers):
        row = dict(row)
        row["answer"] = answer
        t = _tokenize_for_fkl(row, tokenizer, cfg.max_student_len, cfg.max_teacher_len,
                              chat_template_kwargs=cfg.chat_template_kwargs)
        if t is not None:
            tokenized.append(t)
    return tokenized


def _sample_teacher_targets(cfg, dataset, tokenizer, eval_llm, teacher, *, wandb_run):
    """Sample one teacher completion per row with the evaluation engine.

    The engine normally carries the student's weights for evaluation; here the
    frozen teacher's weights are loaded into it for the sampling (from the
    resident HF teacher when FKL holds one, else from the pinned snapshot on
    disk), and the next evaluation's weight sync restores the student. The
    phase is timed and recorded in teacher-targets.jsonl so time-axis plots
    can put the data-generation cost on the training clock.
    """
    import glob

    from vllm import SamplingParams

    started = time.perf_counter()
    started_unix = time.time()
    torch.cuda.empty_cache()
    getattr(eval_llm, "wake_up_for_evaluation", eval_llm.wake_up)()
    llm_model = eval_llm.llm_engine.model_executor.driver_worker.model_runner.model
    if teacher is not None:
        llm_model.load_weights((name, p.detach()) for name, p in teacher.named_parameters())
    else:
        from safetensors import safe_open

        shards = sorted(glob.glob(os.path.join(cfg.model_name, "*.safetensors")))
        if not shards:
            raise FileNotFoundError(f"no safetensors shards under {cfg.model_name}")
        for shard in shards:
            with safe_open(shard, framework="pt", device="cuda") as handle:
                llm_model.load_weights((key, handle.get_tensor(key)) for key in handle.keys())
    try:
        eval_llm.llm_engine.reset_prefix_cache()
    except Exception:
        pass
    torch.cuda.synchronize()
    loaded = time.perf_counter()
    prompts = []
    for row in dataset:
        ck = cfg.chat_template_kwargs or row.get("chat_template_kwargs") or {}
        prompts.append(tokenizer.apply_chat_template(
            row["teacher_prompt"], tokenize=False, add_generation_prompt=True, **ck,
        ))
    params = SamplingParams(
        temperature=cfg.target_temperature,
        top_k=cfg.target_top_k or -1,
        top_p=cfg.target_top_p,
        max_tokens=cfg.target_max_new_tokens,
        seed=cfg.seed * 1000,
    )
    outputs = eval_llm.generate(prompts, sampling_params=params, use_tqdm=False)
    answers = [o.outputs[0].text for o in outputs]
    completion_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    truncated = sum(1 for o in outputs if o.outputs[0].finish_reason == "length")
    generated = time.perf_counter()
    eval_llm.sleep(level=1)
    seconds = time.perf_counter() - started
    record = {
        "started_at_unix": started_unix,
        "finished_at_unix": time.time(), "seconds": seconds,
        "weight_load_seconds": loaded - started, "generation_seconds": generated - loaded,
        "rows": len(answers), "completion_tokens": completion_tokens,
        "mean_completion_tokens": completion_tokens / max(1, len(answers)),
        "truncated_rows": truncated, "empty_rows": sum(1 for a in answers if not a.strip()),
        "sampling": {"temperature": cfg.target_temperature, "top_k": cfg.target_top_k,
                     "top_p": cfg.target_top_p, "max_tokens": cfg.target_max_new_tokens,
                     "seed": cfg.seed * 1000},
    }
    with open(os.path.join(cfg.output_dir, "teacher-targets.jsonl"), "w") as stream:
        stream.write(json.dumps(record) + "\n")
    print(f"  Sampled {len(answers)} teacher targets: "
          f"{completion_tokens} tokens, {seconds:.0f}s ({truncated} truncated)", flush=True)
    if wandb_run:
        wandb_run.log({"train/global_step": 0, "targets/sampling_seconds": seconds,
                       "targets/completion_tokens": completion_tokens,
                       "targets/mean_completion_tokens": record["mean_completion_tokens"],
                       "targets/truncated_rows": truncated})
    return answers, record


def _train(cfg: OffPolicyConfig, dataset, metadata: dict) -> None:
    """Manual off-policy loop: SFT (cross-entropy) or forward KL on completion tokens."""
    nll = cfg.loss == "sft"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.debug_max_rows:
        dataset = dataset.select(range(min(cfg.debug_max_rows, len(dataset))))
        print(f"  debug_max_rows: truncated dataset to {len(dataset)} rows", flush=True)

    expected_cols = {"prompt", "teacher_prompt", "answer"}
    actual_cols = set(dataset.column_names)
    if not expected_cols.issubset(actual_cols):
        raise ValueError(
            f"Dataset has columns {actual_cols}, but off-policy training needs "
            f"{expected_cols}. Rebuild it with src/data/build.py."
        )

    # Wandb
    wandb_run = None
    if cfg.report_to == "wandb":
        import wandb
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)
        dataset_name = metadata.get("dataset", os.path.basename(cfg.dataset))
        run_name = cfg.run_name or f"{cfg.loss}_{dataset_name}"
        wandb_group = cfg.wandb_group or metadata.get("dataset", os.path.basename(cfg.dataset))
        wandb_run = wandb.init(project=cfg.wandb_project, name=run_name, group=wandb_group)

    # Load models
    print(f"Loading student: {cfg.model_name}", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        dtype=torch.bfloat16,
        attn_implementation=cfg.attn_implementation,
    ).to(device)
    if cfg.gradient_checkpointing:
        student.gradient_checkpointing_enable()
        if hasattr(student, "config"):
            student.config.use_cache = False
        print("  Gradient checkpointing enabled on student", flush=True)

    print(f"Loading teacher: {cfg.model_name}", flush=True)
    teacher = None
    if not nll:
        teacher = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            dtype=torch.bfloat16,
            attn_implementation=cfg.attn_implementation,
        ).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from types import SimpleNamespace

    eval_cb, eval_llm = _setup_eval_vllm_and_callback(
        cfg, cfg.model_name, max_model_len=cfg.max_teacher_len,
    )
    if eval_llm is None:
        raise ValueError("teacher targets are sampled through the evaluation vLLM engine: set eval_data and eval_scoring")

    # Sample the teacher targets once; every epoch trains on this set.
    answers, target_record = _sample_teacher_targets(
        cfg, dataset, tokenizer, eval_llm, teacher, wandb_run=wandb_run,
    )
    tokenized = _tokenize_rows(dataset, answers, tokenizer, cfg)
    print(f"  Tokenized: {len(tokenized)} rows (dropped {len(dataset) - len(tokenized)})", flush=True)
    if not tokenized:
        raise ValueError("No completion tokens survived tokenization")

    # Effective batch = per_device x grad_accum rows per optimizer step; each
    # effective batch is split into per_device micro-batches.
    effective_batch = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    epoch_rows = (len(tokenized) // effective_batch * effective_batch
                  if cfg.drop_last_effective_batch else len(tokenized))
    total_steps = ((epoch_rows + effective_batch - 1) // effective_batch) * cfg.num_train_epochs
    if total_steps <= 0:
        raise ValueError("Learning-rate schedule must contain positive steps")
    scheduler = get_scheduler(
        cfg.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * cfg.warmup_ratio),
        num_training_steps=total_steps,
    )

    dataset_name = metadata.get("dataset", os.path.basename(cfg.dataset))
    run_name = cfg.run_name or f"{cfg.loss}_{dataset_name}"
    banner = (
        f"\n{'='*50}\n"
        f"  loss:               {cfg.loss}\n"
        f"  dataset:            {cfg.dataset}\n"
        f"  model:              {cfg.model_name}\n"
        f"  train_rows:         {len(tokenized)}\n"
        f"  epochs:             {cfg.num_train_epochs}\n"
        f"  lr:                 {cfg.learning_rate}\n"
        f"  per_device_batch:   {cfg.per_device_train_batch_size}\n"
        f"  grad_accum:         {cfg.gradient_accumulation_steps}\n"
        f"  effective_batch:    {effective_batch}\n"
        f"  temperature:        {cfg.temperature}\n"
        f"  total_steps:        {total_steps}\n"
        f"  output_dir:         {cfg.output_dir}\n"
        f"  run_name:           {run_name}\n"
    )
    banner += f"  teacher targets:    {len(tokenized)} sampled once ({target_record['seconds']:.0f}s)\n"
    banner += f"{'='*50}\n"
    print(banner, flush=True)

    os.makedirs(cfg.output_dir, exist_ok=True)

    # Training loop
    global_step = 0
    accum_loss = 0.0
    accum_count = 0
    rng = random.Random(cfg.seed)

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    from collections import defaultdict

    from ssd.experiment_timing import ExperimentTiming, verify_frozen_teacher

    timing_args = SimpleNamespace(output_dir=cfg.output_dir)
    timing_state = SimpleNamespace(global_step=0, epoch=0, max_steps=total_steps)
    timing_trainer = SimpleNamespace(
        args=timing_args, state=timing_state, model=student, ref_model=teacher,
        accelerator=SimpleNamespace(backward=lambda loss: loss.backward()),
        _clip_grad_norm=lambda: torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.max_grad_norm),
        _metrics={"train": defaultdict(list)}, _cum_completion_tokens=0,
    )
    timing = ExperimentTiming(timing_trainer) if cfg.record_step_timing else None

    for epoch in range(cfg.num_train_epochs):
        indices = list(range(len(tokenized)))
        if cfg.shuffle_dataset:
            rng.shuffle(indices)
        indices = indices[:epoch_rows]
        student.train()
        t0 = time.time()

        # Iterate in effective-batch chunks; each chunk is split into
        # per_device micro-batches with gradient accumulation so a real batched
        # forward pass is used within each micro-batch (fits GPU memory) while
        # the effective batch drives one optimizer step.
        micro_bs = cfg.per_device_train_batch_size
        for bstart in range(0, len(indices), effective_batch):
            if timing is not None:
                timing.on_step_begin(timing_args, timing_state, None)
            batch_idx = indices[bstart: bstart + effective_batch]
            n_micro = (len(batch_idx) + micro_bs - 1) // micro_bs

            step_loss = 0.0

            for mstart in range(0, len(batch_idx), micro_bs):
                micro_idx = batch_idx[mstart: mstart + micro_bs]
                rows = [tokenized[j] for j in micro_idx]

                student_ids_list = [r["student_ids"] for r in rows]
                teacher_ids_list = [r["teacher_ids"] for r in rows]
                cs_list = [r["completion_start_student"] for r in rows]
                ct_list = [r["completion_start_teacher"] for r in rows]
                nc_list = [r["num_completion_tokens"] for r in rows]
                timing_trainer._cum_completion_tokens += sum(nc_list)
                loss_handle = timing.timer.start("loss_total") if timing is not None else None

                # Batched student forward (grad) on a small micro-batch
                student_logits = _batch_forward_slice(
                    student, student_ids_list, cs_list, nc_list, pad_id, device,
                    with_grad=True,
                )
                teacher_logits = None
                if not nll:
                    teacher_logits = _batch_forward_slice(
                        teacher, teacher_ids_list, ct_list, nc_list, pad_id, device,
                        with_grad=False,
                    )

                if nll:
                    micro_loss = _compute_nll_loss(student_logits, _completion_targets(rows, device))
                else:
                    micro_loss = _compute_fkl_loss(
                        student_logits, teacher_logits, cfg.temperature,
                    )

                if timing is not None:
                    timing.timer.finish(loss_handle)
                timing_trainer.accelerator.backward(micro_loss / n_micro)
                step_loss += micro_loss.item() / n_micro

                del student_logits, teacher_logits, micro_loss

            accum_loss += step_loss
            accum_count += 1

            timing_trainer._clip_grad_norm()
            if timing is not None:
                timing.on_pre_optimizer_step(timing_args, timing_state, None)
            optimizer.step()
            if timing is not None:
                timing.on_optimizer_step(timing_args, timing_state, None)
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            timing_state.global_step = global_step
            timing_state.epoch = epoch + (bstart + len(batch_idx)) / len(indices)
            if timing is not None:
                timing.on_step_end(timing_args, timing_state, None)
                timing_trainer._metrics["train"].clear()

            # Online evaluation fires on its own step cadence.
            if eval_cb is not None:
                control = SimpleNamespace(should_training_stop=(global_step == total_steps or
                    bool(cfg.stop_after_steps and global_step >= cfg.stop_after_steps)))
                eval_cb.on_step_end(args=None, state=timing_state, control=control, model=student)

            if global_step % cfg.logging_steps == 0:
                avg_loss = accum_loss / accum_count if accum_count > 0 else 0
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                samples_done = bstart + len(batch_idx)
                speed = samples_done / elapsed if elapsed > 0 else 0

                log_str = (
                    f"  epoch {epoch+1}/{cfg.num_train_epochs} | "
                    f"step {global_step}/{total_steps} | "
                    f"loss={avg_loss:.4f} | lr={lr:.2e} | "
                    f"{speed:.1f} samples/s"
                )
                wandb_dict = {
                    f"train/{cfg.loss}_loss": avg_loss,
                    "train/learning_rate": lr,
                    "train/epoch": epoch + samples_done / len(indices),
                    "train/global_step": global_step,
                }
                wandb_dict["targets/sampling_seconds"] = target_record["seconds"]

                print(log_str, flush=True)
                if wandb_run:
                    wandb_run.log(wandb_dict)
                accum_loss = 0.0
                accum_count = 0

            if cfg.stop_after_steps and global_step >= cfg.stop_after_steps:
                break

        print(f"  Epoch {epoch+1} done ({time.time() - t0:.0f}s)", flush=True)

        if cfg.save_strategy != "no" and (epoch + 1) % cfg.save_epochs == 0:
            ckpt_dir = os.path.join(cfg.output_dir, f"checkpoint-epoch{epoch+1}")
            student.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"  Saved: {ckpt_dir}", flush=True)
        if cfg.stop_after_steps and global_step >= cfg.stop_after_steps:
            break

    final_dir = os.path.join(cfg.output_dir, "final")
    if cfg.save_final_model:
        student.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"Final model saved to {final_dir}", flush=True)
    if timing is not None:
        timing.on_train_end(timing_args, timing_state, None)
    if cfg.verify_frozen_teacher and teacher is not None:
        verify_frozen_teacher(timing_trainer, cfg.model_name)
    if cfg.record_student_fingerprint:
        from ssd.experiment_timing import record_student_fingerprint
        record_student_fingerprint(timing_trainer)
    timing_state.target_sampling_seconds = target_record["seconds"]
    with open(os.path.join(cfg.output_dir, "completion-state.json"), "w") as stream:
        json.dump(vars(timing_state), stream, indent=2)

    if eval_cb is not None:
        eval_cb.on_train_end(args=None, state=None, control=None)
    if eval_llm is not None:
        torch.cuda.empty_cache()
        eval_llm.wake_up()
        eval_llm.llm_engine.engine_core.shutdown()

    if wandb_run:
        wandb_run.finish()


@pydra.main(OffPolicyConfig)
def main(cfg: OffPolicyConfig) -> None:
    from transformers import set_seed

    set_seed(cfg.seed)
    dataset, metadata = load_built_dataset(cfg.dataset)
    _train(cfg, dataset, metadata)


if __name__ == "__main__":
    main()
