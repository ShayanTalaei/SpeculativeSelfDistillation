#!/usr/bin/env python3
"""On-policy and speculative self-distillation (SSD) trainer.

Two engines, one trainer family:

* ``engine=vllm`` (on-policy): the student samples every completion itself
  with a colocated vLLM and is trained against the teacher distribution on
  every token (forward KL with ``alpha=0``).
* ``engine=paired_vllm`` (SSD): an isolated two-GPU lockstep vLLM pair
  generates the completion. Each token is drafted by the student and handed to
  the teacher when the student/teacher divergence exceeds the routing
  threshold; the teacher's hidden states stay resident on the student GPU and
  are projected through the teacher head inside the loss.

Values are pydra ``key=value`` overrides. ``scripts/launch.py`` composes them
from the YAML files in ``configs/``; append ``--show`` to preview a config.
"""

import os
import time
from pathlib import Path

import pydra
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.core import load_built_dataset

ROOT = Path(__file__).resolve().parents[1]


class TrainConfig(pydra.Config):
    def __init__(self) -> None:
        super().__init__()

        # ── Data and model ───────────────────────────────────────
        self.dataset = pydra.REQUIRED  # built by src/data/build.py
        self.engine = "paired_vllm"  # "vllm" (on-policy) | "paired_vllm" (SSD)
        self.model_name = "Qwen/Qwen3-4B-Instruct-2507"
        self.attn_implementation = "sdpa"
        self.output_dir = pydra.REQUIRED
        self.seed = 42
        self.shuffle_dataset = True

        # ── Loss ─────────────────────────────────────────────────
        self.alpha = 0.0  # 0 = forward KL, 1 = reverse KL, 0.5 = JSD
        self.loss_mask_strategy = "all"  # all | teacher_only | student_only | directional
        self.num_loss_tokens_to_skip = 0

        # ── Token routing (SSD only) ─────────────────────────────
        self.routing_strategy = "pure_student"  # pure_student (on-policy) | kl_switch | jsd_switch
        self.kl_threshold = 1.0
        self.kl_direction = "reverse"
        self.jsd_threshold = 0.5  # JSD in [0, log 2]

        # ── Generation ───────────────────────────────────────────
        self.temperature = 1.0
        self.top_k = 20
        self.top_p = 1.0
        self.max_prompt_length = 2048
        self.max_completion_length = 1024
        self.rollout_max_new_tokens = 1024  # SSD rollout cap; must equal max_completion_length
        self.per_device_train_batch_size = 1
        self.gradient_accumulation_steps = 8
        self.steps_per_generation = 8  # one rollout batch per optimizer step
        self.num_generations = 1
        self.num_iterations = 1

        # ── Optimization ─────────────────────────────────────────
        self.learning_rate = 5e-5
        self.warmup_ratio = 0.1
        self.lr_scheduler_type = "cosine"
        self.max_grad_norm = 1.0
        self.optim = "adamw_torch"
        self.num_train_epochs = 1
        self.max_steps = -1  # -1 = full epochs; otherwise cap at this many optimizer steps
        self.stop_after_steps = 0  # stop early without shortening the LR schedule
        self.gradient_checkpointing = False

        # ── Teacher ──────────────────────────────────────────────
        # sync_ref_model=False keeps the teacher frozen at the initial weights;
        # True makes it an EMA of the student (teacher <- (1-a) teacher + a student
        # every ref_model_sync_steps optimizer steps).
        self.sync_ref_model = False
        self.ref_model_mixup_alpha = 0.01
        self.ref_model_sync_steps = 1
        self.verify_frozen_teacher = False  # assert the teacher is unchanged at the end
        self.persist_ref_model = False

        # ── Checkpoints and run records ──────────────────────────
        self.save_strategy = "no"
        self.save_steps = 200
        self.save_total_limit: int | None = None
        self.save_only_model = False
        self.save_final_model = False
        self.resume_from_checkpoint: str | None = None  # path or "auto"
        self.record_step_timing = False
        self.record_student_fingerprint = False

        # ── On-policy vLLM (engine=vllm) ─────────────────────────
        self.vllm_gpu_memory_utilization = 0.3
        self.vllm_enable_sleep_mode = False
        self.vllm_batch_weight_sync = True
        self.vllm_verify_weight_sync = False
        self.vllm_kv_cache_memory_bytes: int | None = None
        self.vllm_importance_sampling_correction = True

        # ── Lockstep paired vLLM (engine=paired_vllm) ────────────
        self.paired_python = str(ROOT / ".venv/bin/python")
        self.paired_student_gpu = "0"
        self.paired_teacher_gpu = "1"
        self.paired_capacity = 65536
        self.paired_token_budget = 16384
        self.paired_scratch_dir: str | None = None  # default <output_dir>/paired-scratch

        # ── Online evaluation ────────────────────────────────────
        self.eval_data: str | None = None
        self.eval_scoring: str | None = None  # e.g. "evals.scorers:science_xml_match_scorer"
        self.eval_scoring_kwargs: dict = {}
        self.eval_every_n_steps = 100
        self.eval_max_new_tokens = 1024  # must equal max_completion_length
        self.eval_chat_template_kwargs: dict | None = None  # None = same as the dataset rows

        # ── Logging ──────────────────────────────────────────────
        self.report_to = "wandb"
        self.run_name: str | None = None
        self.wandb_project = "speculative-self-distillation"
        self.wandb_group: str | None = None
        self.logging_steps = 1
        self.log_completions = False

    def finalize(self):
        if self.engine not in ("vllm", "paired_vllm"):
            raise ValueError(f"engine must be 'vllm' or 'paired_vllm', got {self.engine!r}")
        if self.engine == "vllm" and self.routing_strategy != "pure_student":
            raise ValueError("engine=vllm is on-policy: routing_strategy must be pure_student")
        if self.engine == "paired_vllm":
            self.persist_ref_model = True
            if self.routing_strategy not in ("kl_switch", "jsd_switch"):
                raise ValueError("engine=paired_vllm routes tokens: routing_strategy must be kl_switch or jsd_switch")
            if self.top_p != 1.0:
                raise ValueError("paired_vllm samples with top_k only; set top_p=1.0")
        caps = {self.max_completion_length, self.eval_max_new_tokens}
        if self.engine == "paired_vllm":
            caps.add(self.rollout_max_new_tokens)
        if len(caps) > 1:
            raise ValueError(
                "Training and evaluation generation caps must match: "
                f"max_completion_length={self.max_completion_length}, "
                f"rollout_max_new_tokens={self.rollout_max_new_tokens}, "
                f"eval_max_new_tokens={self.eval_max_new_tokens}"
            )


@pydra.main(TrainConfig)
def main(cfg: TrainConfig) -> None:
    os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)

    if cfg.engine == "paired_vllm":
        # The paired trainer does not own a TRL vLLM; the evaluator still
        # expects the single-process rendezvous variables to exist.
        from trl.trainer.utils import ensure_master_addr_port

        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        ensure_master_addr_port()

    train_dataset, metadata = load_built_dataset(cfg.dataset)
    wandb_group = cfg.wandb_group or metadata.get("dataset", os.path.basename(cfg.dataset))
    os.environ["WANDB_RUN_GROUP"] = wandb_group

    expected_cols = {"prompt", "teacher_prompt", "chat_template_kwargs"}
    if not expected_cols.issubset(train_dataset.column_names):
        raise ValueError(
            f"Dataset at {cfg.dataset} has columns {set(train_dataset.column_names)}, "
            f"but training needs {expected_cols}. Rebuild it with src/data/build.py."
        )
    # Evaluation renders prompts with the same chat-template kwargs as the
    # rollouts (e.g. enable_thinking=False), unless overridden.
    dataset_chat_template_kwargs = {}
    if len(train_dataset) and train_dataset[0].get("chat_template_kwargs"):
        dataset_chat_template_kwargs = dict(train_dataset[0]["chat_template_kwargs"])

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, dtype=torch.bfloat16, attn_implementation=cfg.attn_implementation,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, dtype=torch.bfloat16, attn_implementation=cfg.attn_implementation,
    )
    if not cfg.sync_ref_model:
        ref_model.requires_grad_(False)
        ref_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    dataset_name = metadata.get("dataset", os.path.basename(cfg.dataset))
    method = "ssd" if cfg.engine == "paired_vllm" else "onpolicy"
    run_name = cfg.run_name or f"{dataset_name}_{method}_s{cfg.seed}"

    common = dict(
        seed=cfg.seed,
        generate_from_teacher=False,
        alpha=cfg.alpha,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        optim=cfg.optim,
        logging_steps=cfg.logging_steps,
        bf16=True,
        fp16=False,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        steps_per_generation=cfg.steps_per_generation,
        num_generations=cfg.num_generations,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        num_train_epochs=cfg.num_train_epochs,
        num_iterations=cfg.num_iterations,
        max_steps=cfg.max_steps,
        save_steps=cfg.save_steps,
        save_strategy=cfg.save_strategy,
        save_total_limit=cfg.save_total_limit,
        save_only_model=cfg.save_only_model,
        persist_ref_model=cfg.persist_ref_model,
        gradient_checkpointing=cfg.gradient_checkpointing,
        max_grad_norm=cfg.max_grad_norm,
        report_to=cfg.report_to,
        output_dir=cfg.output_dir,
        run_name=run_name,
        eval_strategy="no",
        sync_ref_model=cfg.sync_ref_model,
        ref_model_sync_steps=cfg.ref_model_sync_steps,
        ref_model_mixup_alpha=cfg.ref_model_mixup_alpha,
        log_completions=cfg.log_completions,
        num_loss_tokens_to_skip=cfg.num_loss_tokens_to_skip,
        shuffle_dataset=cfg.shuffle_dataset,
    )

    if cfg.engine == "vllm":
        from distil_config import DistilConfig
        from distil_trainer import DistilTrainer

        trainer_args = DistilConfig(
            use_vllm=True,
            vllm_mode="colocate",
            vllm_tensor_parallel_size=1,
            vllm_enforce_eager=False,
            vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
            vllm_enable_sleep_mode=cfg.vllm_enable_sleep_mode,
            vllm_batch_weight_sync=cfg.vllm_batch_weight_sync,
            vllm_verify_weight_sync=cfg.vllm_verify_weight_sync,
            vllm_kv_cache_memory_bytes=cfg.vllm_kv_cache_memory_bytes,
            vllm_importance_sampling_correction=cfg.vllm_importance_sampling_correction,
            top_p=cfg.top_p,
            **common,
        )
        trainer_kwargs = {}
        TrainerClass = DistilTrainer
    else:
        from ssd.config import SSDConfig
        from ssd.distil_trainer import SSDDistilTrainer

        trainer_args = SSDConfig(
            use_vllm=False,
            routing_strategy=cfg.routing_strategy,
            kl_threshold=cfg.kl_threshold,
            kl_direction=cfg.kl_direction,
            jsd_threshold=cfg.jsd_threshold,
            loss_mask_strategy=cfg.loss_mask_strategy,
            log_rollouts=False,
            paired_python=cfg.paired_python,
            paired_student_gpu=cfg.paired_student_gpu,
            paired_teacher_gpu=cfg.paired_teacher_gpu,
            paired_capacity=cfg.paired_capacity,
            paired_token_budget=cfg.paired_token_budget,
            **common,
        )
        trainer_kwargs = {
            "model_path_a": cfg.model_name,
            "model_path_b": cfg.model_name,
            "paired_scratch_dir": cfg.paired_scratch_dir
            or os.path.join(cfg.output_dir, "paired-scratch"),
            "paired_max_model_len": cfg.max_prompt_length + cfg.max_completion_length,
            "vllm_gpu_memory_utilization": cfg.vllm_gpu_memory_utilization,
        }
        TrainerClass = SSDDistilTrainer

    loss_name = {0.0: "forward KL", 0.5: "JSD", 1.0: "reverse KL"}.get(cfg.alpha, f"alpha={cfg.alpha}")
    print(
        f"\n{'=' * 50}\n"
        f"  method:             {method} (engine={cfg.engine})\n"
        f"  dataset:            {cfg.dataset} ({len(train_dataset)} rows)\n"
        f"  model:              {cfg.model_name}\n"
        f"  routing:            {cfg.routing_strategy}"
        + (f" kl_{cfg.kl_direction}>{cfg.kl_threshold}" if cfg.routing_strategy == "kl_switch" else "")
        + (f" jsd>{cfg.jsd_threshold}" if cfg.routing_strategy == "jsd_switch" else "")
        + "\n"
        f"  loss:               {loss_name} on {cfg.loss_mask_strategy} tokens\n"
        f"  teacher:            {'EMA alpha=' + str(cfg.ref_model_mixup_alpha) if cfg.sync_ref_model else 'frozen'}\n"
        f"  batch:              {cfg.per_device_train_batch_size} x {cfg.gradient_accumulation_steps}\n"
        f"  lr / schedule:      {cfg.learning_rate} / {cfg.lr_scheduler_type} (warmup {cfg.warmup_ratio})\n"
        f"  epochs / max_steps: {cfg.num_train_epochs} / {cfg.max_steps}\n"
        f"  prompt / completion:{cfg.max_prompt_length} / {cfg.max_completion_length}\n"
        f"  output_dir:         {cfg.output_dir}\n"
        f"  run_name:           {run_name}\n"
        f"{'=' * 50}\n",
        flush=True,
    )

    trainer = TrainerClass(
        model=model,
        ref_model=ref_model,
        args=trainer_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        **trainer_kwargs,
    )
    if cfg.record_step_timing:
        from ssd.experiment_timing import ExperimentTiming

        trainer.add_callback(ExperimentTiming(trainer))
    if cfg.stop_after_steps:
        from transformers import TrainerCallback

        class StopAfterPrefix(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step >= cfg.stop_after_steps:
                    control.should_training_stop = True
                    control.should_save = args.save_strategy.value != "no"
                return control

        trainer.add_callback(StopAfterPrefix())

    # ── Online evaluation ────────────────────────────────────────
    if cfg.eval_data and cfg.eval_scoring:
        from evals.online_eval import EvalGroup, OnlineEvalCallback, _resolve_scorer, load_eval_data

        rows = load_eval_data(cfg.eval_data)
        scorer = _resolve_scorer(cfg.eval_scoring, cfg.eval_scoring_kwargs)
        eval_ck = cfg.eval_chat_template_kwargs
        if eval_ck is None:
            eval_ck = dataset_chat_template_kwargs or {}
        group = EvalGroup(
            rows=rows,
            scorer=lambda q, ref, resp, md: scorer(q, ref, resp),
            metrics_prefix="eval",
            group_by=["eval_type"],
            chat_template_kwargs=eval_ck,
        )
        if cfg.engine == "vllm":
            eval_vllm = trainer.llm
            # TRL's sync already handles sharded models; delegate to it.
            eval_weight_sync = lambda _m: trainer._move_model_to_vllm()  # noqa: E731
            manage_sleep = cfg.vllm_enable_sleep_mode
        else:
            eval_vllm = trainer._runner.eval_proxy()
            eval_weight_sync = lambda m: trainer._sync_module_to_side(m, side="a")  # noqa: E731
            manage_sleep = True
        eval_cb = OnlineEvalCallback(
            groups=[group],
            vllm_instance=eval_vllm,
            weight_sync_fn=eval_weight_sync,
            engine_is_collective=False,
            manage_sleep=manage_sleep,
            model_name=cfg.model_name,
            eval_every_n_steps=cfg.eval_every_n_steps,
            max_new_tokens=cfg.eval_max_new_tokens,
            output_dir=cfg.output_dir,
        )
        trainer.add_callback(eval_cb)
        trainer._unified_eval_callback = eval_cb
        print(f"  Online eval: {len(rows)} questions every {cfg.eval_every_n_steps} steps, "
              f"scoring={cfg.eval_scoring}", flush=True)

    # ── Train ────────────────────────────────────────────────────
    from ssd.checkpoint_utils import PruneCheckpointsBeforeSave, latest_checkpoint
    from ssd.run_metrics import RunMetricsCallback

    trainer.add_callback(PruneCheckpointsBeforeSave(cfg.output_dir))
    trainer.add_callback(RunMetricsCallback(
        output_dir=cfg.output_dir,
        method=method,
        model=cfg.model_name,
        n_gpu=2 if cfg.engine == "paired_vllm" else 1,
    ))

    resume = cfg.resume_from_checkpoint
    if isinstance(resume, str) and resume.lower() == "auto":
        resume = latest_checkpoint(cfg.output_dir)
        print(f"  Resuming from {resume}" if resume else "  No checkpoint found; starting from scratch.", flush=True)

    trainer.train(resume_from_checkpoint=resume)

    if cfg.record_student_fingerprint and trainer.is_world_process_zero():
        from ssd.experiment_timing import record_student_fingerprint

        record_student_fingerprint(trainer)
    if cfg.verify_frozen_teacher and trainer.is_world_process_zero():
        from ssd.experiment_timing import verify_frozen_teacher

        verify_frozen_teacher(trainer, cfg.model_name)
    if cfg.sync_ref_model and cfg.record_step_timing and trainer.is_world_process_zero():
        from ssd.experiment_timing import verify_teacher_ema_updates

        verify_teacher_ema_updates(trainer)
    if (cfg.verify_frozen_teacher or cfg.record_step_timing) and trainer.is_world_process_zero():
        trainer.state.save_to_json(os.path.join(cfg.output_dir, "completion-state.json"))

    if cfg.save_final_model:
        from contextlib import nullcontext

        from ssd.checkpoint_staging import checkpoint_io_lock

        final_dir = os.path.join(cfg.output_dir, "final")
        export_started = time.perf_counter()
        export_started_unix = time.time()
        lock = checkpoint_io_lock() if trainer.accelerator.num_processes == 1 else nullcontext(0.0)
        with lock as storage_wait:
            trainer.save_model(final_dir)
            tokenizer.save_pretrained(final_dir)
        if trainer.is_world_process_zero():
            from ssd.io_timing import record_io_timing

            record_io_timing(
                cfg.output_dir, trainer.state.global_step, "final_export",
                started_at_unix=export_started_unix, finished_at_unix=time.time(),
                seconds=time.perf_counter() - export_started, storage_wait_seconds=storage_wait,
            )
        print(f"Final model saved to {final_dir}", flush=True)

    if cfg.engine == "paired_vllm":
        trainer._runner.close()
        if cfg.paired_scratch_dir is None:
            import shutil

            shutil.rmtree(os.path.join(cfg.output_dir, "paired-scratch"), ignore_errors=True)
    else:
        trainer.close_vllm()


if __name__ == "__main__":
    main()
