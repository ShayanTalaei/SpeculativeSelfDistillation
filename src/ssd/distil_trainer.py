"""Speculative Self-Distillation (SSD) Trainer — TRL integration.

Extends DistilTrainer to replace standard generation with SSD generation
and add provenance-based loss masking. Keeps all TRL/accelerate distributed training
infrastructure (FSDP, DeepSpeed, gradient checkpointing, etc.) — no vLLM.

Key overrides:
- _generate_and_score_completions: SSD token-by-token generation
- _compute_loss: KL loss with provenance-based masking
"""

import os
import time
from typing import Any, Union

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, ProcessorMixin

from trl.data_utils import maybe_apply_chat_template
from trl.trainer.utils import pad

from distil_trainer import DistilTrainer
from ssd.config import SSDConfig
from ssd.generation import ssd_generate, ssd_generate_batch, GenerationOutput
from ssd.utils import build_strategy
from ssd.teacher_cache import (
    build_partition_logprobs,
    cache_diagnostics,
    stack_caches,
)
from ssd.rollout_logger import RolloutLogger
from ssd.profiling import init_profiler, get_profiler, maybe_phase

from accelerate import logging
from accelerate.utils import gather_object, is_peft_model

logger = logging.get_logger(__name__)

# Provenance encoding: integer tensor values
PROV_PAD = 0
PROV_STUDENT = 1
PROV_TEACHER = 2
PROV_FUSED = 3

PROV_STR_TO_INT = {"pad": PROV_PAD, "student": PROV_STUDENT, "teacher": PROV_TEACHER, "fused": PROV_FUSED}


def _encode_provenance(provenance_list: list[list[str]], max_len: int, device: torch.device) -> torch.Tensor:
    """Encode string provenance labels as an integer tensor.

    Returns (batch, max_len) tensor with values from PROV_* constants.
    """
    if any(len(row) > max_len for row in provenance_list):
        raise ValueError("provenance exceeds completion length")
    rows = [
        [PROV_STR_TO_INT[label] for label in row] + [PROV_PAD] * (max_len - len(row))
        for row in provenance_list
    ]
    # One host-to-device transfer, rather than one CUDA scalar assignment for
    # every generated token (up to 32,768 launches per rollout).
    return torch.tensor(rows, dtype=torch.long, device=device).reshape(len(rows), max_len)


def _maybe_check_provenance(
    provenance_list,
    completion_ids_list,
    provenance_int,
    completion_mask,
    *,
    where: str,
    step: int = 0,
):
    """Optional intermediate checks for the provenance attribution fix.

    Activated by ``SSD_DEBUG_PROVENANCE=1`` (env var). Asserts the
    invariants that the per-token provenance fix in
    the paired engine's per-token provenance tracking is supposed
    to guarantee, and prints per-step counts so the train log has a
    visible trail of "did the fix actually take effect".

    Specifically checks:
    - ``len(provenance_list[i]) == len(completion_ids_list[i])`` for every i
      (1:1 attribution at the runner output).
    - At every (b, t) position where ``completion_mask[b, t] == 1``,
      the encoded provenance is NOT ``PROV_PAD`` (no silent gaps).
    """
    if os.environ.get("SSD_DEBUG_PROVENANCE", "") != "1":
        return

    # Check 1: 1:1 at the runner output
    bad = []
    for i, (prov, ids) in enumerate(zip(provenance_list, completion_ids_list)):
        if len(prov) != len(ids):
            bad.append((i, len(prov), len(ids)))
    if bad:
        print(
            f"[PROV-CHECK FAIL @ {where} step={step}] "
            f"len(provenance) != len(token_ids) for {len(bad)}/{len(provenance_list)} samples: "
            f"first few = {bad[:3]}",
            flush=True,
        )
    else:
        n = len(provenance_list)
        n_tok = sum(len(ids) for ids in completion_ids_list)
        print(
            f"[PROV-CHECK OK   @ {where} step={step}] "
            f"all {n} samples 1:1; total completion tokens={n_tok}",
            flush=True,
        )

    # Check 2: no PROV_PAD at completion positions in the encoded tensor
    if provenance_int is not None and completion_mask is not None:
        is_pad = (provenance_int == PROV_PAD)
        leak_mask = is_pad & (completion_mask.bool())
        leak_count = int(leak_mask.sum().item())
        comp_count = int(completion_mask.sum().item())
        if leak_count > 0:
            print(
                f"[PROV-CHECK FAIL @ {where} step={step}] "
                f"{leak_count}/{comp_count} completion tokens encoded as PROV_PAD",
                flush=True,
            )
        else:
            # Also report the student/teacher split as a sanity log
            is_stu = (provenance_int == PROV_STUDENT) & completion_mask.bool()
            is_tea = (provenance_int == PROV_TEACHER) & completion_mask.bool()
            n_stu = int(is_stu.sum().item())
            n_tea = int(is_tea.sum().item())
            print(
                f"[PROV-CHECK OK   @ {where} step={step}] "
                f"completion tokens={comp_count}  student={n_stu}  teacher={n_tea}  "
                f"pad_at_completion=0  teacher_share={n_tea / max(comp_count, 1):.3f}",
                flush=True,
            )


def _check_teacher_cache_alignment(
    caches,
    completion_ids_list,
    *,
    where: str,
    step: int = 0,
):
    """Assert one cached teacher distribution per generated token.

    Unconditional, not behind a debug flag, because the failure is silent: a
    cache shorter than its completion pads out to fully-invalid slots, which
    contribute exactly 0 to the KL. The loss would keep falling and the
    affected tokens would simply stop training — the same class of bug as the
    ~45% of completion positions that sat at PROV_PAD before the provenance
    attribution fix, and just as invisible in the loss curve.
    """
    bad = [
        (i, len(c), len(ids))
        for i, (c, ids) in enumerate(zip(caches, completion_ids_list))
        if len(c) != len(ids)
    ]
    if bad:
        raise RuntimeError(
            f"[teacher-cache @ {where} step={step}] cache length != completion "
            f"length for {len(bad)}/{len(caches)} samples "
            f"(sample, n_cached, n_tokens): {bad[:5]}. Every generated token "
            f"must have exactly one cached teacher distribution."
        )


class SSDDistilTrainer(DistilTrainer):
    """DistilTrainer with SSD generation and provenance-based loss masking.

    Inherits all distributed training infrastructure from DistilTrainer/TRL.
    Overrides generation to do token-by-token SSD decoding using HuggingFace
    models directly (no vLLM). Overrides loss to support masking by which model
    generated each token.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        ref_model: Union[str, PreTrainedModel],
        args: SSDConfig = None,
        *,
        model_path_a: str,
        model_path_b: str,
        paired_scratch_dir: str,
        paired_max_model_len: int,
        vllm_gpu_memory_utilization: float = 0.8,
        **kwargs,
    ):
        assert args is not None, "SSDDistilTrainer requires args"
        assert args.use_vllm is False, (
            "the SSD trainer owns its own vLLM pair; args.use_vllm must be False"
        )
        assert paired_max_model_len > 0, "paired_max_model_len must be positive"

        super().__init__(model=model, ref_model=ref_model, args=args, **kwargs)

        self.engine = "paired_vllm"
        self.routing_strategy = build_strategy(args)
        self.loss_mask_strategy = args.loss_mask_strategy

        # Teacher-distribution cache. 'off' keeps the original behaviour (the
        # loss re-runs the teacher over prompt+completion); 'verify' runs both
        # and logs how well the cache reproduces the recompute; 'on' drops the
        # teacher forward entirely. See ssd.teacher_cache.
        self.teacher_cache_mode = getattr(args, "teacher_cache", "off")
        self.teacher_cache_topk = (
            int(getattr(args, "teacher_cache_topk", 0))
            if self.teacher_cache_mode != "off"
            else 0
        )
        # The cache stores logsumexp at the temperature the LOSS divides by,
        # which is self.temperature — not the sampling temperature, which may
        # be 0.0 for greedy decoding.
        self.teacher_cache_temperature = float(self.temperature)

        # Staleness guard. The teacher is only frozen if sync_ref_model is off;
        # with it on (the default here), MemoryEfficientSyncRefModelCallback
        # EMA-nudges ref_model toward the student at every on_step_end. Inside
        # one optimizer step that is harmless — generation syncs side B from
        # ref_model, then every accumulation micro-batch sees that same
        # teacher. It stops being harmless once a generation buffer outlives an
        # optimizer step, because the cache would then describe a teacher one
        # or more EMA updates behind the one the loss would have recomputed.
        if self.teacher_cache_topk > 0 and args.sync_ref_model:
            buffer_spans_steps = (
                args.steps_per_generation * self.num_iterations
            ) / max(args.gradient_accumulation_steps, 1)
            if buffer_spans_steps > 1:
                logger.warning(
                    "teacher_cache=%r with sync_ref_model=True and a generation "
                    "buffer spanning %.1f optimizer steps (steps_per_generation=%d "
                    "* num_iterations=%d / gradient_accumulation_steps=%d). The "
                    "teacher is EMA-updated (alpha=%g) once per optimizer step, so "
                    "cached distributions lag the recomputed ones by up to %d "
                    "update(s). Set steps_per_generation == "
                    "gradient_accumulation_steps, or sync_ref_model=False, to "
                    "keep the cache exact.",
                    self.teacher_cache_mode,
                    buffer_spans_steps,
                    args.steps_per_generation,
                    self.num_iterations,
                    args.gradient_accumulation_steps,
                    args.ref_model_mixup_alpha,
                    int(buffer_spans_steps),
                )
        self.print_memory_usage = False  # Enable detailed memory logging for debugging
        # Optional per-step profiler. Enabled via args.profile.
        if getattr(args, "profile", False):
            init_profiler(
                output_dir=os.path.join(args.output_dir, "profile"),
                enabled=True,
            )

        # Rollouts and teacher targets come from the isolated lockstep vLLM
        # pair; the trainer only sees token ids, provenance and resident caches.
        from ssd.paired_runner import PairedRunner

        self._last_synced_step = -1
        self._runner_on_all_ranks = False
        self._runner = PairedRunner(
            self, model_path_a, model_path_b, paired_scratch_dir,
            paired_max_model_len, vllm_gpu_memory_utilization,
        )
        self.rollout_logger = None

    def __del__(self):
        try:
            if self._runner is not None:
                self._runner.close()
                self._runner = None
        except Exception:
            pass
        
    def _ssd_generate_batch(
        self,
        student_prompts: list[str],
        teacher_prompts: list[str],
    ) -> tuple[list[list[int]], list[list[str]], list[GenerationOutput]]:
        """Generate speculative rollouts for a batch of prompts through the paired runner."""
        return self._runner_generate_batch(student_prompts, teacher_prompts)

    # ---- Paired runner helpers ----------------------------------------

    def _runner_generate_batch(
        self,
        student_prompts: list[str],
        teacher_prompts: list[str],
    ) -> tuple[list[list[int]], list[list[str]], list[GenerationOutput]]:
        """Wake the paired vLLM engine, sync weights, generate, sleep."""
        from accelerate.utils import broadcast_object_list

        def _mem(tag: str) -> None:
            if torch.cuda.is_available() and self.print_memory_usage:
                alloc = torch.cuda.memory_allocated() / (1 << 30)
                reserved = torch.cuda.memory_reserved() / (1 << 30)
                print(
                    f"[paired:step={self.state.global_step}] {tag}  "
                    f"torch_alloc={alloc:.2f} GiB  reserved={reserved:.2f} GiB",
                    flush=True,
                )

        # Phase timing (enabled via SSD_PROFILE=1)
        _profile = os.environ.get("SSD_PROFILE", "") == "1"
        _t = time.perf_counter
        def _tick(tag, t0):
            if _profile:
                torch.cuda.synchronize()
                print(f"[prof:step={self.state.global_step}] {tag:30s} {(_t()-t0)*1000:8.1f} ms", flush=True)
                return _t()
            return t0
        t0 = _t() if _profile else None

        # Wake vLLM up (no-op if enable_sleep_mode=False or already awake).
        # Every rank owning a TP shard must wake, not just rank 0.
        if self._runner is not None:
            self._runner.wake_up()
        _mem("after wake / before weight sync")
        t0 = _tick("wake_up", t0)

        # Sync once per global step.
        if self.state.global_step != self._last_synced_step:
            self._sync_module_to_side(self.model, side="a")
            t0 = _tick("sync student (side a)", t0)
            if self.args.sync_ref_model:
                self._sync_module_to_side(self.ref_model, side="b")
                t0 = _tick("sync teacher (side b)", t0)
            self._last_synced_step = self.state.global_step
        _mem("after weight sync / before vllm gen")

        # Gather prompts from all ranks, run on main, broadcast back.
        all_student = gather_object(student_prompts)
        all_teacher = gather_object(teacher_prompts)

        # With the runner sharded across ranks (external_launcher, TP=world),
        # llm.generate is a COLLECTIVE: every rank drives its own TP shard and
        # they must be called with identical inputs, in step. all_student /
        # all_teacher are already identical everywhere, being the output of
        # gather_object. Calling it on rank 0 alone would hang the others.
        if self._runner is not None:
            outs = self._runner.generate_batch(
                student_prompts=all_student,
                teacher_prompts=all_teacher,
                max_new_tokens=self.args.max_completion_length,
                temperature=self.args.temperature,
                top_k=self.args.top_k if self.args.top_k else 0,
            )
        else:
            outs = None
        t0 = _tick("paired generate_batch", t0)
        # Still broadcast rank 0's copy. Every rank computed the same thing, but
        # sampling that diverged even slightly between ranks would give each one
        # a different rollout to train on while they share a gradient — a
        # corruption that would be invisible in the loss curve.
        #
        # The teacher caches hang off these same GenerationOutput objects, so
        # they ride this broadcast and every rank trains against rank 0's
        # teacher distributions — the same guarantee, extended to the loss
        # inputs. Deliberate: each rank's local copy should be bitwise
        # identical (the engine may be TP-sharded, so the hidden states are
        # all-reduced and lm_head_b is replicated), but "should be" is exactly
        # the assumption this broadcast exists to avoid relying on. At K=100
        # over a 16-sequence global batch the extra payload is ~10 MB.
        container = [outs]
        if self.accelerator.num_processes > 1:
            broadcast_object_list(container, from_process=0)
        outs = container[0]
        n_local = len(student_prompts)
        rank = self.accelerator.process_index
        gen_outputs = outs[rank * n_local : (rank + 1) * n_local]

        # Pad fields the HF path fills but the paired engine does not, so downstream
        # metric logging doesn't hit empty lists.
        for g in gen_outputs:
            n = len(g.token_ids)
            if not g.kl_reverse:
                g.kl_reverse = [0.0] * n
            if not g.student_logprobs:
                g.student_logprobs = [0.0] * n
            if not g.teacher_logprobs:
                g.teacher_logprobs = [0.0] * n

        completion_ids_list = [g.token_ids for g in gen_outputs]
        provenance_list = [g.provenance for g in gen_outputs]
        _mem("after vllm gen / before sleep")
        t0 = _tick("gen post-processing", t0)

        # Put vLLM to sleep so backprop has full GPU memory.
        # (no-op if enable_sleep_mode=False). Collective, as for wake_up.
        if self._runner is not None:
            self._runner.sleep(level=1)
        _mem("after vllm sleep")
        t0 = _tick("vllm sleep", t0)
        return completion_ids_list, provenance_list, gen_outputs

    def _sync_module_to_side(self, module, *, side: str) -> None:
        """Push all parameters from a live HF module into the PairedRunner.

        Three cases, and the distinction matters for correctness:

        * **Unsharded** — every rank holds whole parameters, so the main
          process just hands its ``state_dict()`` over.
        * **FSDP / DeepSpeed ZeRO-3** — each rank holds only a *shard*. The
          full tensor has to be re-formed before it can be written into the
          engine, and the re-forming primitives (``summon_full_params``,
          ``GatheredParameters``, ``DTensor.full_tensor``) are **collectives**:
          every rank must enter them or the ones that do hang forever. That is
          why the gather below is NOT wrapped in ``is_main_process`` — only the
          push into the runner is, since the runner lives on the main process.

        Mirrors ``DistilTrainer._move_model_to_vllm``'s gather branches, but
        routes to ``PairedRunner.update_named_params`` (which rewrites HF
        names to per-side engine names) instead of vLLM's ``load_weights``.
        """
        if self._runner is None and not (
            self.is_fsdp_enabled or self._is_zero3_enabled()
        ):
            # Non-main rank in an unsharded run: nothing to gather, nothing to
            # push. Returning early here is safe precisely because no collective
            # is involved.
            return

        # Push wherever a runner exists. When the engine is sharded across
        # ranks each one owns a TP slice and must receive the full tensor —
        # vLLM's weight_loader then writes only that rank's slice. Pushing from
        # rank 0 alone would leave every other shard holding stale weights,
        # which does not crash and does not show up in the loss.
        push = (
            self._runner.update_named_params if self._runner is not None else None
        )

        # LoRA: the engine holds a plain model, so it needs the EFFECTIVE
        # weights (base + BA), not base weights plus adapter tensors it has no
        # slot for.
        #
        # Under FSDP the merge canNOT happen here. merge_adapter() computes
        # `base.weight += B @ A`, and FSDP has replaced base.weight with a flat
        # shard, so PEFT fails with
        #     RuntimeError: inconsistent tensor size, expected tensor [16384]
        #     and src [81920] to have the same number of elements
        # The merge has to happen per FSDP unit, while that unit is summoned —
        # see _merge_lora_units. Gathering the whole model first would work but
        # peaks at the full unsharded size (29.6 GB/rank at 14B), which defeats
        # the point of sharding.
        peft_root = self.accelerator.unwrap_model(module)
        is_peft = is_peft_model(peft_root)

        if is_peft and not self.is_fsdp_enabled:
            peft_root.merge_adapter()
            try:
                self._dispatch_sync(module, side, push)
            finally:
                peft_root.unmerge_adapter()
            return

        self._dispatch_sync(module, side, push)

    def _dispatch_sync(self, module, side: str, push) -> None:
        """Route to the right gather for the sharding scheme in use."""
        if self.is_fsdp_enabled:
            fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
            fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
            if fsdp_version == 1:
                self._ensure_fsdp_root_initialized(module)
                self._gather_fsdp1_params_to_side(module, side, push)
            else:
                self._gather_fsdp2_params_to_side(module, side, push)
            return

        if self._is_zero3_enabled():
            import deepspeed

            unwrapped = self.accelerator.unwrap_model(module)
            for name, param in unwrapped.named_parameters():
                target = self._runner_param_name(name)
                if target is None:
                    continue
                # One parameter at a time keeps peak memory at a single full
                # tensor rather than the whole model.
                with deepspeed.zero.GatheredParameters([param]):
                    if push is not None:
                        push(side, [(target, param.data)])
            return

        if push is not None:
            unwrapped = self.accelerator.unwrap_model(module)
            named = []
            for name, tensor in unwrapped.state_dict().items():
                target = self._runner_param_name(name)
                if target is not None:
                    named.append((target, tensor))
            push(side, named)

    def _is_zero3_enabled(self) -> bool:
        plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        return plugin is not None and plugin.zero_stage == 3

    def _runner_param_name(self, raw: str) -> str | None:
        """Map a training-side parameter name to its HF name, or None to skip.

        Strips the wrappers the trainer stacks on, in whatever combination is
        active: FSDP (`_fsdp_wrapped_module.`), gradient checkpointing
        (`_checkpoint_wrapped_module.`) and PEFT (`base_model.model.` plus the
        `base_layer.` that LoRA inserts under each replaced Linear).

        Returns None for adapter-only tensors. After merge_adapter the base
        weight already carries BA, so lora_A/lora_B are redundant, and the
        engine has no parameter to receive them anyway —
        load_named_params_into_side would silently drop them, but skipping here
        avoids shipping them across the wire every step.
        """
        if ".lora_A" in raw or ".lora_B" in raw or "lora_embedding_" in raw:
            return None
        if "original_module." in raw:  # modules_to_save keeps a frozen copy
            return None
        name = self._fix_param_name_to_vllm(
            raw,
            extra_prefixes=[
                "_fsdp_wrapped_module.",
                "base_model.model.",
                "base_layer.",
                "modules_to_save.default.",
            ],
        )
        return name

    @staticmethod
    def _ensure_fsdp_root_initialized(module) -> None:
        """Run FSDP's deferred root setup before touching any child unit.

        FSDP1 initialises lazily on the first forward: the outermost instance
        marks itself root and every nested one non-root. SSD syncs weights at
        the *start* of a step, so the very first sync of a run happens before
        any forward has run. The gather walks the tree post-order (children
        first, to keep only one unit unsharded at a time), so without this the
        first child summoned initialises itself as a root, and the real root
        then trips:

            AssertionError: Non-root FSDP instance's `_is_root` should not have
            been set yet or should have been set to `False`

        Initialising the root first fixes the order. Idempotent, so later steps
        pay nothing.
        """
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        root = module if isinstance(module, FSDP) else next(
            (m for m in module.modules() if isinstance(m, FSDP)), None
        )
        if root is None:
            return
        try:
            from torch.distributed.fsdp._runtime_utils import _lazy_init

            _lazy_init(root, root)
        except Exception:
            # Private API; if it moves, fall through. The traversal still works
            # once a forward has happened, which is every step after the first.
            pass

    def _gather_fsdp1_params_to_side(
        self, module, side: str, push, prefix: str = "", visited=None
    ) -> None:
        """Post-order walk of the FSDP tree, summoning one unit at a time.

        Post-order (children before the parent) with ``recurse=False`` means
        only one FSDP unit's parameters are unsharded at any moment, so peak
        memory is one unit rather than the whole model — the same reason TRL
        does it this way.
        """
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if visited is None:
            visited = set()
        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._gather_fsdp1_params_to_side(
                child_module, side, push, prefix=child_prefix, visited=visited
            )

        if isinstance(module, FSDP):
            # Collective — entered by every rank, including non-main ones.
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                # Merge LoRA only now, while THIS unit's base weights are whole.
                # writeback=False means the mutation is discarded on exit, but
                # PEFT's `merged_adapters` bookkeeping is not, so unmerge has to
                # happen inside the context too — outside it the weight is a
                # shard again and unmerge would hit the same shape error.
                merged_layers = self._merge_lora_units(module)
                try:
                    self._collect_and_push(module, side, push, prefix, visited)
                finally:
                    for layer in merged_layers:
                        layer.unmerge()

    def _collect_and_push(self, module, side, push, prefix, visited) -> None:
        """Read one summoned FSDP unit's parameters and ship them.

        Must be called inside that unit's ``summon_full_params`` context: the
        full tensors cease to exist when it exits.
        """
        batch = []
        for param_name, param in module.named_parameters():
            raw = f"{prefix}.{param_name}" if prefix else param_name
            full_name = self._runner_param_name(raw)
            if full_name is None:
                continue  # adapter tensor; already merged into the base
            if full_name in visited:
                continue  # already emitted by a nested FSDP unit
            visited.add(full_name)
            p = param.data
            # Under fsdp_offload_params the summoned tensor lands on CPU.
            # vLLM's weight_loader slices into a CUDA parameter, so hand it
            # something already on device.
            if p.is_cpu:
                p = p.to(torch.device("cuda"), non_blocking=True)
            batch.append((full_name, p))
        # One push per FSDP unit, not per parameter: update_named_params
        # already splits by payload size, and a per-parameter call would be
        # ~500 round trips per step on a 14B model instead of ~40.
        if push is not None and batch:
            push(side, batch)

    @staticmethod
    def _merge_lora_units(module) -> list:
        """Merge LoRA adapters belonging to THIS FSDP unit only.

        Returns the layers merged, for the caller to unmerge inside the same
        context. Nested FSDP units are skipped: their parameters are still
        sharded here (the post-order walk already handled them under their own
        summon), and merging against a shard is exactly the
        ``inconsistent tensor size`` failure this whole dance exists to avoid.
        """
        try:
            from peft.tuners.lora import LoraLayer
        except Exception:
            return []
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        merged = []
        stack = [module]
        while stack:
            m = stack.pop()
            for child in m.children():
                if isinstance(child, FSDP):
                    continue  # a different unit's problem
                stack.append(child)
            if isinstance(m, LoraLayer) and not m.merged:
                m.merge()
                merged.append(m)
        return merged

    def _gather_fsdp2_params_to_side(self, module, side: str, push) -> None:
        """FSDP2 (``fully_shard``) keeps DTensors, so state_dict() covers all."""
        for name, param in module.state_dict().items():
            target = self._runner_param_name(name)
            if target is None:
                continue
            if param.is_cpu:
                param = param.to(torch.device("cuda"))
            # Collective — every rank must call full_tensor().
            full = param.full_tensor() if hasattr(param, "full_tensor") else param
            if push is not None:
                push(side, [(target, full)])

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """Override: SSD generation instead of vLLM/standard generate.

        Returns the same dict format as parent, plus a 'provenance' integer tensor
        that gets auto-split by split_tensor_dict in _prepare_inputs.

        Fast path: for pure_student, uses model.generate() for ~10-50x speedup.
        """
        # Profiler: mark start of new step
        prof = get_profiler()
        if prof is not None:
            prof.start_step(self.state.global_step)
            prof.snapshot_mem("start_of_step")
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]
        teacher_prompts = [x["teacher_prompt"] for x in inputs]

        tokenizer = self.processing_class
        if isinstance(tokenizer, ProcessorMixin):
            tokenizer = tokenizer.tokenizer

        # Forward chat_template_kwargs (e.g. enable_thinking) if present in inputs
        chat_template_kwargs = {}
        if inputs and "chat_template_kwargs" in inputs[0]:
            chat_template_kwargs = inputs[0]["chat_template_kwargs"]

        with maybe_phase("generation/chat_template"):
            prompts_text = [
                maybe_apply_chat_template(
                    {"prompt": p, "chat_template_kwargs": chat_template_kwargs},
                    self.processing_class,
                )["prompt"]
                for p in prompts
            ]
            teacher_prompts_text = [
                maybe_apply_chat_template(
                    {"prompt": p, "chat_template_kwargs": chat_template_kwargs},
                    self.processing_class,
                )["prompt"]
                for p in teacher_prompts
            ]

        # Slow path: SSD token-by-token generation
        with maybe_phase("generation/ssd_loop_total"):
            completion_ids_list, provenance_list, gen_outputs = self._ssd_generate_batch(
                prompts_text, teacher_prompts_text,
            )
        if prof is not None:
            prof.snapshot_mem("after_generation")

        # Tokenize student prompts
        student_inputs = tokenizer(
            prompts_text, return_tensors="pt", padding=True,
            padding_side="left", max_length=self.max_prompt_length,
            truncation=True, add_special_tokens=False,
        )
        student_prompt_ids = student_inputs["input_ids"].to(device)
        student_prompt_mask = student_inputs["attention_mask"].to(device)

        # Tokenize teacher prompts
        teacher_inputs = tokenizer(
            teacher_prompts_text, return_tensors="pt", padding=True,
            padding_side="left", max_length=self.max_prompt_length,
            truncation=True, add_special_tokens=False,
        )
        teacher_prompt_ids = teacher_inputs["input_ids"].to(device)
        teacher_prompt_mask = teacher_inputs["attention_mask"].to(device)

        # Pad completion ids
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones(len(ids), dtype=torch.long, device=device) for ids in completion_ids_list]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")

        # Encode provenance as integer tensor — survives split_tensor_dict
        max_comp_len = completion_ids.shape[1]
        provenance = _encode_provenance(provenance_list, max_comp_len, device)

        # Optional: verify per-token provenance attribution invariants. The
        # paired vLLM engine's _model_forward override (vllm_plugin) is supposed
        # to guarantee that len(prov)==len(token_ids) for every sample so the
        # loss-mask in directional / teacher_only / student_only never quietly
        # drops completion tokens. Set SSD_DEBUG_PROVENANCE=1 to log this
        # invariant per step in the train log.
        _maybe_check_provenance(
            provenance_list,
            completion_ids_list,
            provenance,
            completion_mask,
            where="_compute_loss",
            step=self.state.global_step,
        )

        # Log rollouts
        if self.rollout_logger and self.state.global_step % self.args.rollout_log_interval == 0:
            for i, gen in enumerate(gen_outputs):
                self.rollout_logger.log_rollout(
                    step=self.state.global_step,
                    student_prompt=prompts_text[i],
                    teacher_prompt=teacher_prompts_text[i],
                    gen_output=gen,
                    per_token_loss=[0.0] * len(gen.tokens),
                    strategy_name=self.args.routing_strategy,
                    strategy_config={"kl_threshold": self.args.kl_threshold},
                    total_loss=0.0,
                )
            self.rollout_logger.flush(self.state.global_step)

        # Log completion stats
        comp_lengths = torch.tensor([len(ids) for ids in completion_ids_list], device=device)
        self._metrics[mode]["completions/mean_length"].append(comp_lengths.float().mean().item())
        # Log cumulative completion-token count (tokens with gradient signal).
        # Matches TRL SFTTrainer's `num_tokens` convention (completion-only,
        # not prompt). Allows cross-method fairness plots: score vs tokens.
        # Gathered across ranks: this is the cross-method comparison axis, and
        # the SDFT arm counts globally, so a per-rank figure here would have
        # understated SSD's token use by world_size.
        step_tokens = self.accelerator.gather(comp_lengths).sum().item()
        self._cum_num_tokens = getattr(self, "_cum_num_tokens", 0.0) + float(step_tokens)
        self._metrics[mode]["num_tokens"].append(self._cum_num_tokens)

        s_count = sum(p.count("student") for p in provenance_list)
        t_count = sum(p.count("teacher") for p in provenance_list)
        total = s_count + t_count
        if total > 0:
            self._metrics[mode]["ssd/student_ratio"].append(s_count / total)
            self._metrics[mode]["ssd/teacher_ratio"].append(t_count / total)

        # Log generation-time per-step stats from gen_outputs
        if gen_outputs:
            all_gen_kl_fwd = [kl for g in gen_outputs for kl in g.kl_forward]
            all_gen_kl_rev = [kl for g in gen_outputs for kl in g.kl_reverse]
            all_gen_s_ent = [e for g in gen_outputs for e in g.student_entropy]
            all_gen_t_ent = [e for g in gen_outputs for e in g.teacher_entropy]
            if all_gen_kl_fwd:
                self._metrics[mode]["gen/mean_kl_forward"].append(sum(all_gen_kl_fwd) / len(all_gen_kl_fwd))
                self._metrics[mode]["gen/mean_kl_reverse"].append(sum(all_gen_kl_rev) / len(all_gen_kl_rev))
                self._metrics[mode]["gen/mean_student_entropy"].append(sum(all_gen_s_ent) / len(all_gen_s_ent))
                self._metrics[mode]["gen/mean_teacher_entropy"].append(sum(all_gen_t_ent) / len(all_gen_t_ent))

        completions_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))

        batch = {
            "prompt_ids": student_prompt_ids,
            "prompt_mask": student_prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "teacher_prompt_ids": teacher_prompt_ids,
            "teacher_prompt_mask": teacher_prompt_mask,
            "provenance": provenance,
            "advantages": torch.zeros_like(completion_ids, dtype=torch.float32),
            "num_items_in_batch": comp_lengths.sum(),
        }

        # A plain list, not a padded tensor: sequences differ in length by a
        # lot here (140-450 tokens observed), and both shuffle_sequence_dict
        # and split_tensor_dict index lists along the batch axis exactly as
        # they do tensors — so the caches stay glued to their own rollout
        # through the shuffle and the split into accumulation micro-batches.
        # Padding happens later, per micro-batch, in _compute_loss.
        if self.teacher_cache_topk > 0:
            caches = [g.teacher_cache for g in gen_outputs]
            missing = [i for i, c in enumerate(caches) if c is None]
            if missing:
                raise RuntimeError(
                    f"teacher_cache={self.teacher_cache_mode!r} but "
                    f"{len(missing)}/{len(caches)} generations came back without a "
                    f"teacher cache (first few: {missing[:3]}). The generation "
                    f"backend is not capturing it — check engine wiring."
                )
            _check_teacher_cache_alignment(
                caches, completion_ids_list, where="_generate_and_score_completions",
                step=self.state.global_step,
            )
            batch["teacher_cache"] = caches

        if self.engine == "paired_vllm" and self.args.paired_teacher_cache in ("hidden", "logits"):
            caches = [g.full_teacher_cache for g in gen_outputs]
            if any(c is None for c in caches):
                raise RuntimeError("paired backend returned missing full teacher cache")
            batch["full_teacher_cache"] = caches

        return batch

    def _compute_loss(self, model, inputs):
        """KL loss with token provenance-based masking.

        Provenance tensor from inputs["provenance"] (auto-split by parent).
        """
        from ssd.paired_cache import trim_cached_completion_padding

        inputs = trim_cached_completion_padding(inputs)
        _profile = os.environ.get("SSD_PROFILE", "") == "1"
        _lt = time.perf_counter
        def _ltick(tag, t0):
            if _profile:
                torch.cuda.synchronize()
                print(f"[prof-loss:step={self.state.global_step}] {tag:30s} {(_lt()-t0)*1000:8.1f} ms", flush=True)
                return _lt()
            return t0
        _t0 = _lt() if _profile else None
        if torch.cuda.is_available() and self.print_memory_usage:
            alloc = torch.cuda.memory_allocated() / (1 << 30)
            reserved = torch.cuda.memory_reserved() / (1 << 30)
            print(
                f"[loss:step={self.state.global_step}] before forward  "
                f"torch_alloc={alloc:.2f} GiB  reserved={reserved:.2f} GiB",
                flush=True,
            )
        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        teacher_prompt_ids = inputs["teacher_prompt_ids"]
        teacher_prompt_mask = inputs["teacher_prompt_mask"]
        provenance = inputs["provenance"]  # (B, L) integer tensor

        # Build full sequences
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        teacher_input_ids = torch.cat([teacher_prompt_ids, completion_ids], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        prof = get_profiler()
        # Student forward (with gradients)
        with maybe_phase("loss/student_forward"):
            per_token_logps, all_logps, entropies = self._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, logits_to_keep, compute_entropy=True,
            )
        if prof is not None:
            prof.snapshot_mem("after_student_forward")
        _t0 = _ltick("student forward+logsmax", _t0)

        # Teacher distributions. Either recomputed here with a second full
        # forward, or read back from the generation-time cache — the two are
        # the same quantity at the same positions, since generation step t and
        # loss position t both predict completion token t.
        teacher_cache = inputs.get("teacher_cache")
        use_cache = self.teacher_cache_mode == "on" and teacher_cache is not None
        paired_timing = getattr(self, "engine", None) == "paired_vllm"
        if paired_timing:
            torch.cuda.synchronize()
            teacher_target_start = time.perf_counter()

        full_cache = inputs.get("full_teacher_cache")
        if full_cache is None:
            raise RuntimeError("lockstep loss requires reused teacher hidden states")
        if full_cache is not None:
            from ssd.paired_cache import full_teacher_logps
            with maybe_phase("loss/teacher_cache_projection"):
                teacher_all_logps = full_teacher_logps(
                    full_cache, self.ref_model.get_output_embeddings(), completion_ids,
                    self.state.global_step, self.temperature, all_logps.device,
                )
                teacher_per_token_logps = teacher_all_logps.gather(-1, completion_ids[..., None]).squeeze(-1)
        elif use_cache:
            teacher_all_logps = None
            teacher_per_token_logps = None
        else:
            with maybe_phase("loss/teacher_forward"):
                with torch.no_grad():
                    teacher_per_token_logps, teacher_all_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model, teacher_input_ids, teacher_attention_mask, logits_to_keep, compute_entropy=False,
                    )
            if prof is not None:
                prof.snapshot_mem("after_teacher_forward")
            _t0 = _ltick("teacher forward+logsmax", _t0)

        # Coarsen both distributions onto {kept tokens} ∪ {tail} when the cache
        # is in play. The result is just a narrower vocabulary axis, so every
        # KL formula below consumes it unchanged.
        cache_student_logps = cache_teacher_logps = None
        cache_stacked = p_kept_mass = q_kept_mass = None
        if teacher_cache is not None and self.teacher_cache_mode != "off":
            with maybe_phase("loss/teacher_cache_expand"):
                cache_stacked = stack_caches(
                    teacher_cache, logits_to_keep, all_logps.device
                )
                c_idx, c_val, c_lse, c_valid = cache_stacked
                (
                    cache_student_logps,
                    cache_teacher_logps,
                    p_kept_mass,
                    q_kept_mass,
                ) = build_partition_logprobs(
                    all_logps, c_idx, c_val, c_lse, c_valid,
                    self.teacher_cache_temperature,
                )
            _t0 = _ltick("teacher cache expand", _t0)

        if paired_timing:
            torch.cuda.synchronize()
            metric_mode = "train" if self.model.training else "eval"
            self._metrics[metric_mode]["paired/teacher_target_seconds"].append(
                time.perf_counter() - teacher_target_start
            )
            self._metrics[metric_mode]["paired/teacher_forward_skipped"].append(
                float(use_cache or full_cache is not None)
            )

        # Loss mask: completion_mask with optional token skipping
        loss_completion_mask = completion_mask.clone().float()
        if self.num_loss_tokens_to_skip > 0:
            B, L = completion_mask.shape
            positions = torch.arange(L, device=completion_mask.device).unsqueeze(0).expand(B, -1)
            loss_completion_mask = loss_completion_mask * (positions >= self.num_loss_tokens_to_skip).float()

        is_student = (provenance == PROV_STUDENT).float()
        is_teacher = (provenance == PROV_TEACHER).float()

        # Optional: verify directional loss covers every completion token.
        # Without the provenance-attribution fix, ~45% of completion positions
        # stayed at PROV_PAD and silently contributed 0 to per_token_loss (both
        # is_student and is_teacher were 0 there). With the fix,
        # is_student + is_teacher == 1 everywhere completion_mask==1.
        if (
            self.loss_mask_strategy == "directional"
            and os.environ.get("SSD_DEBUG_PROVENANCE", "") == "1"
        ):
            cover = (is_student + is_teacher) * loss_completion_mask
            cmask = loss_completion_mask
            n_covered = int(cover.sum().item())
            n_completion = int(cmask.sum().item())
            if n_covered != n_completion:
                print(
                    f"[PROV-CHECK FAIL @ directional-loss step={self.state.global_step}] "
                    f"directional coverage: {n_covered}/{n_completion} "
                    f"completion tokens have a (student|teacher) label",
                    flush=True,
                )
            else:
                print(
                    f"[PROV-CHECK OK   @ directional-loss step={self.state.global_step}] "
                    f"directional coverage: {n_covered}/{n_completion} "
                    f"(is_student+is_teacher==1 at every completion position)",
                    flush=True,
                )

        def _per_token_loss(student_logps, teacher_logps):
            """Per-token loss from a (student, teacher) log-prob pair.

            Agnostic to what the last axis is: the full vocabulary, or the
            {kept tokens} ∪ {tail} partition the cache expands to. Every rule
            below is elementwise then summed over that axis, so both work.
            """
            if self.loss_mask_strategy == "directional":
                # Reverse KL on student tokens, forward KL on teacher tokens
                fwd_kl = F.kl_div(student_logps, teacher_logps, reduction="none", log_target=True).sum(-1)
                rev_kl = F.kl_div(teacher_logps, student_logps, reduction="none", log_target=True).sum(-1)
                return rev_kl * is_student + fwd_kl * is_teacher
            out = self._compute_kl_per_token(student_logps, teacher_logps)
            if self.loss_mask_strategy == "teacher_only":
                out = out * (provenance == PROV_TEACHER).float()
            elif self.loss_mask_strategy == "student_only":
                out = out * (provenance == PROV_STUDENT).float()
            # "all": no extra masking
            return out

        # Compute per-token loss with provenance masking
        with maybe_phase("loss/kl_compute"):
            if use_cache:
                per_token_loss = _per_token_loss(cache_student_logps, cache_teacher_logps)
            else:
                per_token_loss = _per_token_loss(all_logps, teacher_all_logps)

        # 'verify': train on the recomputed teacher exactly as before, and
        # separately measure what the cache would have produced. Deliberately
        # keeps the cache out of the gradient so a verification run stays
        # comparable to a plain run — the point is to decide whether to trust
        # the cache, not to half-adopt it.
        if self.teacher_cache_mode == "verify" and cache_student_logps is not None:
            with torch.no_grad():
                cached_per_token_loss = _per_token_loss(
                    cache_student_logps.detach(), cache_teacher_logps
                )
                self._log_teacher_cache_diagnostics(
                    mode="train" if self.model.training else "eval",
                    teacher_all_logps=teacher_all_logps,
                    cache_stacked=cache_stacked,
                    exact_per_token_loss=per_token_loss,
                    cached_per_token_loss=cached_per_token_loss,
                    p_kept_mass=p_kept_mass,
                    q_kept_mass=q_kept_mass,
                    token_mask=loss_completion_mask,
                )
            _t0 = _ltick("teacher cache verify", _t0)

        # Normalize: mean over tokens, mean over batch
        loss = ((per_token_loss * loss_completion_mask).sum(-1) / loss_completion_mask.sum(-1).clamp(min=1.0)).mean()
        loss = loss / self.current_gradient_accumulation_steps

        # Visualization callback hook (same as parent DistilTrainer). The
        # visualizer wants full-vocabulary teacher log-probs; under
        # teacher_cache='on' those were never materialised, so hand it the
        # partition pair instead — its per-token views (loss, top-token
        # agreement) still read correctly off a narrower axis.
        if hasattr(self, '_viz_callback') and self._viz_callback is not None:
            with torch.no_grad():
                self._viz_callback.log_step(
                    self, self.state.global_step, inputs,
                    all_logps if not use_cache else cache_student_logps,
                    teacher_all_logps if not use_cache else cache_teacher_logps,
                    per_token_loss,
                    loss_completion_mask, entropies, teacher_entropies=None)

        # Log metrics
        mode = "train" if self.model.training else "eval"
        with torch.no_grad():
            # Log actual distributional KL (same as loss, before masking/normalization)
            kl_mean = (per_token_loss * loss_completion_mask).sum() / loss_completion_mask.sum()
        self._metrics[mode]["kl"].append(self.accelerator.gather(kl_mean).nanmean().item())

        mean_entropy = (entropies * loss_completion_mask).sum() / loss_completion_mask.sum().clamp(min=1.0)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        if prof is not None:
            prof.snapshot_mem("end_of_loss")
            prof.dump_step(self.state.global_step)

        if torch.cuda.is_available() and self.print_memory_usage:
            alloc = torch.cuda.memory_allocated() / (1 << 30)
            reserved = torch.cuda.memory_reserved() / (1 << 30)
            print(
                f"[loss:step={self.state.global_step}] after forward (pre-backward)  "
                f"torch_alloc={alloc:.2f} GiB  reserved={reserved:.2f} GiB",
                flush=True,
            )

        return loss

    def _log_teacher_cache_diagnostics(
        self,
        *,
        mode,
        teacher_all_logps,
        cache_stacked,
        exact_per_token_loss,
        cached_per_token_loss,
        p_kept_mass,
        q_kept_mass,
        token_mask,
    ):
        """Measure the cache against the teacher forward it would replace.

        Two independent error sources, reported separately because they have
        different fixes:

        * **Fidelity** — do the cached distributions match what the teacher
          forward produces at the same positions? ``tcache/logprob_*``. A
          one-token misalignment shows up here as a large error. A small
          systematic one means the two teacher evaluations see slightly
          different context: generation tokenizes prompts with
          ``add_special_tokens`` on and no truncation and derives position_ids
          from the attention mask, while the loss path does the opposite on
          all three counts, and the paired vLLM engine additionally inserts
          alignment marker tokens into the teacher prompt. Those are
          pre-existing discrepancies between the two evaluations; the cache
          just makes them visible.
        * **Truncation** — how much of each distribution the top-K covers, and
          what that costs the loss. ``tcache/*_kept_mass``, ``tcache/kl_*``.
          Raise ``teacher_cache_topk`` if the kept masses drop.

        Note ``kl_rel_err`` mixes both effects and its sign is not meaningful.
        Partition KL lower-bounds the true KL of the *cached* teacher's own
        distribution, but ``kl_exact`` here comes from a different teacher
        evaluation (see the marker point above), and at K=100 that difference
        (~0.018 prob-weighted) dwarfs the truncation term (~0.0003 at 99.99%
        kept mass). So the measured error is dominated by conditioning, not
        truncation, and lands on either side of zero. Judge truncation by the
        kept masses; judge magnitude by |kl_rel_err|.
        """
        idx, val, lse, valid = cache_stacked

        stats = cache_diagnostics(
            teacher_all_logps=teacher_all_logps,
            indices=idx,
            logits=val,
            lse=lse,
            valid=valid,
            temperature=self.teacher_cache_temperature,
            token_mask=token_mask,
        )

        denom = token_mask.sum().clamp(min=1.0)
        exact = (exact_per_token_loss * token_mask).sum() / denom
        cached = (cached_per_token_loss * token_mask).sum() / denom

        stats["kl_exact"] = float(exact)
        stats["kl_cached"] = float(cached)
        stats["kl_abs_err"] = float(exact - cached)
        stats["kl_rel_err"] = float((exact - cached) / exact.abs().clamp(min=1e-12))
        stats["student_kept_mass"] = float(
            (p_kept_mass * token_mask).sum() / denom
        )
        stats["teacher_kept_mass_from_cache"] = float(
            (q_kept_mass * token_mask).sum() / denom
        )
        stats["slots_used"] = float(
            (valid.float().sum(-1) * token_mask).sum() / denom
        )

        for key, value in stats.items():
            tensor = torch.tensor(value, device=self.accelerator.device)
            self._metrics[mode][f"tcache/{key}"].append(
                self.accelerator.gather(tensor).nanmean().item()
            )
