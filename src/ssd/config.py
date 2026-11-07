"""Configuration for the speculative self-distillation (SSD) trainer.

Extends DistilConfig with the token routing rule, provenance-based loss
masking and the lockstep paired-vLLM settings.
"""

from dataclasses import dataclass, field
from typing import Optional

from distil_config import DistilConfig


@dataclass
class SSDConfig(DistilConfig):
    """Configuration for SSDDistilTrainer (the SSD trainer).

    Inherits all DistilConfig fields (KL loss alpha, temperature, etc.)
    and adds the routing and paired-engine settings.
    """

    # -- Routing strategy selection --
    routing_strategy: str = field(
        default="pure_student",
        metadata={
            "help": "Token-level routing rule: 'kl_switch' (hand the token to the teacher "
            "when the student/teacher KL exceeds kl_threshold) or 'jsd_switch' (same with JSD)."
        },
    )

    # -- KL switch parameters --
    kl_threshold: float = field(
        default=1.0,
        metadata={"help": "KL divergence threshold for kl_switch strategy."},
    )
    kl_direction: str = field(
        default="forward",
        metadata={"help": "KL direction for kl_switch: 'forward' (KL(T||S)) or 'reverse' (KL(S||T))."},
    )

    # -- JSD switch parameters --
    jsd_threshold: float = field(
        default=0.5,
        metadata={"help": "Jensen-Shannon divergence threshold for jsd_switch strategy. JSD ∈ [0, log 2 ≈ 0.693]."},
    )

    # -- Loss masking --
    loss_mask_strategy: str = field(
        default="all",
        metadata={
            "help": "How to mask loss by token provenance. Options: "
            "'all' (loss on every token), "
            "'teacher_only' (loss only on teacher-generated tokens), "
            "'student_only' (loss only on student-generated tokens), "
            "'directional' (reverse KL on student tokens, forward KL on teacher tokens)."
        },
    )

    # -- Teacher-distribution cache --
    teacher_cache: str = field(
        default="off",
        metadata={
            "help": "Reuse the teacher distributions computed during generation "
            "instead of re-running the teacher in the loss. "
            "'off': re-run the teacher (original behaviour). "
            "'verify': run BOTH — train on the recomputed teacher, but log how "
            "closely the cache reproduces it (tcache/* metrics). Slower than "
            "'off'; use it to validate before switching. "
            "'on': use the cache only, skipping the teacher forward entirely. "
            "Requires a routing strategy that runs the teacher during generation "
            "(i.e. not pure_student)."
        },
    )
    teacher_cache_topk: int = field(
        default=100,
        metadata={
            "help": "K for the cached teacher distribution. Each generated token "
            "stores the union of the teacher's and the student's top-K vocabulary "
            "entries, plus the exact full-vocab logsumexp, so kept entries carry "
            "their true probability and the dropped tail becomes one bucket. "
            "KL over that partition lower-bounds the true KL."
        },
    )

    # -- Lockstep paired vLLM (SSD) --
    paired_teacher_cache: str = "hidden"  # teacher targets reused as GPU-resident hidden states
    paired_python: str = "python"  # interpreter for the isolated vLLM server process
    paired_teacher_gpu: str = "1"
    paired_student_gpu: str = "0"
    paired_capacity: int = 65536  # paired scheduler KV capacity (tokens)
    paired_token_budget: int = 16384  # tokens per lockstep forward
    paired_compile_model: bool = True
    # True ships fp32 teacher logits per token to the student GPU; False ships
    # post-norm hidden states and projects them with a student-side copy of
    # the teacher head (refreshed each SYNC). Same speed on A100 NVLink.
    paired_transfer_logits: bool = True

    # -- PPO-style mini-batching --
    mini_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Mini-batch size for PPO-style updates. After generating rollouts for "
            "per_device_train_batch_size samples, iterate through "
            "per_device_train_batch_size/mini_batch_size mini-batches, updating the student "
            "after each. If None, equals per_device_train_batch_size (no mini-batching)."
        },
    )

    # -- Rollout logging --
    log_rollouts: bool = field(
        default=True,
        metadata={"help": "Whether to save per-token rollout data for the trainer visualizer."},
    )
    rollout_log_interval: int = field(
        default=10,
        metadata={"help": "Save rollout logs every N training steps."},
    )

    # -- Profiling --
    profile: bool = field(
        default=False,
        metadata={"help": "Enable lightweight per-step profiling (timings + CUDA memory) "
                          "to {output_dir}/profile/profile.jsonl."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.paired_teacher_cache != "hidden":
            raise ValueError("the lockstep pair reuses teacher hidden states: paired_teacher_cache must be 'hidden'")
        if self.routing_strategy not in {"jsd_switch", "kl_switch"}:
            raise ValueError("SSD routes on a divergence: routing_strategy must be jsd_switch or kl_switch")

        if self.mini_batch_size is None:
            self.mini_batch_size = self.per_device_train_batch_size

        valid_mask_strategies = {"all", "teacher_only", "student_only", "directional"}
        if self.loss_mask_strategy not in valid_mask_strategies:
            raise ValueError(
                f"loss_mask_strategy must be one of {valid_mask_strategies}, "
                f"got '{self.loss_mask_strategy}'"
            )

        valid_teacher_cache = {"off", "verify", "on"}
        if self.teacher_cache not in valid_teacher_cache:
            raise ValueError(
                f"teacher_cache must be one of {valid_teacher_cache}, "
                f"got '{self.teacher_cache}'"
            )

        if self.teacher_cache != "off":
            if self.teacher_cache_topk < 1:
                raise ValueError(
                    f"teacher_cache_topk must be >= 1, got {self.teacher_cache_topk}"
                )
            # pure_student never evaluates the teacher during generation, so
            # there is nothing to cache and the loss must keep recomputing it.
            if self.routing_strategy == "pure_student":
                raise ValueError(
                    "teacher_cache requires a routing strategy that evaluates the "
                    "teacher during generation; routing_strategy='pure_student' "
                    "does not. Leave teacher_cache='off' for this arm."
                )

        if self.kl_direction not in ("forward", "reverse"):
            raise ValueError(
                f"kl_direction must be 'forward' or 'reverse', got '{self.kl_direction}'"
            )
