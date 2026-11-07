"""Speculative self-distillation (SSD) trainer package.

Token-level routing between student and teacher during generation (a
RoutingStrategy decides per token), provenance-based loss masking, and the
bridge to the lockstep paired vLLM engine.
"""

from ssd.strategies import (
    TokenDecision,
    RoutingStrategy,
    PureStudentStrategy,
    PureTeacherStrategy,
    KLSwitchStrategy,
    EntropySwitchStrategy,
    TopKIntersectionStrategy,
    LogitFusionStrategy,
)
from ssd.generation import ssd_generate, ssd_generate_batch, GenerationOutput
from ssd.loss import compute_kl_loss, apply_provenance_mask, compute_masked_loss
from ssd.utils import split_into_mini_batches, build_strategy, STRATEGY_REGISTRY
from ssd.config import SSDConfig
from ssd.rollout_logger import RolloutLogger, TrainerRolloutToken, TrainerRollout
