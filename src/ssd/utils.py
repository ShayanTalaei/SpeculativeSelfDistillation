"""SSD utilities.

Strategy registry, factory function, and mini-batch splitting for PPO-style updates.
"""

from typing import Any, Union

import torch
from transformers import PreTrainedModel

from ssd.strategies import (
    RoutingStrategy,
    PureStudentStrategy,
    PureTeacherStrategy,
    KLSwitchStrategy,
    JSDSwitchStrategy,
)
from ssd.generation import ssd_generate, GenerationOutput
from ssd.loss import compute_masked_loss


def split_into_mini_batches(
    data: dict[str, Any],
    mini_batch_size: int,
) -> list[dict[str, Any]]:
    """Split a training batch into mini-batches for PPO-style updates.

    Handles both tensor values (split along dim 0) and list values (sliced).
    Drops remainder if batch_size is not evenly divisible by mini_batch_size.

    Args:
        data: dict mapping keys to tensors of shape (N, ...) or lists of length N
        mini_batch_size: number of samples per mini-batch

    Returns:
        list of dicts, each with mini_batch_size samples
    """
    # Determine batch size from first tensor value
    batch_size = None
    for v in data.values():
        if isinstance(v, torch.Tensor):
            batch_size = v.shape[0]
            break
        elif isinstance(v, list):
            batch_size = len(v)
            break

    if batch_size is None:
        return [data]

    num_complete = batch_size // mini_batch_size
    batches = []

    for i in range(num_complete):
        start = i * mini_batch_size
        end = start + mini_batch_size
        mini = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                mini[k] = v[start:end]
            elif isinstance(v, list):
                mini[k] = v[start:end]
            else:
                mini[k] = v
        batches.append(mini)

    return batches


STRATEGY_REGISTRY = {
    "pure_student": lambda cfg: PureStudentStrategy(),
    "pure_teacher": lambda cfg: PureTeacherStrategy(),
    "kl_switch": lambda cfg: KLSwitchStrategy(
        threshold=cfg.kl_threshold,
        direction=cfg.kl_direction,
    ),
    "jsd_switch": lambda cfg: JSDSwitchStrategy(
        threshold=cfg.jsd_threshold,
    ),
}


def build_strategy(config) -> RoutingStrategy:
    """Instantiate a RoutingStrategy from a config object."""
    name = config.routing_strategy
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown routing strategy '{name}'. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return STRATEGY_REGISTRY[name](config)
