"""Full-vocabulary targets from CPU storage or trainer-owned CUDA hidden states."""

from dataclasses import dataclass

import torch


@dataclass
class FullTeacherCache:
    kind: str
    values: torch.Tensor
    version: int
    token_ids: list[int]


def trim_cached_completion_padding(inputs):
    """Drop only the trailing padding unused by this accumulation microbatch.

    Rollouts are padded together before being shuffled and split. A single
    2048-token completion otherwise makes every microbatch project 2048 logits
    per row, including batches whose completions are only a few hundred tokens.
    Cache lengths are already on the CPU, so this needs no CUDA readback.
    """
    caches = inputs.get("full_teacher_cache")
    if not caches:
        return inputs
    length = max(1, max(len(cache.token_ids) for cache in caches))
    if length >= inputs["completion_ids"].shape[1]:
        return inputs
    trimmed = inputs.copy()
    for key in ("completion_ids", "completion_mask", "provenance", "advantages"):
        if key in trimmed:
            trimmed[key] = trimmed[key][:, :length]
    return trimmed


@torch.no_grad()
def full_teacher_logps(caches, head, completion_ids, version, temperature, device):
    if temperature <= 0 or len(caches) != len(completion_ids):
        raise ValueError("invalid cache batch or loss temperature")
    width = head.weight.shape[0]
    result = torch.zeros(
        (*completion_ids.shape, width), device=device, dtype=torch.float32
    )
    for i, cache in enumerate(caches):
        if cache.version != version:
            raise ValueError("teacher cache version differs from current EMA teacher")
        n = len(cache.token_ids)
        if len(cache.values) != n or completion_ids[i, :n].tolist() != cache.token_ids:
            raise ValueError("teacher cache token alignment mismatch")
        if n > completion_ids.shape[1]:
            raise ValueError("teacher cache exceeds completion length")
        for start in range(0, n, 64):
            value = cache.values[start : start + 64].to(device)
            if cache.kind == "hidden":
                value = head(value.to(head.weight.dtype))
            elif cache.kind != "logits":
                raise ValueError(f"unknown full cache kind: {cache.kind}")
            result[i, start : start + len(value)] = (
                value.float() / temperature
            ).log_softmax(-1)
    return result
