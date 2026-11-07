"""CPU ledger for optimistic JSD rollouts; only verified tokens are public."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


def eos_token_ids(primary, configured=None):
    """Match vLLM's primary EOS plus additional generation-config stop IDs."""
    extra = (
        []
        if configured is None
        else [configured]
        if isinstance(configured, int)
        else configured
    )
    return tuple(dict.fromkeys(t for t in [primary, *extra] if t is not None))


def uniform(seed: int, request_id: str, position: int) -> float:
    """A schedule-independent categorical draw, shared by both source models."""
    key = f"{seed}:{len(request_id)}:{request_id}:{position}".encode()
    value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
    return ((value >> 11) + 0.5) / (1 << 53)


@dataclass
class RolloutRequest:
    request_id: str
    teacher_prompt: list[int]
    student_prompt: list[int]
    max_tokens: int = 128
    stop_tokens: tuple[int, ...] = ()


def admission_batches(requests, max_requests, capacities, block_size):
    """Reserve complete continuations on each GPU; queue excess work in order."""
    batches, batch, used = [], [], {side: 0 for side in capacities}
    for request in requests:
        need = {}
        for side, capacity in capacities.items():
            prompt = request.teacher_prompt if side == 0 else request.student_prompt
            need[side] = (
                (len(prompt) + request.max_tokens + block_size - 1) // block_size
            ) * block_size
            if need[side] > capacity:
                raise ValueError(
                    f"request {request.request_id!r} exceeds GPU {side} KV capacity"
                )
        if batch and (
            len(batch) == max_requests
            or any(used[s] + need[s] > capacities[s] for s in capacities)
        ):
            batches.append(batch)
            batch, used = [], {side: 0 for side in capacities}
        batch.append(request)
        for side in capacities:
            used[side] += need[side]
    if batch:
        batches.append(batch)
    return batches


@dataclass
class Draft:
    request_id: str
    version: int
    start: int
    tokens: list[int]


@dataclass
class Lookahead:
    """Per-request AIMD lead limit; changes scheduling, never token policy.

    Start with two blocks, halve on a differing-token correction, and grow
    by one block after four clean blocks. Existing in-flight work is allowed
    to drain when the limit shrinks.
    """

    chunk: int
    maximum: int
    adaptive: bool = False
    clean_tokens: int = 0
    limit: int = field(init=False)

    def __post_init__(self):
        if self.chunk < 1 or self.maximum < 2 * self.chunk:
            raise ValueError("lookahead must allow at least two positive blocks")
        self.limit = 2 * self.chunk if self.adaptive else self.maximum

    def observe(self, committed: int, corrected: bool):
        if not self.adaptive:
            return
        if corrected:
            self.limit = max(
                2 * self.chunk, self.limit // (2 * self.chunk) * self.chunk
            )
            self.clean_tokens = 0
        else:
            self.clean_tokens += committed
            if self.clean_tokens >= 4 * self.chunk:
                self.limit = min(self.maximum, self.limit + self.chunk)
                self.clean_tokens = 0


@dataclass
class Ledger:
    request: RolloutRequest
    tokens: list[int] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    drafted: list[int] = field(default_factory=list)
    version: int = 0
    corrections: int = 0
    discarded: int = 0

    @property
    def done(self):
        return len(self.tokens) >= self.request.max_tokens or (
            bool(self.tokens) and self.tokens[-1] in self.request.stop_tokens
        )

    def append(self, draft):
        if draft.version != self.version or self.done:
            self.discarded += len(draft.tokens)
            return False
        if draft.start != len(self.drafted):
            raise ValueError("noncontiguous student draft")
        self.drafted.extend(draft.tokens)
        return True

    def apply(self, draft, chosen, sources):
        """Commit through the first correction or stop; invalidate its suffix."""
        if draft.version != self.version:
            return False
        if self.done or draft.start != len(self.tokens):
            raise ValueError("verification does not start at committed frontier")
        if not chosen or len(chosen) != len(sources) or len(chosen) > len(draft.tokens):
            raise ValueError("invalid verification length")
        changed = False
        for i, (token, source) in enumerate(zip(chosen, sources)):
            if source not in ("student", "teacher"):
                raise ValueError("unknown token source")
            if token != draft.tokens[i] and source != "teacher":
                raise ValueError("student verification changed a draft token")
            self.tokens.append(token)
            self.sources.append(source)
            changed = token != draft.tokens[i]
            if changed or self.done:
                break
        if not changed and not self.done and len(chosen) != len(draft.tokens):
            raise ValueError("partial verification without a correction")
        if changed or self.done:
            self.discarded += max(0, len(self.drafted) - len(self.tokens))
            self.drafted = self.tokens.copy()
            self.version += 1
            self.corrections += int(changed)
        return changed
