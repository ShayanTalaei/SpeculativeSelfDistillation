"""CPU-only paired scheduler. No CUDA imports or autonomous worker scheduling.

Capacity is measured in usable token slots; physical page zero is kept out of
the allocator. Reservations include the full declared continuation budget.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field, fields
from functools import cached_property


class ProtocolError(RuntimeError):
    pass


EMPTY_HISTORY_DIGEST = hashlib.sha256(b"").hexdigest()


def advance_history_digest(previous, token):
    return hashlib.sha256(
        bytes.fromhex(previous) + token.to_bytes(8, "little")
    ).hexdigest()


@dataclass(frozen=True)
class PairConfig:
    capacity: int = 16384
    block_size: int = 16
    token_budget: int = 512
    max_requests: int = 32
    reservation_mode: str = "pools"
    conditioning_site: str = "routing_only"
    prefix_caching: bool = False
    strict_pair_tail: int | None = None
    teacher_revision: str = "unspecified"
    student_revision: str = "unspecified"
    policy_revision: str = "fixed-v1"
    feature_width: int = 0
    feature_dtype: str = ""

    def __post_init__(self):
        if self.block_size not in (8, 16):
            raise ValueError("resolved allocation block size must be 8 or 16")
        if self.capacity <= 0 or self.capacity % self.block_size:
            raise ValueError("capacity must be a positive multiple of block_size")
        if min(self.token_budget, self.max_requests) <= 0:
            raise ValueError("budgets must be positive")
        if self.reservation_mode not in ("pools", "reservations"):
            raise ValueError("unknown reservation mode")
        if self.conditioning_site != "routing_only":
            raise ValueError(
                "initial adapter supports routing_only; trained conditioning requires its own adapter"
            )
        if self.prefix_caching:
            raise ValueError(
                "prefix caching is disabled until conditioned replay is validated"
            )
        if (
            self.strict_pair_tail is not None
            and 2 * (self.block_size - 1) > self.strict_pair_tail
        ):
            raise ValueError("resolved pair tail bound exceeds strict budget")


class PagePool:
    def __init__(self, capacity, block_size):
        self.capacity, self.block_size = capacity, block_size
        self.credits: dict[str, int] = {}
        self.pages: dict[str, list[int]] = {}
        self.block_digests: dict[str, str] = {}
        self.free = list(range(capacity // block_size, 0, -1))

    def rounded(self, tokens):
        return ((tokens + self.block_size - 1) // self.block_size) * self.block_size

    @property
    def reserved(self):
        return sum(self.credits.values())

    @property
    def unmapped_credit(self):
        return self.reserved - sum(map(len, self.pages.values())) * self.block_size

    def can_reserve(self, slots):
        return self.reserved + slots <= self.capacity

    def reserve(self, key, slots):
        if key in self.credits or not self.can_reserve(slots):
            raise ProtocolError("invalid reservation")
        self.credits[key], self.pages[key] = slots, []
        self.block_digests[key] = EMPTY_HISTORY_DIGEST

    def map(self, key, valid):
        required = self.rounded(valid)
        if required > self.credits[key]:
            raise ProtocolError("mapping exceeds reservation")
        pages = self.pages[key]
        delta = required // self.block_size - len(pages)
        if delta > len(self.free):
            raise ProtocolError("reserved credit was lent out")
        for _ in range(delta):
            page = self.free.pop()
            pages.append(page)
            self.block_digests[key] = advance_history_digest(
                self.block_digests[key], page
            )
        return tuple(pages)

    def release(self, key):
        self.free.extend(self.pages.pop(key, []))
        self.credits.pop(key, None)
        self.block_digests.pop(key, None)


@dataclass(frozen=True)
class PairedTokenEvent:
    request_id: str
    incarnation: int
    output_index: int
    token_id: int
    source_model: str
    source_checkpoint_revision: str
    policy_revision: str


@dataclass
class PairedRequest:
    request_id: str
    prompts: tuple[tuple[int, ...], tuple[int, ...]]
    max_tokens: int
    stop_tokens: tuple[int, ...]
    incarnation: int = 0
    committed: list[int] = field(default_factory=list)
    events: list[PairedTokenEvent] = field(default_factory=list)
    computed: list[int] = field(default_factory=lambda: [0, 0])
    status: str = "waiting"
    cancelled: bool = False
    replaying: bool = False
    history_digest: str = EMPTY_HISTORY_DIGEST


@dataclass(frozen=True)
class WorkRow:
    request_id: str
    incarnation: int
    output_index: int
    computed: int
    tokens: tuple[int, ...]
    blocks: tuple[int, ...]
    prompt: tuple[int, ...]
    prompt_length: int
    history: tuple[int, ...]
    max_tokens: int
    history_digest: str = EMPTY_HISTORY_DIGEST
    block_start: int = 0
    block_digest: str = EMPTY_HISTORY_DIGEST

    @property
    def key(self):
        return f"{self.request_id}:{self.incarnation}"


@dataclass(frozen=True)
class StepManifest:
    epoch: int
    step_id: int
    teacher: tuple[WorkRow, ...]
    student: tuple[WorkRow, ...]
    ready: tuple[tuple[str, int, int], ...]
    finished: tuple[str, ...] = ()
    contract: tuple[str, ...] = ()
    feature_shape: tuple[int, int] = (0, 0)

    @cached_property
    def digest(self):
        # Avoid recursive dataclass deepcopy of long prompts on every control
        # message. The manifest is immutable; serialize its values once.
        row_fields = fields(WorkRow)
        payload = (
            self.epoch,
            self.step_id,
            [tuple(getattr(r, f.name) for f in row_fields) for r in self.teacher],
            [tuple(getattr(r, f.name) for f in row_fields) for r in self.student],
            self.ready,
            self.finished,
            self.contract,
            self.feature_shape,
        )
        return hashlib.sha256(pickle.dumps(payload, protocol=5)).hexdigest()


class PairedEngineCore:
    """One owner and one in-flight transaction. Worker results cannot select work."""

    def __init__(self, config: PairConfig):
        self.config = config
        self.pools = [PagePool(config.capacity, config.block_size) for _ in range(2)]
        self.requests: dict[str, PairedRequest] = {}
        self.incarnations: dict[str, int] = {}
        self.epoch, self.step_id = 0, 0
        self.inflight: StepManifest | None = None
        self.phase = "RETIRED"
        self.proposal = {}
        self.events = []
        self.finished = []

    def add(
        self, request_id, teacher_prompt, student_prompt, max_tokens, stop_tokens=()
    ):
        if request_id in self.requests:
            raise ValueError("request identity already exists")
        if not teacher_prompt or not student_prompt or max_tokens <= 0:
            raise ValueError("nonempty prompts and positive output budget required")
        if any(type(t) is not int or t < 0 for t in [*teacher_prompt, *student_prompt]):
            raise ValueError("prompts must contain nonnegative token IDs")
        prompts = (tuple(teacher_prompt), tuple(student_prompt))
        if any(
            self.pools[0].rounded(len(p) + max_tokens) > self.config.capacity
            for p in prompts
        ):
            raise ValueError("request cannot fit full continuation reservation")
        incarnation = self.incarnations.get(request_id, -1) + 1
        self.incarnations[request_id] = incarnation
        self.requests[request_id] = PairedRequest(
            request_id, prompts, max_tokens, tuple(stop_tokens), incarnation=incarnation
        )

    def _admit(self):
        active = sum(r.status == "active" for r in self.requests.values())
        for r in self.requests.values():
            if r.status != "waiting" or active >= self.config.max_requests:
                continue
            sizes = [
                p.rounded(len(prompt) + r.max_tokens)
                for p, prompt in zip(self.pools, r.prompts)
            ]
            if self.config.reservation_mode == "reservations":
                sizes = [max(sizes)] * 2
            if not all(p.can_reserve(n) for p, n in zip(self.pools, sizes)):
                continue
            try:
                for p, n in zip(self.pools, sizes):
                    p.reserve(r.request_id, n)
            except Exception:
                for p in self.pools:
                    p.release(r.request_id)
                raise
            r.status = "active"
            active += 1

    def plan(self):
        if self.inflight is not None:
            raise ProtocolError("previous transaction has not retired")
        self._admit()
        # Existing decodes get service ahead of long prefills.
        active = sorted(
            (r for r in self.requests.values() if r.status == "active"),
            key=lambda r: not bool(r.committed),
        )
        rows = [[], []]
        after = {r.request_id: r.computed.copy() for r in active}
        for side in (0, 1):
            budget = self.config.token_budget
            for r in active:
                prompt = r.prompts[side]
                computed = r.computed[side]
                n = min(len(prompt) + len(r.committed) - computed, budget)
                if n <= 0:
                    continue
                valid = r.computed[side] + n
                query = (
                    prompt[computed : min(valid, len(prompt))]
                    if computed < len(prompt)
                    else ()
                )
                query += tuple(
                    r.committed[
                        max(0, computed - len(prompt)) : max(0, valid - len(prompt))
                    ]
                )
                block_start = len(self.pools[side].pages[r.request_id])
                blocks = self.pools[side].map(r.request_id, valid)
                rows[side].append(
                    WorkRow(
                        r.request_id,
                        r.incarnation,
                        len(r.committed),
                        r.computed[side],
                        query,
                        blocks[block_start:],
                        r.prompts[side] if r.computed[side] == 0 else (),
                        len(r.prompts[side]),
                        tuple(r.committed) if computed == 0 else (),
                        r.max_tokens,
                        r.history_digest,
                        block_start,
                        self.pools[side].block_digests[r.request_id],
                    )
                )
                after[r.request_id][side] = valid
                budget -= n
        ready = tuple(
            (r.request_id, r.incarnation, len(r.committed))
            for r in active
            if len(r.committed) < r.max_tokens
            and not (r.committed and r.committed[-1] in r.stop_tokens)
            if all(
                after[r.request_id][i] == len(r.prompts[i]) + len(r.committed)
                for i in (0, 1)
            )
        )
        self.step_id += 1
        self.inflight = StepManifest(
            self.epoch,
            self.step_id,
            tuple(rows[0]),
            tuple(rows[1]),
            ready,
            tuple(self.finished),
            (
                self.config.teacher_revision,
                self.config.student_revision,
                self.config.policy_revision,
                self.config.conditioning_site,
                self.config.feature_dtype,
            ),
            (len(ready), self.config.feature_width),
        )
        self.finished.clear()
        self.phase, self.proposal, self.events = "PLANNED", {}, []
        return self.inflight

    def _check(self, phase, *digests):
        if (
            not self.inflight
            or self.phase != phase
            or any(d != self.inflight.digest for d in digests)
        ):
            raise ProtocolError(f"stale or out-of-order message in {self.phase}")

    def prepared(self, teacher_digest, student_digest):
        self._check("PLANNED", teacher_digest, student_digest)
        self.phase = "PREPARED"

    def executed(self, teacher_digest, student_digest):
        self._check("PREPARED", teacher_digest, student_digest)
        self.phase = "EXECUTING"
        for side, rows in enumerate((self.inflight.teacher, self.inflight.student)):
            for row in rows:
                self.requests[row.request_id].computed[side] += len(row.tokens)

    def propose(self, digest, choices):
        self._check("EXECUTING", digest)
        expected = {r[0] for r in self.inflight.ready}
        if set(choices) != expected or any(
            type(t) is not int or t < 0 or s not in ("teacher", "student")
            for t, s in choices.values()
        ):
            raise ProtocolError(
                "proposal must contain exactly the ready requests and valid sources"
            )
        self.proposal = dict(choices)
        self.phase = "PROPOSED"

    def commit(self):
        if self.phase == "COMMITTED":
            return self.events
        self._check("PROPOSED")
        for rid, (token, source) in self.proposal.items():
            r = self.requests[rid]
            if r.cancelled:
                continue
            revision = (
                self.config.teacher_revision
                if source == "teacher"
                else self.config.student_revision
            )
            event = PairedTokenEvent(
                rid,
                r.incarnation,
                len(r.committed),
                token,
                source,
                revision,
                self.config.policy_revision,
            )
            r.committed.append(token)
            r.history_digest = advance_history_digest(r.history_digest, token)
            r.events.append(event)
            self.events.append(event)
        self.phase = "COMMITTED"
        return self.events

    def retire(self, teacher_digest, student_digest):
        self._check("COMMITTED", teacher_digest, student_digest)
        for r in self.requests.values():
            if any(e.request_id == r.request_id for e in self.events):
                r.replaying = False
            if (
                r.status == "active"
                and (
                    r.cancelled
                    or len(r.committed) >= r.max_tokens
                    or (r.committed and r.committed[-1] in r.stop_tokens)
                )
                and (
                    not r.replaying
                    or all(
                        r.computed[i] == len(r.prompts[i]) + len(r.committed)
                        for i in (0, 1)
                    )
                )
            ):
                self._release(r)
        self.inflight, self.phase = None, "RETIRED"

    def _release(self, r):
        for p in self.pools:
            p.release(r.request_id)
        r.status = "cancelled" if r.cancelled else "finished"
        self.finished.append(f"{r.request_id}:{r.incarnation}")

    def cancel(self, request_id):
        r = self.requests[request_id]
        r.cancelled = True
        if self.inflight is None or r.status == "waiting":
            self._release(r)

    def recover(self):
        """Call only AFTER stopping both workers; a new epoch cannot reuse buffers."""
        self.epoch += 1
        self.pools = [
            PagePool(self.config.capacity, self.config.block_size) for _ in range(2)
        ]
        for r in self.requests.values():
            if r.status == "active":
                if r.cancelled:
                    self._release(r)
                else:
                    r.incarnation += 1
                    self.incarnations[r.request_id] = r.incarnation
                    r.computed = [0, 0]
                    r.status = "waiting"
                    r.replaying = True
        self.inflight, self.phase, self.finished = None, "RETIRED", []

    @property
    def unfinished(self):
        return any(r.status in ("waiting", "active") for r in self.requests.values())
