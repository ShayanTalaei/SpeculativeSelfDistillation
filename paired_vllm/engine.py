"""Spawned worker lifecycle and the single paired transaction control loop."""

from __future__ import annotations

import math
import multiprocessing as mp
import socket
import time
from dataclasses import dataclass, field, replace
from multiprocessing.connection import wait
from queue import Empty, SimpleQueue

from paired_vllm.core import PairConfig, PairedEngineCore, ProtocolError
from paired_vllm.worker import process_main


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@dataclass(frozen=True)
class RuntimeConfig:
    teacher_model: str
    student_model: str
    scheduler: PairConfig = field(default_factory=PairConfig)
    teacher_gpu: str = "0"
    student_gpu: str = "1"
    policy: str = "alternating"
    dtype: str = "float16"
    max_model_len: int = 8192
    memory_utilization: float = 0.8
    timeout: float = 120
    startup_timeout: float = 600
    teacher_only: bool = False
    graphs: bool = False
    head_graphs: bool = True
    temperature: float = 0.0
    top_k: int = 0
    capture_targets: bool = False
    target_format: str = "hidden"
    target_topk: int = 100
    target_temperature: float = 1.0
    enable_sleep_mode: bool = False
    keep_teacher_resident: bool = False
    compile_model: bool = False
    gpu_target_cache: bool = False
    fused_dispatch: bool = False

    def __post_init__(self):
        if self.compile_model and not self.graphs:
            raise ValueError("model compilation requires graphs in this adapter")
        if self.gpu_target_cache and self.target_format != "hidden":
            raise ValueError("GPU target buffering supports hidden states only")
        if self.target_format not in ("hidden", "logits", "topk"):
            raise ValueError("unsupported target format")
        if self.target_topk < 1 or self.target_temperature <= 0:
            raise ValueError("invalid target cache configuration")
        if (
            not math.isfinite(self.temperature)
            or self.temperature < 0
            or self.top_k < 0
        ):
            raise ValueError("invalid sampling temperature/top_k")
        if not self.teacher_only and self.teacher_gpu == self.student_gpu:
            raise ValueError("teacher and student require separate physical GPUs")
        if self.policy not in ("teacher", "student", "alternating"):
            prefix, _, threshold = self.policy.partition(":")
            if (
                prefix not in ("kl_forward", "kl_reverse", "jsd")
                or not threshold
                or not math.isfinite(float(threshold))
                or float(threshold) < 0
            ):
                raise ValueError("unsupported routing policy")
        if self.max_model_len > self.scheduler.capacity:
            raise ValueError("capacity must fit max_model_len")
        if self.scheduler.token_budget < self.scheduler.max_requests:
            raise ValueError("MRV2 token_budget must be at least max_requests")
        if min(self.timeout, self.startup_timeout) <= 0:
            raise ValueError("worker timeouts must be positive")

    @property
    def policy_revision(self):
        if self.temperature == 0 and self.top_k == 0:
            return self.policy
        return f"{self.policy};temperature={self.temperature};top_k={self.top_k}"

    @property
    def compilation_config(self):
        if not self.graphs:
            return {}
        maximum = self.scheduler.max_requests
        return {
            "mode": 3 if self.compile_model else 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": sorted(
                {1, maximum, *(n for n in (2, 4, 8, 16) if n <= maximum)}
            ),
        }


class PairedEngine:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.workers, self.connections = [], []
        self.metrics = []
        self.closed = False
        self.weights_ready = True
        self.recovery_events = []
        self.cancellations = SimpleQueue()
        try:
            self._start()
        except BaseException:
            self.close()
            raise

    def _start(self):
        started = time.perf_counter()
        config = self.config
        ctx = mp.get_context("spawn")
        pair_port = None if config.teacher_only else free_port()
        for side, gpu in enumerate(
            (config.teacher_gpu,)
            if config.teacher_only
            else (config.teacher_gpu, config.student_gpu)
        ):
            parent, child = ctx.Pipe()
            process = ctx.Process(
                target=process_main,
                args=(
                    child,
                    config,
                    side,
                    gpu,
                    f"tcp://127.0.0.1:{free_port()}",
                    pair_port,
                ),
                daemon=True,
            )
            process.start()
            child.close()
            self.workers.append(process)
            self.connections.append(parent)
        metadata = self._receive("BOOT", config.startup_timeout)
        if len(metadata) == 2:
            if (
                metadata[0]["checkpoint_revision"]
                != metadata[1]["teacher_projection_revision"]
            ):
                raise ValueError(
                    "copied teacher head and teacher body have different checkpoint revisions"
                )
            if (
                metadata[0]["tokenizer_digest"] != metadata[1]["tokenizer_digest"]
                or metadata[0]["vocab"] != metadata[1]["vocab"]
            ):
                raise ValueError("models do not share output token-ID semantics")
            if metadata[0]["gpu_uuid"] == metadata[1]["gpu_uuid"]:
                raise ValueError("workers resolved to the same physical GPU")
        capacity = min(
            config.scheduler.capacity, *(m["feasible_slots"] for m in metadata)
        )
        if capacity < config.max_model_len:
            raise ValueError("equal feasible KV capacity cannot hold max_model_len")
        scheduler_config = replace(
            config.scheduler,
            capacity=capacity,
            teacher_revision=metadata[0]["checkpoint_revision"],
            student_revision=metadata[-1]["checkpoint_revision"],
            policy_revision=config.policy_revision,
            feature_width=metadata[0]["feature_contract"]["width"],
            feature_dtype=metadata[0]["feature_contract"]["dtype"],
        )
        self.core = PairedEngineCore(scheduler_config)
        self._send("INIT", capacity)
        self.metadata = self._receive("INIT", config.startup_timeout)
        self.startup_seconds = time.perf_counter() - started

    def _send(self, command, payload):
        for conn in self.connections:
            conn.send((command, payload))

    def _receive(self, command, timeout=None):
        deadline = time.monotonic() + (timeout or self.config.timeout)
        pending = set(range(len(self.connections)))
        results = [None] * len(self.connections)
        while pending:
            self._drain_cancellations()
            ready = wait(
                [self.connections[i] for i in pending],
                timeout=min(0.1, max(0, deadline - time.monotonic())),
            )
            for i in list(pending):
                conn = self.connections[i]
                if conn in ready:
                    kind, result = conn.recv()
                    if kind == "ERROR":
                        raise RuntimeError(f"worker {i} failed:\n{result}")
                    if kind != command:
                        raise ProtocolError(f"expected {command}, received {kind}")
                    results[i] = result
                    pending.remove(i)
                elif not self.workers[i].is_alive():
                    raise RuntimeError(f"worker {i} exited during {command}")
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"paired {command} timed out; entire epoch must be reset"
                )
        return results

    def _both(self, values):
        return values if len(values) == 2 else values * 2

    def cancel(self, request_id):
        """Thread-safe cancellation submission; the scheduler applies it."""
        self.cancellations.put(request_id)

    def _drain_cancellations(self):
        if not hasattr(self, "core"):
            return
        while True:
            try:
                request_id = self.cancellations.get_nowait()
            except Empty:
                return
            request = self.core.requests.get(request_id)
            if request is not None and request.status in ("waiting", "active"):
                self.core.cancel(request_id)

    def add(
        self, request_id, teacher_prompt, student_prompt, max_tokens, stop_tokens=()
    ):
        if not self.weights_ready:
            raise ProtocolError(
                "reload weights after training sleep before admitting requests"
            )
        if (
            max(len(teacher_prompt), len(student_prompt)) + max_tokens
            > self.config.max_model_len
        ):
            raise ValueError("request exceeds configured max_model_len")
        if self.config.teacher_only:
            student_prompt = teacher_prompt
        self.core.add(
            request_id, teacher_prompt, student_prompt, max_tokens, stop_tokens
        )

    def step(self):
        """Return events only after both GPU histories and buffer lifetimes retire."""
        if self.closed:
            raise RuntimeError("engine is closed")
        started = time.perf_counter()
        self._drain_cancellations()
        manifest = self.core.plan()
        # Compute once before IPC; immutable manifests carry the cached digest
        # to both workers instead of hashing the same rows three times.
        _ = manifest.digest
        accounting = [
            {
                "reserved_slots": p.reserved,
                "unmapped_credit_slots": p.unmapped_credit,
                "mapped_slots": sum(map(len, p.pages.values())) * p.block_size,
                "valid_tokens_after_forward": sum(
                    r.computed[i]
                    for r in self.core.requests.values()
                    if r.status == "active"
                )
                + sum(
                    len(row.tokens)
                    for row in (manifest.teacher if i == 0 else manifest.student)
                ),
            }
            for i, p in enumerate(self.core.pools)
        ]
        try:
            if self.config.fused_dispatch:
                # Each worker validates before queuing its forward. Any failed
                # validation closes BOTH peers, including one waiting in NCCL.
                # No tokens or KV credits are committed before both proposals.
                prepared = time.perf_counter()
                self._send("PREPARE_RUN", manifest)
                proposed = self._receive("PREPARE_RUN")
                self.core.prepared(*self._both([r["digest"] for r in proposed]))
            else:
                self._send("PREPARE", manifest)
                self.core.prepared(*self._both(self._receive("PREPARE")))
                prepared = time.perf_counter()
                # Issue both EXECUTEs before waiting on either worker.
                self._send("RUN", manifest.digest)
                proposed = self._receive("RUN")
            self.core.executed(*self._both([r["digest"] for r in proposed]))
            if any(p["digest"] != manifest.digest for p in proposed):
                raise ProtocolError("proposal digest mismatch")
            self.core.propose(manifest.digest, proposed[-1]["choices"])
            events = self.core.commit()
            decided = time.perf_counter()
            self._send("FINALIZE", (manifest.digest, events))
            applied = self._receive("FINALIZE")
            if any(d != manifest.digest for d in applied):
                raise ProtocolError("commit acknowledgement mismatch")
            self.core.retire(*self._both(applied))
        except BaseException:
            # Never leave one peer alive in the old NCCL sequence.
            self.close()
            raise
        ended = time.perf_counter()
        self.metrics.append(
            {
                "step_id": manifest.step_id,
                "seconds": ended - started,
                "prepare_seconds": prepared - started,
                "execute_route_seconds": decided - prepared,
                "commit_seconds": ended - decided,
                "teacher_tokens": sum(len(r.tokens) for r in manifest.teacher),
                "student_tokens": sum(len(r.tokens) for r in manifest.student),
                "outputs": len(events),
                "capacity_accounting": accounting,
                "worker_metrics": [
                    {k: v for k, v in p.items() if k not in ("choices", "digest")}
                    for p in proposed
                ],
            }
        )
        replayed = []
        for event in self.recovery_events:
            r = self.core.requests[event.request_id]
            if r.status == "finished" or not r.replaying:
                replayed.append(event)
        self.recovery_events = [e for e in self.recovery_events if e not in replayed]
        return replayed + events

    def recover(self):
        """Explicit bounded epoch reset; force committed IDs during reconstruction.

        Scheduler state is in memory. A scheduler-process crash is not recoverable
        without an application-provided durable journal.
        """
        if (
            self.config.temperature
            or self.config.capture_targets
            or getattr(self, "training_snapshot", False)
        ):
            raise ProtocolError(
                "training/stochastic recovery requires saved weights, target cache and RNG state; restart the rollout"
            )
        old_core = self.core
        if old_core.phase == "COMMITTED":
            self.recovery_events.extend(
                e for e in old_core.events if e not in self.recovery_events
            )
        self.close()
        old_core.recover()
        self.workers, self.connections = [], []
        self.closed = False
        try:
            self._start()
            if self.core.config.capacity != old_core.config.capacity:
                raise RuntimeError(
                    "recovered capacity changed; cannot preserve reservations"
                )
            for name in (
                "teacher_revision",
                "student_revision",
                "policy_revision",
                "feature_width",
                "feature_dtype",
            ):
                if getattr(self.core.config, name) != getattr(old_core.config, name):
                    raise RuntimeError(
                        f"recovered {name} changed; replay would be invalid"
                    )
            self.core = old_core
        except BaseException:
            self.close()
            raise

    def run(self):
        events = []
        while self.core.unfinished:
            events.extend(self.step())
        return events

    def take_finished(self):
        """Release CPU records after the caller has consumed their outputs."""
        if self.core.inflight is not None or self.recovery_events:
            raise ProtocolError("cannot discard records before retirement/replay")
        finished = {
            key: r
            for key, r in self.core.requests.items()
            if r.status in ("finished", "cancelled")
        }
        for key in finished:
            del self.core.requests[key]
        return finished

    def inspect(self):
        self._send("INSPECT", None)
        return self._receive("INSPECT")

    def training_command(self, command, payload=None):
        if self.core.unfinished or self.core.inflight is not None:
            raise ProtocolError("training operations require a fully retired rollout")
        if command not in ("SYNC", "SLEEP", "WAKE", "TARGETS", "SAMPLING"):
            raise ValueError("unknown training command")
        if command == "SAMPLING":
            updated = replace(
                self.config,
                policy=payload["policy"],
                temperature=payload["temperature"],
                top_k=payload["top_k"],
            )
        try:
            self._send(command, payload)
            result = self._receive(command, self.config.startup_timeout)
        except BaseException:
            self.close()
            raise
        if command == "SLEEP":
            self.weights_ready = False
        if command == "SAMPLING":
            self.config = updated
            self.core.config = replace(
                self.core.config, policy_revision=updated.policy_revision
            )
        if command == "SYNC":
            self.weights_ready = True
            self.training_snapshot = True
            self.core.config = replace(
                self.core.config,
                teacher_revision=payload["version"],
                student_revision=payload["version"],
            )
        return result

    def close(self):
        self.closed = True
        for conn in self.connections:
            try:
                conn.send(("STOP", None))
            except (OSError, EOFError, BrokenPipeError):
                pass
        for process in self.workers:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
        for conn in self.connections:
            conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
