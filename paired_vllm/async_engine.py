"""Bounded asynchronous student/teacher rollout pipeline on two independent GPUs."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from multiprocessing.connection import wait
from typing import ClassVar

from paired_vllm.async_core import Ledger, Lookahead, RolloutRequest, admission_batches
from paired_vllm.engine import RuntimeConfig, free_port


@dataclass(frozen=True)
class AsyncConfig:
    supports_training: ClassVar[bool] = False
    score_policies: ClassVar[tuple[str, ...]] = ("jsd",)
    runtime: RuntimeConfig
    chunk_size: int = 10
    max_ahead_chunks: int = 3
    seed: int = 42
    student_only: bool = False
    overlap_prefill: bool = True
    fast_draft: bool = True
    reuse_metadata: bool = True
    fast_verify: bool = True
    verify_backend: str = "FLASH_ATTN"
    verify_body_graphs: bool = False
    adaptive_lookahead: bool = False

    def __post_init__(self):
        if self.chunk_size < 1 or self.max_ahead_chunks < 2:
            raise ValueError(
                "positive chunk size and at least two lookahead chunks required"
            )
        if self.verify_backend not in ("FLASH_ATTN", "TRITON_ATTN"):
            raise ValueError("unsupported asynchronous verifier attention backend")
        if self.runtime.policy.partition(":")[0] not in self.score_policies and not (
            self.supports_training and self.runtime.policy in ("student", "teacher")
        ):
            raise ValueError("unsupported routing policy for this engine")
        if self.runtime.capture_targets and self.runtime.target_format != "hidden":
            raise ValueError(
                "asynchronous target capture supports post-norm hidden states"
            )
        if (
            self.chunk_size * self.runtime.scheduler.max_requests
            > self.runtime.scheduler.token_budget
        ):
            raise ValueError("token budget must fit a full verification batch")
        if self.runtime.enable_sleep_mode and not self.supports_training:
            raise ValueError(
                "asynchronous engine does not yet support sleep or training weight refresh"
            )

    @property
    def threshold(self):
        return float(self.runtime.policy.split(":")[1])


class AsyncPairedEngine:
    """Fixed-admission batches, with independently advancing requests inside each.

    generate() exposes only verified tokens. Model weights stay fixed for the
    engine lifetime. The synchronous PairedEngine remains the training adapter.
    """

    def __init__(self, config: AsyncConfig, *, worker_entrypoint=None):
        if worker_entrypoint is None:
            from paired_vllm.async_worker import process_main as worker_entrypoint

        self.config = config
        self.connections, self.workers, self.expected = {}, {}, {}
        self.controls = {}
        self.metadata = {}
        self.closed = False
        self.last_metrics = {}
        started = time.perf_counter()
        ctx = mp.get_context("spawn")
        port = None if config.student_only else free_port()
        runtime = config.runtime
        try:
            for side in [1] if config.student_only else [0, 1]:
                parent, child = ctx.Pipe()
                control_child, control_parent = ctx.Pipe(duplex=False)
                gpu = runtime.teacher_gpu if side == 0 else runtime.student_gpu
                process = ctx.Process(
                    target=worker_entrypoint,
                    args=(
                        child,
                        config,
                        side,
                        gpu,
                        f"tcp://127.0.0.1:{free_port()}",
                        port,
                        control_child,
                    ),
                )
                process.start()
                child.close()
                control_child.close()
                self.controls[side] = control_parent
                self.connections[side], self.workers[side] = parent, process
                self.expected[side] = ["BOOT"]
            self.metadata = self._drain(runtime.startup_timeout)
            if len(self.metadata) == 2:
                if self.metadata[0]["gpu_uuid"] == self.metadata[1]["gpu_uuid"]:
                    raise ValueError("teacher and student resolved to the same GPU")
                if (
                    self.metadata[0]["tokenizer_digest"]
                    != self.metadata[1]["tokenizer_digest"]
                ):
                    raise ValueError("teacher/student tokenizer semantics differ")
            capacity = min(
                runtime.scheduler.capacity,
                *(m["feasible_slots"] for m in self.metadata.values()),
            )
            if capacity < runtime.max_model_len:
                raise ValueError("insufficient KV capacity")
            for side in self.connections:
                self._send(side, "INIT", capacity)
            self.metadata = self._drain(runtime.startup_timeout)
        except BaseException:
            self.close()
            raise
        self.startup_seconds = time.perf_counter() - started

    def _send(self, side, command, payload):
        if os.getenv("PAIRED_ASYNC_DEBUG"):
            print(
                "async send",
                side,
                command,
                payload if command == "DRAFT" else "",
                flush=True,
            )
        self.connections[side].send((command, payload))
        self.expected[side].append(command)

    def _receive(self, side):
        command, result = self.connections[side].recv()
        if os.getenv("PAIRED_ASYNC_DEBUG"):
            print(
                "async receive",
                side,
                command,
                result if command in ("DRAFT", "VERIFY") else "",
                flush=True,
            )
        if command == "ERROR":
            raise RuntimeError(f"async worker {side} failed:\n{result}")
        if not self.expected[side] or self.expected[side].pop(0) != command:
            raise RuntimeError("unexpected worker reply")
        return command, result

    def _drain(self, timeout=None):
        deadline = time.monotonic() + (timeout or self.config.runtime.timeout)
        results = {}
        while any(self.expected.values()):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("asynchronous worker deadline exceeded")
            ready = wait(
                [self.connections[s] for s in self.connections if self.expected[s]],
                timeout=min(remaining, 1),
            )
            for side, conn in self.connections.items():
                if conn in ready:
                    _, results[side] = self._receive(side)
        return results

    def _validate(self, requests):
        runtime = self.config.runtime
        if not requests:
            raise ValueError("at least one request is required")
        if len({r.request_id for r in requests}) != len(requests):
            raise ValueError("duplicate request IDs")
        vocab = self.metadata[1]["vocab"]
        for r in requests:
            if r.max_tokens < 1:
                raise ValueError("max_tokens must be positive")
            for prompt in (r.teacher_prompt, r.student_prompt):
                if not prompt or len(prompt) + r.max_tokens > runtime.max_model_len:
                    raise ValueError("empty or overlong context")
                if any(not isinstance(t, int) or not 0 <= t < vocab for t in prompt):
                    raise ValueError("prompt token outside vocabulary")
            if any(not 0 <= t < vocab for t in r.stop_tokens):
                raise ValueError("stop token outside vocabulary")
        return admission_batches(
            requests,
            runtime.scheduler.max_requests,
            {s: m["capacity"] for s, m in self.metadata.items()},
            runtime.scheduler.block_size,
        )

    def generate(self, requests: list[RolloutRequest]):
        if self.closed:
            raise RuntimeError("engine is closed")
        batches = self._validate(requests)
        self._last_targets = {}
        self.last_request_metrics = {}
        started = time.perf_counter()
        try:
            outputs, metrics = [], []
            target_export_seconds = 0.0
            for batch in batches:
                outputs.extend(self._generate(batch))
                metrics.append(self.last_metrics)
                if self.config.runtime.capture_targets and not self.config.student_only:
                    export_started = time.perf_counter()
                    self._last_targets.update(self._collect_targets())
                    target_export_seconds += time.perf_counter() - export_started
            self.last_metrics = {
                key: sum(m[key] for m in metrics) for key in metrics[0]
            }
            self.last_metrics["max_unverified_tokens"] = max(
                m["max_unverified_tokens"] for m in metrics
            )
            self.last_metrics["seconds"] = time.perf_counter() - started
            self.last_metrics["tokens_per_second"] = (
                self.last_metrics["verified_tokens"] / self.last_metrics["seconds"]
            )
            self.last_metrics["admission_batches"] = len(batches)
            self.last_metrics["target_export_seconds"] = target_export_seconds
            return outputs
        except BaseException:
            # A peer may be waiting in NCCL; never leave it alive on failure.
            self.close()
            raise

    def _collect_targets(self):
        self._send(0, "TARGETS", None)
        return self._drain()[0]

    def _generate(self, requests):
        started = time.perf_counter()
        for side in self.connections:
            self._send(side, "BEGIN", requests)
            self._send(side, "PREFILL", None)
        prefill_finished = {}
        if not self.config.overlap_prefill:
            self._drain()
            prefill_finished = {side: time.perf_counter() for side in self.connections}
        ledger = {r.request_id: Ledger(r) for r in requests}
        pending, corrections = [], {}
        chunk = self.config.chunk_size
        max_ahead = chunk * self.config.max_ahead_chunks
        lookaheads = {
            rid: Lookahead(chunk, max_ahead, self.config.adaptive_lookahead)
            for rid in ledger
        }
        finished_at = {}
        metrics = {
            "drafted_tokens": 0,
            "interrupted_tokens": 0,
            "verification_positions": 0,
            "verification_seconds": 0.0,
            "overlapping_student_chunks": 0,
            "overlapping_student_launches": 0,
            "student_chunks": 0,
            "student_forward_steps": 0,
            "student_available_rows": 0,
            "student_metadata_updates": 0,
            "max_unverified_tokens": 0,
            "student_wait_seconds": 0.0,
            "lead_limit_wait_seconds": 0.0,
            "student_chunks_during_teacher_prefill": 0,
            "first_student_draft_seconds": 0.0,
            "lookahead_grows": 0,
            "lookahead_shrinks": 0,
        }
        deadline = time.monotonic() + self.config.runtime.timeout
        while not all(r.done for r in ledger.values()) or any(self.expected.values()):
            if time.monotonic() > deadline:
                raise TimeoutError("asynchronous rollout made no token progress")
            pending = [
                d
                for d in pending
                if d.version == ledger[d.request_id].version
                and not ledger[d.request_id].done
            ]
            # One ordered verification per request at a time. Export snapshots
            # are queued before any correction can overwrite student features.
            if not self.config.student_only and not self.expected[0] and pending:
                batch, remaining, seen = [], [], set()
                for d in pending:
                    if d.request_id not in seen and d.start == len(
                        ledger[d.request_id].tokens
                    ):
                        batch.append(d)
                        seen.add(d.request_id)
                    else:
                        remaining.append(d)
                if batch:
                    pending = remaining
                    self._send(1, "EXPORT", batch)
                    self._send(0, "VERIFY", batch)
            runnable = []
            for rid, row in ledger.items():
                if row.done or len(row.drafted) >= row.request.max_tokens:
                    continue
                if row.drafted and row.drafted[-1] in row.request.stop_tokens:
                    continue  # Tentative EOS still needs teacher verification.
                if len(row.drafted) - len(row.tokens) < lookaheads[rid].limit:
                    runnable.append(rid)
            # EXPORT only queues a snapshot/send on a separate CUDA stream.
            # Queue the next draft behind it immediately instead of paying an
            # extra coordinator round trip for its acknowledgement.
            if self.expected[1] in ([], ["EXPORT"]) and runnable:
                if not metrics["first_student_draft_seconds"]:
                    metrics["first_student_draft_seconds"] = (
                        time.perf_counter() - started
                    )
                metrics["student_chunks_during_teacher_prefill"] += int(
                    not self.config.student_only and 0 not in prefill_finished
                )
                metrics["overlapping_student_launches"] += int(
                    bool(self.expected.get(0))
                )
                limits = {
                    rid: min(
                        chunk,
                        lookaheads[rid].limit
                        - len(ledger[rid].drafted)
                        + len(ledger[rid].tokens),
                    )
                    for rid in runnable
                }
                self._send(1, "DRAFT", (limits, chunk, corrections))
                corrections = {}
            if not any(self.expected.values()):
                if all(r.done for r in ledger.values()):
                    break
                raise RuntimeError("asynchronous rollout made no progress")
            waiting_on_teacher = not self.expected[1] and bool(self.expected.get(0))
            before_wait = time.perf_counter()
            ready = wait(
                [self.connections[s] for s in self.connections if self.expected[s]],
                timeout=1,
            )
            if waiting_on_teacher:
                waited = time.perf_counter() - before_wait
                metrics["student_wait_seconds"] += waited
                if any(
                    not row.done
                    and len(row.drafted) - len(row.tokens) >= lookaheads[rid].limit
                    for rid, row in ledger.items()
                ):
                    metrics["lead_limit_wait_seconds"] += waited
            if not ready:
                if time.monotonic() > deadline:
                    raise TimeoutError("asynchronous rollout worker stalled")
                continue
            # Process corrections before completed drafts from the abandoned branch.
            for side in (0, 1):
                if side not in self.connections or self.connections[side] not in ready:
                    continue
                command, result = self._receive(side)
                if command == "PREFILL":
                    prefill_finished[side] = time.perf_counter()
                elif command == "DRAFT":
                    if result["generated_tokens"]:
                        deadline = time.monotonic() + self.config.runtime.timeout
                    metrics["student_chunks"] += 1
                    metrics["student_forward_steps"] += result.get("forward_steps", 0)
                    metrics["student_available_rows"] += result.get("available_rows", 0)
                    metrics["student_metadata_updates"] += result.get(
                        "metadata_updates", 0
                    )
                    metrics["overlapping_student_chunks"] += int(
                        bool(self.expected.get(0))
                    )
                    metrics["drafted_tokens"] += result["generated_tokens"]
                    metrics["interrupted_tokens"] += result["interrupted_tokens"]
                    for draft in result["drafts"]:
                        row = ledger[draft.request_id]
                        if row.append(draft):
                            metrics["max_unverified_tokens"] = max(
                                metrics["max_unverified_tokens"],
                                len(row.drafted) - len(row.tokens),
                            )
                            if self.config.student_only:
                                row.apply(
                                    draft, draft.tokens, ["student"] * len(draft.tokens)
                                )
                                if row.done:
                                    finished_at[draft.request_id] = (
                                        time.perf_counter() - started
                                    )
                            else:
                                pending.append(draft)
                elif command == "VERIFY":
                    deadline = time.monotonic() + self.config.runtime.timeout
                    metrics["verification_seconds"] += result["seconds"]
                    metrics["verification_positions"] += result["positions"]
                    for draft, chosen, sources, _ in result["rows"]:
                        row = ledger[draft.request_id]
                        previous = len(row.tokens)
                        corrected = row.apply(draft, chosen, sources)
                        if row.done:
                            finished_at[draft.request_id] = (
                                time.perf_counter() - started
                            )
                        lead = lookaheads[draft.request_id]
                        old_limit = lead.limit
                        lead.observe(len(row.tokens) - previous, corrected)
                        metrics["lookahead_grows"] += int(lead.limit > old_limit)
                        metrics["lookahead_shrinks"] += int(lead.limit < old_limit)
                        if corrected:
                            self.controls[1].send(
                                (draft.request_id, row.version, row.tokens.copy())
                            )
        duration = time.perf_counter() - started
        self.last_request_metrics = {
            **getattr(self, "last_request_metrics", {}),
            **{
                rid: {
                    "corrections": row.corrections,
                    "discarded_published_tokens": row.discarded,
                    "finished_seconds": finished_at[rid],
                    "final_lookahead_tokens": lookaheads[rid].limit,
                }
                for rid, row in ledger.items()
            },
        }
        tokens = sum(len(r.tokens) for r in ledger.values())
        self.last_metrics = {
            **metrics,
            "seconds": duration,
            "prefill_seconds": max(prefill_finished.values()) - started,
            "student_prefill_seconds": prefill_finished[1] - started,
            "teacher_prefill_seconds": prefill_finished.get(0, started) - started,
            "verified_tokens": tokens,
            "tokens_per_second": tokens / duration,
            "corrections": sum(r.corrections for r in ledger.values()),
            "discarded_tokens": sum(r.discarded for r in ledger.values())
            + metrics["interrupted_tokens"],
            "teacher_tokens": sum(r.sources.count("teacher") for r in ledger.values()),
            "completion_tail_seconds": max(finished_at.values())
            - min(finished_at.values()),
        }
        return [
            {"request_id": rid, "token_ids": r.tokens, "sources": r.sources}
            for rid, r in ledger.items()
        ]

    def targets(self):
        if not self.config.runtime.capture_targets or self.config.student_only:
            raise ValueError("teacher target capture is disabled")
        return self._last_targets

    def close(self):
        if self.closed:
            return
        self.closed = True
        for conn in self.connections.values():
            try:
                conn.send(("STOP", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.workers.values():
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join(timeout=3)
        for conn in self.connections.values():
            conn.close()
        for conn in self.controls.values():
            conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
