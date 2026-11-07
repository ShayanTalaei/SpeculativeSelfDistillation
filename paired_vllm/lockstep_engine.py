"""Non-speculative paired inference with GPU-to-GPU definitive-token feedback.

Teacher and student forward in parallel. Teacher states go to the student GPU;
full JSD, keyed sampling and the final token are computed there. Both native
histories advance on device. CPU publication/EOS polling is blockwise, but every
generated token is definitive before either model consumes it.
"""

from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass, replace
from typing import ClassVar

from paired_vllm.async_core import uniform
from paired_vllm.async_engine import AsyncConfig, AsyncPairedEngine
from paired_vllm.async_worker import AsyncWorker
from paired_vllm.worker import load_teacher_projection


@dataclass(frozen=True)
class LockstepConfig(AsyncConfig):
    supports_training: ClassVar[bool] = True
    score_policies: ClassVar[tuple[str, ...]] = ("jsd", "kl_forward", "kl_reverse")
    readback_interval: int = 8
    transfer_logits: bool = False
    fused_jsd: bool = False
    resident_targets: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.readback_interval < 1:
            raise ValueError("readback interval must be positive")


class LockstepWorker(AsyncWorker):
    def evaluate(self, payload):
        from paired_vllm.student_eval import evaluate_student

        return evaluate_student(self, payload)

    def __init__(self, config, side, rendezvous, port):
        super().__init__(config, side, rendezvous, port)
        # Reuse native initialization, KV reservations and prefill, but reverse
        # the feature direction. Release the asynchronous-only allocations
        # before initializing the KV cache; no second model body is loaded.
        for name in ("features", "student_projection"):
            value = getattr(self, name)
            if value is not None:
                self.available_memory += value.numel() * value.element_size()
                setattr(self, name, None)
        from transformers import AutoConfig

        teacher_config = AutoConfig.from_pretrained(config.runtime.teacher_model)
        self.teacher_width = teacher_config.hidden_size
        self.teacher_projection = None
        if side == 1 and not config.transfer_logits:
            self.teacher_projection, _ = load_teacher_projection(
                config.runtime.teacher_model,
                self.config.model_config.dtype,
                teacher_config,
            )
            self.available_memory -= (
                self.teacher_projection.numel() * self.teacher_projection.element_size()
            )
        self.target_buffer = None
        self.target_exports = {}
        self.snapshot_version = "initial"
        if side == 0 and config.runtime.capture_targets:
            self.target_buffer = self.torch.empty(
                (
                    self.settings.scheduler.max_requests,
                    self.settings.max_model_len,
                    self.teacher_width,
                ),
                device="cuda",
                dtype=self.config.model_config.dtype,
            )
            self.available_memory -= (
                self.target_buffer.numel() * self.target_buffer.element_size()
            )
        self.metadata.update(
            feasible_slots=max(
                0, self.available_memory // self.metadata["page_bytes"] - 1
            )
            * self.settings.scheduler.block_size,
            async_student_projection_bytes=0,
            async_hidden_buffer_bytes=0,
            lockstep_teacher_projection_bytes=(
                self.teacher_projection.numel() * self.teacher_projection.element_size()
                if self.teacher_projection is not None
                else 0
            ),
            lockstep_feature_bytes_per_token=(
                self.metadata["vocab"] * 4
                if config.transfer_logits
                else self.teacher_width
                * self.torch.empty(
                    (), dtype=self.config.model_config.dtype
                ).element_size()
            ),
            lockstep_transfer_logits=config.transfer_logits,
        )

    def init(self, capacity):
        metadata = super().init(capacity)
        from paired_vllm.lockstep_head import LockstepHead

        torch = self.torch
        shape = (self.settings.scheduler.max_requests, self.settings.max_model_len)
        self.output_tokens = torch.zeros(shape, device="cuda", dtype=torch.int64)
        self.output_sources = torch.zeros(shape, device="cuda", dtype=torch.bool)
        self.uniforms = torch.zeros(shape, device="cuda", dtype=torch.float64)
        self.lockstep_head = LockstepHead(self)
        torch.cuda.synchronize()
        return metadata

    def generate(self, requests):
        torch = self.torch
        started = time.perf_counter()
        self.begin(requests)
        if self.role == 1:
            draws = torch.zeros(
                self.uniforms.shape, dtype=torch.float64, pin_memory=True
            )
            for r in requests:
                draws[self.slots[r.request_id], : r.max_tokens] = torch.tensor(
                    [
                        uniform(self.async_config.seed, r.request_id, p)
                        for p in range(r.max_tokens)
                    ],
                    dtype=torch.float64,
                )
            self.uniforms.copy_(draws, non_blocking=True)
            self._draws_host = draws
        self.prefill(None)
        prefill_seconds = time.perf_counter() - started
        done, sources, finished = set(), {r.request_id: [] for r in requests}, {}
        events = []
        steps, positions, after_eos = 0, 0, 0
        while len(done) != len(requests):
            starts = {
                rid: len(self.histories[rid])
                for rid in self.requests
                if rid not in done
            }
            for _ in range(self.async_config.readback_interval):
                if len(events) == 2:
                    events.pop(0).synchronize()
                active = [
                    rid
                    for rid in starts
                    if len(self.histories[rid]) < self.requests[rid].max_tokens
                ]
                if not active:
                    break
                contexts = {
                    rid: (
                        self.requests[rid].teacher_prompt
                        if self.role == 0
                        else self.requests[rid].student_prompt
                    )
                    + self.histories[rid]
                    for rid in active
                }
                rows = self.forward(contexts)
                b = self.lockstep_head.prepare(
                    torch.stack([rows[rid][-1] for rid in active]),
                    [self.runner.req_states.req_id_to_index[rid] for rid in active],
                    [self.slots[rid] for rid in active],
                    [len(self.histories[rid]) for rid in active],
                )
                stream = torch.cuda.current_stream()
                self.lockstep_head.project(b)
                if self.role == 0:
                    self.comm.send(
                        b["logits"]
                        if self.async_config.transfer_logits
                        else b["hidden"],
                        1,
                        stream,
                    )
                    self.comm.recv(b["tokens"], 1, stream)
                    self.lockstep_head.run(b)
                else:
                    self.comm.recv(b["teacher"], 0, stream)
                    self.lockstep_head.run(b)
                    self.comm.send(b["tokens"], 0, stream)
                for rid in active:
                    self.histories[rid].append(
                        0
                    )  # Native decode reads the definitive GPU token.
                steps += 1
                positions += len(active)
                event = torch.cuda.Event()
                event.record()
                events.append(event)
            spans = [
                (rid, start, len(self.histories[rid])) for rid, start in starts.items()
            ]
            tokens = torch.cat(
                [
                    self.output_tokens[self.slots[rid], start:end]
                    for rid, start, end in spans
                ]
            ).tolist()
            routes = (
                torch.cat(
                    [
                        self.output_sources[self.slots[rid], start:end]
                        for rid, start, end in spans
                    ]
                ).tolist()
                if self.role == 1
                else [False] * len(tokens)
            )
            offset = 0
            for rid, start, end in spans:
                values = tokens[offset : offset + end - start]
                selected = routes[offset : offset + end - start]
                offset += end - start
                for i, token in enumerate(values):
                    if token in self.requests[rid].stop_tokens:
                        after_eos += len(values) - i - 1
                        values, selected = values[: i + 1], selected[: i + 1]
                        done.add(rid)
                        break
                self.histories[rid][start:] = values
                sources[rid].extend("teacher" if s else "student" for s in selected)
                if len(self.histories[rid]) == self.requests[rid].max_tokens:
                    done.add(rid)
                if rid in done:
                    finished[rid] = time.perf_counter() - started
        return {
            "rows": [
                {"request_id": rid, "token_ids": tokens, "sources": sources[rid]}
                for rid, tokens in self.histories.items()
            ],
            "prefill_seconds": prefill_seconds,
            "forward_steps": steps,
            "model_positions": positions,
            "positions_after_eos": after_eos,
            "finished": finished,
        }

    def targets(self, _):
        if self.target_buffer is None:
            raise ValueError("target capture is disabled")
        return {
            rid: self.target_buffer[self.slots[rid], : len(history)].cpu()
            for rid, history in self.histories.items()
        }

    def targets_gpu(self, lease):
        from paired_vllm.cuda_cache import CudaTargetSlab

        if self.role != 0 or self.target_buffer is None:
            raise ValueError("teacher hidden capture is required")
        # Separate owned allocation: the capture buffer is reused for the next
        # admission group. This allocation is outside vLLM's sleep memory pool.
        packed = self.torch.cat(
            [
                self.target_buffer[self.slots[rid], : len(history)]
                for rid, history in self.histories.items()
            ]
        )
        slab = CudaTargetSlab(packed)
        self.target_exports[lease] = slab
        offset, result = 0, {}
        for rid, history in self.histories.items():
            result[rid] = slab.descriptor(
                start=offset,
                version=self.snapshot_version,
                token_ids=history,
                lease=lease,
            )
            offset += len(history)
        return result

    def release_targets(self, leases):
        for lease in leases:
            if lease not in self.target_exports:
                raise ValueError("unknown target lease")
            self.target_exports[lease].close()
            del self.target_exports[lease]
        return True

    def sleep(self, _):
        if self.role == 0 and self.async_config.runtime.keep_teacher_resident:
            return True
        self.worker.sleep(level=2)
        self.sleeping = True
        return True

    def shutdown(self):
        from paired_vllm.cuda_weights import close_cuda_weight_cache

        try:
            close_cuda_weight_cache(self.worker)
            self.release_targets(list(getattr(self, "target_exports", {})))
        finally:
            super().shutdown()

    def wake(self, _):
        if self.sleeping:
            self.worker.wake_up()
            self.sleeping = False
        return True

    def sync(self, payload):
        from paired_vllm.training_ops import load_vllm_weights, read_weights

        timings = {}

        if self.target_exports:
            raise RuntimeError("release target leases before refreshing weights")
        sync_teacher = payload.get("sync_teacher", True)
        if not sync_teacher and not (
            self.async_config.runtime.keep_teacher_resident and self.teacher_synced
        ):
            raise RuntimeError(
                "skipping teacher refresh requires a resident initialized teacher"
            )
        if self.role == 1 or sync_teacher:
            # The student maps the trainer's resident weights on its own GPU;
            # an EMA teacher pulls the trainer's reference model across GPUs
            # with peer copies. Neither path stages weight bytes on the host.
            side = "student" if self.role == 1 else "teacher"
            exported = payload.get(f"{side}_cuda")
            if exported is not None:
                from paired_vllm.cuda_weights import load_cuda_weights

                timings = load_cuda_weights(
                    self.worker,
                    exported,
                    cache=payload.get(f"cache_{side}_cuda_handles", False),
                )
            else:
                from paired_vllm.cuda_weights import close_cuda_weight_cache

                close_cuda_weight_cache(self.worker)
                load_vllm_weights(self.worker, payload[side])
            if self.role == 0:
                timings = {f"teacher_{k}": v for k, v in timings.items()}
        if self.role == 1 and sync_teacher and self.teacher_projection is not None:
            if payload.get("teacher_cuda") is not None:
                from paired_vllm.cuda_weights import copy_exported_tensor

                copy_exported_tensor(
                    payload["teacher_cuda"], "lm_head.weight", self.teacher_projection
                )
            else:
                self.teacher_projection.copy_(
                    read_weights(payload["teacher"])["lm_head.weight"]
                )
        self.teacher_synced = True
        self.snapshot_version = str(payload["version"])
        self.async_config = replace(self.async_config, seed=payload["seed"])
        return timings or True

    def sampling(self, payload):
        from paired_vllm.lockstep_head import LockstepHead

        updated = replace(
            self.settings,
            policy=payload["policy"],
            temperature=payload["temperature"],
            top_k=payload["top_k"],
        )
        changed = updated != self.settings
        self.settings = updated
        self.async_config = replace(
            self.async_config,
            runtime=replace(
                self.async_config.runtime,
                policy=payload["policy"],
                temperature=payload["temperature"],
                top_k=payload["top_k"],
            ),
        )
        if changed:
            # Sampling policy is static inside head graphs. Rebuild only at a
            # drained phase boundary (e.g. greedy evaluation), never per token.
            self.torch.cuda.synchronize()
            self.lockstep_head = LockstepHead(self)
        return True


def process_main(conn, config, side, gpu, rendezvous, port, control=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["VLLM_PLUGINS"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if config.runtime.dtype == "float32":
        os.environ["TRITON_F32_DEFAULT"] = "ieee"
    try:
        worker = LockstepWorker(config, side, rendezvous, port)
        worker.control = control
        conn.send(("BOOT", worker.metadata))
        while True:
            command, payload = conn.recv()
            if command == "STOP":
                worker.shutdown()
                break
            with worker.torch.inference_mode():
                result = getattr(worker, command.lower())(payload)
            conn.send((command, result))
    except EOFError:
        pass
    except BaseException:  # noqa: BLE001 -- propagate child failures
        conn.send(("ERROR", traceback.format_exc()))
    finally:
        conn.close()


class LockstepPairedEngine(AsyncPairedEngine):
    def __init__(
        self,
        runtime,
        readback_interval=8,
        seed=42,
        *,
        transfer_logits=False,
        fused_jsd=False,
        resident_targets=False,
    ):
        config = LockstepConfig(
            runtime,
            chunk_size=1,
            max_ahead_chunks=2,
            seed=seed,
            fast_draft=False,
            fast_verify=False,
            readback_interval=readback_interval,
            transfer_logits=transfer_logits,
            fused_jsd=fused_jsd,
            resident_targets=resident_targets,
        )
        super().__init__(config, worker_entrypoint=process_main)
        self.target_leases = set()
        self.next_lease = 0
        self.weights_ready = not runtime.enable_sleep_mode
        self.sleeping = False

    def generate(self, requests):
        if not self.weights_ready:
            raise RuntimeError("refresh weights after wake before generation")
        if self.target_leases:
            raise RuntimeError("release prior target leases before another rollout")
        return super().generate(requests)

    def evaluate_student(self, payload):
        if self.closed or self.sleeping or not self.weights_ready:
            raise RuntimeError("wake and synchronize student weights before evaluation")
        if self.target_leases:
            raise RuntimeError("release training targets before evaluation")
        try:
            self._send(1, "EVALUATE", payload)
            return self._drain(self.config.runtime.timeout)[1]
        except BaseException:
            self.close()
            raise

    def _collect_targets(self):
        if not self.config.resident_targets:
            return super()._collect_targets()
        lease = f"targets-{self.next_lease}"
        self.next_lease += 1
        self._send(0, "TARGETS_GPU", lease)
        result = self._drain()[0]
        self.target_leases.add(lease)
        return result

    def release_targets(self, leases):
        leases = set(leases)
        if not leases <= self.target_leases:
            raise ValueError("unknown target lease")
        if leases:
            self._send(0, "RELEASE_TARGETS", list(leases))
            self._drain()
            self.target_leases.difference_update(leases)
            self._last_targets = {
                rid: value
                for rid, value in self._last_targets.items()
                if value["lease"] not in leases
            }

    def training_command(self, command, payload=None):
        if self.closed:
            raise RuntimeError("engine is closed")
        if command not in ("SYNC", "SLEEP", "WAKE", "SAMPLING"):
            raise ValueError("unknown lockstep training command")
        if command == "SYNC" and self.target_leases:
            raise RuntimeError("release target leases before refreshing weights")
        if command == "SYNC" and self.sleeping:
            raise RuntimeError("wake the student before refreshing weights")
        if command in ("SLEEP", "WAKE") and not self.config.runtime.enable_sleep_mode:
            raise ValueError("sleep mode was not enabled")
        try:
            sides = (
                [1]
                if command in ("SLEEP", "WAKE")
                and self.config.runtime.keep_teacher_resident
                else self.connections
            )
            for side in sides:
                self._send(side, command, payload)
            result = self._drain(self.config.runtime.startup_timeout)
            if command == "SLEEP":
                self.weights_ready = False
                self.sleeping = True
            elif command == "WAKE":
                self.sleeping = False
            elif command == "SYNC":
                self.weights_ready = True
            elif command == "SAMPLING":
                self.config = replace(
                    self.config,
                    runtime=replace(
                        self.config.runtime,
                        policy=payload["policy"],
                        temperature=payload["temperature"],
                        top_k=payload["top_k"],
                        capture_targets=payload["capture"],
                    ),
                )
            return result
        except BaseException:
            self.close()
            raise

    def _generate(self, requests):
        started = time.perf_counter()
        for side in self.connections:
            self._send(side, "GENERATE", requests)
        results = self._drain()
        teacher, student = results[0], results[1]
        if [(r["request_id"], r["token_ids"]) for r in teacher["rows"]] != [
            (r["request_id"], r["token_ids"]) for r in student["rows"]
        ]:
            raise RuntimeError("teacher and student definitive-token histories differ")
        count = sum(len(r["token_ids"]) for r in student["rows"])
        duration = time.perf_counter() - started
        self.last_metrics = {
            "seconds": duration,
            "tokens_per_second": count / duration,
            "verified_tokens": count,
            "max_unverified_tokens": 0,
            "corrections": 0,
            "discarded_tokens": 0,
            "student_forward_steps": student["forward_steps"],
            "student_available_rows": student["forward_steps"] * len(requests),
            "model_positions": student["model_positions"],
            "positions_after_eos": student["positions_after_eos"],
            "prefill_seconds": max(
                teacher["prefill_seconds"], student["prefill_seconds"]
            ),
            "student_prefill_seconds": student["prefill_seconds"],
            "teacher_prefill_seconds": teacher["prefill_seconds"],
            "teacher_tokens": sum(
                r["sources"].count("teacher") for r in student["rows"]
            ),
            "completion_tail_seconds": max(student["finished"].values())
            - min(student["finished"].values()),
        }
        self.last_request_metrics.update(
            {
                rid: {"finished_seconds": t, "corrections": 0}
                for rid, t in student["finished"].items()
            }
        )
        return student["rows"]
