"""MRV2 adapter for independent drafting and teacher-forced block verification.

No installed vLLM modules are patched. Re-adding a request rebuilds native
metadata while retaining its private KV pages and the valid computed prefix.
"""

from __future__ import annotations

import os
import time
import traceback
from dataclasses import replace

from paired_vllm.async_core import Draft, uniform
from paired_vllm.worker import WorkerAdapter, load_teacher_projection


def sample(torch, logits, keys, seed, temperature, top_k):
    if temperature == 0:
        return logits.argmax(-1)
    values, indices = (
        logits.topk(min(top_k, logits.shape[-1]), dim=-1) if top_k else (logits, None)
    )
    cumulative = (values.float() / temperature).softmax(-1).cumsum(-1)
    draws = torch.tensor(
        [uniform(seed, rid, position) for rid, position in keys],
        device=logits.device,
        dtype=torch.float64,
    )
    selected = (cumulative.double() < draws[:, None]).sum(-1)
    selected.clamp_(max=values.shape[-1] - 1)
    return selected if indices is None else indices.gather(1, selected[:, None])[:, 0]


class AsyncWorker(WorkerAdapter):
    def __init__(self, config, side, rendezvous, port):
        runtime = config.runtime
        model = runtime.teacher_model if side == 0 else runtime.student_model
        # Reuse model/KV initialization without the synchronous pair's head,
        # target cache or communicator. Each process sees its own local cuda:0.
        local = replace(
            runtime,
            teacher_model=model,
            student_model=model,
            teacher_only=True,
            capture_targets=False,
        )
        backend = (
            "TRITON_ATTN"
            if runtime.dtype == "float32"
            else config.verify_backend
            if side == 0
            else "FLASH_ATTN"
        )
        compilation = None
        if side == 0 and config.verify_body_graphs and runtime.graphs:
            maximum = runtime.scheduler.max_requests * config.chunk_size
            compilation = {
                **runtime.compilation_config,
                "cudagraph_mode": "FULL",
                "cudagraph_capture_sizes": sorted(
                    {maximum, *(1 << i for i in range(maximum.bit_length()))}
                ),
            }
        super().__init__(
            local,
            0,
            rendezvous,
            None,
            attention_backend=backend,
            compilation_config=compilation,
        )
        self.async_config, self.role = config, side
        self.metadata["async_attention_backend"] = backend
        torch = self.torch
        if runtime.dtype == "float32":
            torch.set_float32_matmul_precision("highest")
        from transformers import AutoConfig

        student_config = AutoConfig.from_pretrained(runtime.student_model)
        self.width = student_config.hidden_size
        self.student_projection = None
        if side == 0:
            self.student_projection, _ = load_teacher_projection(
                runtime.student_model, self.config.model_config.dtype, student_config
            )
            self.available_memory -= (
                self.student_projection.numel() * self.student_projection.element_size()
            )
        self.features = (
            torch.empty(
                (runtime.scheduler.max_requests, runtime.max_model_len, self.width),
                dtype=self.config.model_config.dtype,
                device="cuda",
            )
            if side == 1
            else None
        )
        if self.features is not None:
            self.available_memory -= (
                self.features.numel() * self.features.element_size()
            )
        # Reserve head/verification temporary space in addition to native profiling.
        self.available_memory -= (
            runtime.scheduler.max_requests * config.chunk_size * self.vocab * 64
        )
        page_bytes = self.metadata["page_bytes"]
        self.metadata["feasible_slots"] = (
            max(0, self.available_memory // page_bytes - 1)
            * runtime.scheduler.block_size
        )
        self.metadata["async_student_projection_bytes"] = (
            self.student_projection.numel() * self.student_projection.element_size()
            if self.student_projection is not None
            else 0
        )
        self.metadata["async_hidden_buffer_bytes"] = (
            self.features.numel() * self.features.element_size()
            if self.features is not None
            else 0
        )
        self.metadata["async_feature_bytes_per_token"] = (
            self.width
            * torch.empty((), dtype=self.config.model_config.dtype).element_size()
        )
        if port is not None:
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
            from vllm.distributed.utils import StatelessProcessGroup

            group = StatelessProcessGroup.create(
                "127.0.0.1", port, side, 2, store_timeout=int(runtime.timeout)
            )
            self.comm = PyNcclCommunicator(group, device=0)
            group.barrier(timeout=runtime.timeout)
            if not self.comm.available or self.comm.disabled:
                raise RuntimeError("asynchronous hidden-state communicator unavailable")
        self.requests, self.histories, self.computed = {}, {}, {}
        self.blocks, self.versions, self.slots = {}, {}, {}
        self.pending_sends = []
        self.reset_ids = set()
        self.teacher_targets = {}
        self.forward_steps = 0
        self.metadata_updates = 0
        self.forward_events = []

    def init(self, capacity):
        metadata = super().init(capacity)
        manager = self.runner.cudagraph_manager
        metadata["async_body_graphs"] = (
            [str(descriptor) for descriptor in manager.graphs] if manager else []
        )
        if self.async_config.fast_draft or self.async_config.fast_verify:
            from paired_vllm.async_head import DraftHead, VerifyHead

            shape = (self.settings.scheduler.max_requests, self.settings.max_model_len)
            self.uniforms = self.torch.zeros(
                shape, device="cuda", dtype=self.torch.float64
            )
            if self.role == 1 and self.async_config.fast_draft:
                self.draft_tokens = self.torch.empty(
                    shape, device="cuda", dtype=self.torch.int64
                )
                self.draft_head = DraftHead(self)
            elif self.role == 0 and self.async_config.fast_verify:
                self.verify_head = VerifyHead(self)
            self.torch.cuda.current_stream().synchronize()
        return metadata

    def begin(self, requests):
        from vllm.v1.core.sched.output import SchedulerOutput

        # The prior batch is fully drained before BEGIN. A final EOS correction
        # may not have needed another draft call; never carry it into ID reuse.
        while self.control is not None and self.control.poll():
            self.control.recv()

        output = SchedulerOutput.make_empty()
        output.finished_req_ids = set(self.requests)
        self.worker.execute_model(output)
        self.requests = {r.request_id: r for r in requests}
        self.histories = {r.request_id: [] for r in requests}
        self.computed = {r.request_id: 0 for r in requests}
        self.versions = {r.request_id: 0 for r in requests}
        self.slots = {r.request_id: i for i, r in enumerate(requests)}
        self.blocks, self.teacher_targets = {}, {}
        block, next_page = self.settings.scheduler.block_size, 1
        for r in requests:
            prompt = r.teacher_prompt if self.role == 0 else r.student_prompt
            pages = (len(prompt) + r.max_tokens + block - 1) // block
            self.blocks[r.request_id] = list(range(next_page, next_page + pages))
            next_page += pages
        if (next_page - 1) * block > self.metadata["capacity"]:
            raise ValueError("rollout batch exceeds reserved KV capacity")
        self.reset_ids = set(self.requests)
        if self.async_config.fast_draft or self.async_config.fast_verify:
            # The table is immutable for an admission group. Rollback indexes
            # the same seed/request/position draws, regardless of scheduling.
            draws = self.torch.zeros(
                self.uniforms.shape, dtype=self.torch.float64, pin_memory=True
            )
            for r in requests:
                draws[self.slots[r.request_id], : r.max_tokens] = self.torch.tensor(
                    [
                        uniform(self.async_config.seed, r.request_id, p)
                        for p in range(r.max_tokens)
                    ],
                    dtype=self.torch.float64,
                )
            self.uniforms.copy_(draws, non_blocking=True)
            self._draws_host = draws
        return True

    def forward(self, contexts, reset=False):
        """Run just the uncached part of each context; return every queried state."""
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput

        if len(self.forward_events) == 2:
            self.forward_events.pop(0).synchronize()
        output = SchedulerOutput.make_empty()
        output.num_common_prefix_blocks = [0]
        states = self.runner.req_states
        reuse = self.async_config.reuse_metadata and all(
            rid in states.req_id_to_index for rid in contexts
        )
        staged = False
        self.metadata_updates += sum(reset or rid in self.reset_ids for rid in contexts)
        for rid, context in contexts.items():
            computed = self.computed[rid]
            count = len(context) - computed
            if count <= 0:
                raise ValueError("forward must advance at least one cache position")
            needs_reset = reset or rid in self.reset_ids
            if needs_reset and reuse:
                index = states.req_id_to_index[rid]
                states.prompt_len.np[index] = len(context)
                states.prefill_len.np[index] = len(context)
                states.max_seq_len[index] = self.settings.max_model_len
                states.total_len.stage_write_elem(index, len(context))
                states.num_computed_tokens.stage_write_elem(index, computed)
                states.all_token_ids.stage_write(index, computed, context[computed:])
                self.reset_ids.discard(rid)
                staged = True
            if needs_reset and not reuse:
                # All supplied tokens are teacher-forced inputs. Marking the
                # whole context as prefill makes MRV2 read them from all_token_ids.
                output.scheduled_new_reqs.append(
                    NewRequestData(
                        req_id=rid,
                        prompt_token_ids=context,
                        prefill_token_ids=context,
                        mm_features=[],
                        sampling_params=SamplingParams(temperature=0, max_tokens=1),
                        pooling_params=None,
                        block_ids=(self.blocks[rid],),
                        num_computed_tokens=computed,
                        lora_request=None,
                    )
                )
                self.reset_ids.discard(rid)
            else:
                cached = output.scheduled_cached_reqs
                cached.req_ids.append(rid)
                cached.new_block_ids.append(None)
                cached.num_computed_tokens.append(computed)
                cached.num_output_tokens.append(len(self.histories[rid]))
            output.num_scheduled_tokens[rid] = count
        if staged:
            # No requests are added in this path: exactly one staging rotation
            # per forward, so the two-in-flight UVA ownership bound still holds.
            states.apply_staged_writes()
        output.total_num_scheduled_tokens = sum(output.num_scheduled_tokens.values())
        if output.total_num_scheduled_tokens > self.settings.scheduler.token_budget:
            raise ValueError("forward exceeds configured token budget")
        self.worker.execute_model(output)
        self.forward_steps += 1
        state = self.runner.execute_model_state
        batch = state.input_batch
        # CPU query offsets are authoritative even if MRV2 reorders rows.
        offsets = batch.query_start_loc_np
        rows = {
            rid: state.hidden_states[int(offsets[i]) : int(offsets[i + 1])]
            for i, rid in enumerate(batch.req_ids)
        }
        self.runner.postprocess_num_computed_tokens(batch)
        self.runner.execute_model_state = None
        done = self.torch.cuda.Event()
        done.record()
        self.forward_events.append(done)
        for rid, context in contexts.items():
            self.computed[rid] = len(context)
        return rows

    def prefill(self, _):
        # Teacher retains all but the final prompt token. Each verification then
        # queries K positions: last prefix token plus K-1 drafted tokens.
        pending = {}
        for rid, r in self.requests.items():
            prompt = r.teacher_prompt if self.role == 0 else r.student_prompt
            pending[rid] = prompt[:-1]
        while any(self.computed[rid] < len(p) for rid, p in pending.items()):
            budget = self.settings.scheduler.token_budget
            contexts = {}
            for rid, prompt in pending.items():
                count = min(budget, len(prompt) - self.computed[rid])
                if count:
                    contexts[rid] = prompt[: self.computed[rid] + count]
                    budget -= count
            self.forward(contexts, reset=True)
        self.reset_ids.update(self.requests)
        self.torch.cuda.current_stream().synchronize()
        return True

    def apply_correction(self, rid, version, history):
        if version <= self.versions[rid]:
            return False
        self.versions[rid] = version
        self.histories[rid] = list(history)
        self.computed[rid] = len(self.requests[rid].student_prompt) + len(history) - 1
        self.reset_ids.add(rid)
        return True

    def poll_corrections(self, starts):
        discarded = 0
        while self.control is not None and self.control.poll():
            rid, version, history = self.control.recv()
            lost = (
                max(0, len(self.histories[rid]) - starts[rid]) if rid in starts else 0
            )
            if self.apply_correction(rid, version, history) and rid in starts:
                discarded += lost
                starts[rid] = len(history)
        return discarded

    def draft(self, payload):
        before_steps, before_updates = self.forward_steps, self.metadata_updates
        if self.async_config.fast_draft:
            result = self.draft_device(payload)
        else:
            result = self.draft_reference(payload)
        result["forward_steps"] = self.forward_steps - before_steps
        result["available_rows"] = result["forward_steps"] * len(self.requests)
        result["metadata_updates"] = self.metadata_updates - before_updates
        return result

    def draft_reference(self, payload):
        torch = self.torch
        ids, count, corrections = payload
        for rid, (version, history) in corrections.items():
            self.apply_correction(rid, version, history)
        starts = {rid: len(self.histories[rid]) for rid in ids}
        generated, interrupted = 0, 0
        for _ in range(count):
            interrupted += self.poll_corrections(starts)
            active = [
                rid
                for rid in ids
                if len(self.histories[rid]) < self.requests[rid].max_tokens
                and len(self.histories[rid]) - starts[rid] < ids[rid]
                and (
                    not self.histories[rid]
                    or self.histories[rid][-1] not in self.requests[rid].stop_tokens
                )
            ]
            if not active:
                break
            contexts = {
                rid: self.requests[rid].student_prompt + self.histories[rid]
                for rid in active
            }
            rows = self.forward(contexts)
            hidden = torch.stack([rows[rid][-1] for rid in active])
            keys = [(rid, len(self.histories[rid])) for rid in active]
            for i, (rid, position) in enumerate(keys):
                self.features[self.slots[rid], position].copy_(hidden[i])
            logits = self.runner.model.compute_logits(hidden).float()
            tokens = sample(
                torch,
                logits,
                keys,
                self.async_config.seed,
                self.settings.temperature,
                self.settings.top_k,
            )
            indices = torch.tensor(
                [self.runner.req_states.req_id_to_index[rid] for rid in active],
                device="cuda",
                dtype=torch.int32,
            )
            ones = torch.ones(len(active), dtype=torch.int32, device="cuda")
            self.runner.postprocess_sampled(
                indices,
                tokens[:, None],
                ones,
                torch.zeros_like(ones),
                query_start_loc=None,
            )
            # Only local student completion is read back, never teacher progress.
            for rid, token in zip(active, tokens.tolist()):
                self.histories[rid].append(token)
                generated += 1
            self.pending_sends = [
                (event, value)
                for event, value in self.pending_sends
                if not event.query()
            ]
        drafts = [
            Draft(
                rid, self.versions[rid], starts[rid], self.histories[rid][starts[rid] :]
            )
            for rid in ids
            if len(self.histories[rid]) > starts[rid]
        ]
        return {
            "drafts": drafts,
            "generated_tokens": generated,
            "interrupted_tokens": interrupted,
        }

    def draft_device(self, payload):
        """Pipeline two native forwards; materialize tokens once per block.

        Unknown CPU history entries are only length placeholders. Native decode
        consumes last_sampled_tokens on device; any reset uses an authoritative
        correction history. EOS is trimmed at publication, before verification.
        """
        torch = self.torch
        ids, count, corrections = payload
        for rid, (version, history) in corrections.items():
            self.apply_correction(rid, version, history)
        starts = {rid: len(self.histories[rid]) for rid in ids}
        generated, interrupted, events = 0, 0, []
        for _ in range(count):
            # MRV2 and DraftHead each own two UVA/pinned staging slots. A third
            # in-flight use may not overwrite a slot still read by the GPU.
            if len(events) == 2:
                events.pop(0).synchronize()
            interrupted += self.poll_corrections(starts)
            active = [
                rid
                for rid in ids
                if len(self.histories[rid]) < self.requests[rid].max_tokens
                and len(self.histories[rid]) - starts[rid] < ids[rid]
                and not (
                    len(self.histories[rid]) == starts[rid]
                    and self.histories[rid]
                    and self.histories[rid][-1] in self.requests[rid].stop_tokens
                )
            ]
            if not active:
                break
            contexts = {
                rid: self.requests[rid].student_prompt + self.histories[rid]
                for rid in active
            }
            rows = self.forward(contexts)
            self.draft_head(
                torch.stack([rows[rid][-1] for rid in active]),
                [self.runner.req_states.req_id_to_index[rid] for rid in active],
                [self.slots[rid] for rid in active],
                [len(self.histories[rid]) for rid in active],
            )
            for rid in active:
                self.histories[rid].append(
                    0
                )  # Length only; never used as an input token.
            generated += len(active)
            event = torch.cuda.Event()
            event.record()
            events.append(event)
        selected = [rid for rid in ids if len(self.histories[rid]) > starts[rid]]
        if selected:
            values = torch.cat(
                [
                    self.draft_tokens[
                        self.slots[rid], starts[rid] : len(self.histories[rid])
                    ]
                    for rid in selected
                ]
            ).tolist()
            offset = 0
            for rid in selected:
                length = len(self.histories[rid]) - starts[rid]
                tokens = values[offset : offset + length]
                offset += length
                for i, token in enumerate(tokens):
                    if token in self.requests[rid].stop_tokens:
                        interrupted += len(tokens) - i - 1
                        tokens = tokens[: i + 1]
                        break
                self.histories[rid][starts[rid] :] = tokens
        elif events:
            events[-1].synchronize()
        self.pending_sends = [
            (event, value) for event, value in self.pending_sends if not event.query()
        ]
        return {
            "drafts": [
                Draft(
                    rid,
                    self.versions[rid],
                    starts[rid],
                    self.histories[rid][starts[rid] :],
                )
                for rid in selected
            ],
            "generated_tokens": generated,
            "interrupted_tokens": interrupted,
        }

    def export(self, drafts):
        torch = self.torch
        # Snapshot before subsequent rollback/drafting can overwrite positions.
        data = torch.cat(
            [
                self.features[
                    self.slots[d.request_id], d.start : d.start + len(d.tokens)
                ]
                for d in drafts
            ]
        )
        self.comm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.comm_stream):
            self.comm.send(data, 0, self.comm_stream)
            done = torch.cuda.Event()
            done.record()
        self.pending_sends.append((done, data))
        # Deliberately no compute-stream wait for the send, or device fence.
        return True

    def verify(self, drafts):
        torch = self.torch
        count = sum(len(d.tokens) for d in drafts)
        features = torch.empty(
            (count, self.width), device="cuda", dtype=self.config.model_config.dtype
        )
        with torch.cuda.stream(self.comm_stream):
            self.comm.recv(features, 1, self.comm_stream)
        contexts = {}
        for d in drafts:
            rid = d.request_id
            if d.start != len(self.histories[rid]):
                raise ValueError("teacher verification frontier mismatch")
            contexts[rid] = (
                self.requests[rid].teacher_prompt + self.histories[rid] + d.tokens[:-1]
            )
            self.computed[rid] = len(self.requests[rid].teacher_prompt) + d.start - 1
        started = time.perf_counter()
        rows = self.forward(contexts, reset=True)
        teacher_hidden = torch.cat([rows[d.request_id] for d in drafts])
        torch.cuda.current_stream().wait_stream(self.comm_stream)
        if self.async_config.fast_verify:
            values = self.verify_head(teacher_hidden, features, drafts)
        else:
            values = self.verify_reference(teacher_hidden, features, drafts)
        results, offset = [], 0
        for d in drafts:
            tokens, sources, jsds = [], [], []
            for i, (token, source, score) in enumerate(
                values[offset : offset + len(d.tokens)]
            ):
                token = int(token)
                tokens.append(token)
                sources.append("teacher" if source else "student")
                jsds.append(score)
                if (
                    token != d.tokens[i]
                    or token in self.requests[d.request_id].stop_tokens
                ):
                    break
            self.histories[d.request_id].extend(tokens)
            if self.async_config.runtime.capture_targets:
                self.teacher_targets.setdefault(d.request_id, []).append(
                    teacher_hidden[offset : offset + len(tokens)].clone()
                )
            results.append((d, tokens, sources, jsds))
            offset += len(d.tokens)
        return {
            "rows": results,
            "seconds": time.perf_counter() - started,
            "positions": count,
        }

    def verify_reference(self, teacher_hidden, features, drafts):
        torch = self.torch
        teacher = self.runner.model.compute_logits(teacher_hidden).float()
        torch.cuda.current_stream().wait_stream(self.comm_stream)
        student = torch.nn.functional.linear(features, self.student_projection).float()
        t, s = teacher.log_softmax(-1), student.log_softmax(-1)
        mixture = torch.logaddexp(t, s) - 0.6931471805599453
        scores = (
            (t.exp() * (t - mixture)).sum(-1) + (s.exp() * (s - mixture)).sum(-1)
        ) / 2
        route = scores > self.async_config.threshold
        keys = [
            (d.request_id, d.start + i) for d in drafts for i in range(len(d.tokens))
        ]
        candidates = sample(
            torch,
            teacher,
            keys,
            self.async_config.seed,
            self.settings.temperature,
            self.settings.top_k,
        )
        original = torch.tensor(
            [token for d in drafts for token in d.tokens], device="cuda"
        )
        chosen = torch.where(route, candidates, original)
        # One readback per verification block; the student is in a different process.
        values = torch.stack(
            (chosen.double(), route.double(), scores.double()), -1
        ).tolist()
        return values

    def targets(self, _):
        return {
            rid: self.torch.cat(rows).cpu()
            for rid, rows in self.teacher_targets.items()
        }


def process_main(conn, config, side, gpu, rendezvous, port, control=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["VLLM_PLUGINS"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if config.runtime.dtype == "float32":
        # Precision-validation mode: Triton's default FP32 dot uses TF32.
        os.environ["TRITON_F32_DEFAULT"] = "ieee"
    try:
        worker = AsyncWorker(config, side, rendezvous, port)
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
    except BaseException:  # noqa: BLE001 -- propagate child failure to coordinator
        conn.send(("ERROR", traceback.format_exc()))
    finally:
        conn.close()
