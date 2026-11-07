"""Narrow adapter for vLLM 0.31.0 GPUWorker + Model Runner V2.

Imported in a spawned child. CUDA visibility is set before importing torch or
vLLM. Stock attention kernels, input preparation and post-update are reused;
Paired training supplies its own scheduling and sampling; student-only
evaluation can reuse the native scheduler and sampler on these allocations.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from pathlib import Path


def load_teacher_projection(model_path, dtype, config=None):
    """Load only the exact unquantized Qwen output weight, never a second body."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig

    config = config or AutoConfig.from_pretrained(model_path)
    if config.model_type not in ("qwen3", "qwen2") or getattr(
        config, "quantization_config", None
    ):
        raise ValueError(
            "hidden-state projection adapter requires unquantized Qwen2/Qwen3"
        )
    path = Path(model_path)
    if not path.is_dir():
        path = Path(
            snapshot_download(
                model_path,
                revision=getattr(config, "_commit_hash", None),
                allow_patterns=["*.json", "*.safetensors"],
            )
        )
    key = (
        "model.embed_tokens.weight" if config.tie_word_embeddings else "lm_head.weight"
    )
    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as weights:
            if key in weights.keys():  # noqa: SIM118 -- safe_open is not a dict
                weight = weights.get_tensor(key).to(device="cuda", dtype=dtype)
                if weight.shape != (config.vocab_size, config.hidden_size):
                    raise ValueError("teacher projection shape mismatch")
                return weight, config
    raise ValueError(f"missing exact teacher output projection {key}")


def process_main(conn, settings, side, gpu, rendezvous, pair_port):
    if os.getenv("PAIRED_WORKER_DEBUG"):
        import faulthandler

        faulthandler.dump_traceback_later(45, repeat=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["VLLM_PLUGINS"] = ""  # Do not load the old ensemble plugin.
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache/huggingface"))
    os.environ.setdefault("VLLM_CACHE_ROOT", str(Path.home() / ".cache/vllm"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(Path.home() / ".cache/triton"))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        adapter = WorkerAdapter(settings, side, rendezvous, pair_port)
        conn.send(("BOOT", adapter.metadata))
        while True:
            command, payload = conn.recv()
            if os.getenv("PAIRED_WORKER_DEBUG"):
                print(f"worker {side}: {command}", flush=True)
            if command == "STOP":
                adapter.shutdown()
                break
            result = getattr(adapter, command.lower())(payload)
            conn.send((command, result))
    except EOFError:
        pass
    except BaseException:  # noqa: BLE001 -- report child failures to the owner
        conn.send(("ERROR", traceback.format_exc()))
    finally:
        conn.close()


class WorkerAdapter:
    def __init__(
        self,
        settings,
        side,
        rendezvous,
        pair_port,
        attention_backend="FLASH_ATTN",
        compilation_config=None,
    ):
        import torch
        import vllm
        from transformers import AutoTokenizer
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.worker.gpu_worker import Worker

        from paired_vllm import VLLM_VERSION

        if vllm.__version__ != VLLM_VERSION:
            raise RuntimeError(f"requires vLLM {VLLM_VERSION}, got {vllm.__version__}")
        torch.set_num_threads(1)
        self.torch, self.settings, self.side = torch, settings, side
        model = settings.teacher_model if side == 0 else settings.student_model
        started = time.perf_counter()
        self.config = EngineArgs(
            model=model,
            dtype=settings.dtype,
            seed=42,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            max_model_len=settings.max_model_len,
            max_num_seqs=settings.scheduler.max_requests,
            max_num_batched_tokens=settings.scheduler.token_budget,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            async_scheduling=False,
            enforce_eager=not settings.graphs,
            compilation_config=(
                settings.compilation_config
                if compilation_config is None
                else compilation_config
            ),
            kernel_config={"enable_jit_warmup": False},
            block_size=settings.scheduler.block_size,
            gpu_memory_utilization=settings.memory_utilization,
            attention_backend=attention_backend,
            disable_log_stats=True,
            enable_sleep_mode=settings.enable_sleep_mode,
        ).create_engine_config()
        from vllm.config import set_current_vllm_config

        self.config_context = set_current_vllm_config(self.config)
        self.config_context.__enter__()
        self.worker = Worker(
            self.config,
            local_rank=0,
            rank=0,
            distributed_init_method=rendezvous,
            is_driver_worker=True,
        )
        self.worker.init_device()
        self.worker.load_model()
        self.runner = self.worker.model_runner
        if self.runner.__class__.__module__ != "vllm.v1.worker.gpu.model_runner":
            raise RuntimeError("Model Runner V2 was not selected")
        self.specs = self.worker.get_kv_cache_spec()
        from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

        layout = resolve_kv_cache_layout(
            self.config,
            [self.worker.get_supported_kv_cache_layouts()],
            self.specs.values(),
        )
        self.worker.set_kv_cache_layout(layout.name)
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        if not self.specs or any(
            type(s) is not FullAttentionSpec for s in self.specs.values()
        ):
            raise ValueError("only ordinary uniform full-attention KV is supported")
        if any(
            s.block_size != settings.scheduler.block_size for s in self.specs.values()
        ):
            raise ValueError(
                "resolved allocation block size differs from configured bound"
            )
        self.vocab = self.runner.vocab_size
        if self.config.model_config.hf_config.model_type not in ("qwen3", "qwen2"):
            raise ValueError(
                "final post-norm hidden-state contract currently supports Qwen2/Qwen3"
            )
        n = settings.scheduler.max_requests
        # Allocate retained decision and transport storage BEFORE profiling KV.
        self.logits = torch.empty((n, self.vocab), dtype=torch.float32, device="cuda")
        from transformers import AutoConfig

        teacher_config = AutoConfig.from_pretrained(settings.teacher_model)
        self.teacher_width = teacher_config.hidden_size
        self.payload = torch.empty(
            (n, self.teacher_width), dtype=self.config.model_config.dtype, device="cuda"
        )
        self.hidden = torch.empty_like(self.payload) if side == 0 else None
        self.teacher_projection = None
        self.reconstructed_logits = None
        if side == 1:
            self.teacher_projection, _ = load_teacher_projection(
                settings.teacher_model, self.config.model_config.dtype, teacher_config
            )
        self.route = torch.empty(n, dtype=torch.int32, device="cuda")
        self.candidates = torch.empty(n, dtype=torch.int64, device="cuda")
        self.choice_tokens_cpu = torch.empty(n, dtype=torch.int64, pin_memory=True)
        self.choice_sources_cpu = torch.empty(n, dtype=torch.int32, pin_memory=True)
        # Retire fences each transaction, so these pinned buffers can be reused
        # without allocating and synchronously uploading four tensors per token.
        self.commit_indices_cpu = torch.empty(n, dtype=torch.int32, pin_memory=True)
        self.commit_tokens_cpu = torch.empty((n, 1), dtype=torch.int64, pin_memory=True)
        self.commit_indices = torch.empty(n, dtype=torch.int32, device="cuda")
        self.commit_tokens = torch.empty((n, 1), dtype=torch.int64, device="cuda")
        self.commit_ones = torch.ones(n, dtype=torch.int32, device="cuda")
        self.commit_zeros = torch.zeros(n, dtype=torch.int32, device="cuda")
        self.head = None
        if side == 1:
            from paired_vllm.head import DecisionHead

            self.head = DecisionHead(
                torch,
                self.teacher_projection,
                self.payload,
                self.route,
                self.candidates,
                settings.policy,
                settings.temperature,
                settings.top_k,
                enable_graphs=settings.graphs and settings.head_graphs,
            )
        self.target_cache = None
        self.capture_active = settings.capture_targets
        self.sleeping = False
        self.teacher_synced = False
        if settings.capture_targets and side == 1:
            from paired_vllm.training_ops import BufferedHiddenCache, TargetCache

            self.target_cache = (
                BufferedHiddenCache() if settings.gpu_target_cache else TargetCache()
            )
        self.comm_stream = torch.cuda.Stream()
        self.comm = None
        if pair_port is not None:
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
            from vllm.distributed.utils import StatelessProcessGroup

            group = StatelessProcessGroup.create(
                "127.0.0.1", pair_port, side, 2, store_timeout=int(settings.timeout)
            )
            self.comm = PyNcclCommunicator(group, device=0)
            group.barrier(timeout=settings.timeout)
            if not self.comm.available or self.comm.disabled:
                raise RuntimeError("dedicated paired NCCL communicator unavailable")
        # Profile the real dependent head at the largest supported batch inside
        # GPUWorker's memory-profiling window. Dummy execution uses local buffers
        # and never waits for a peer. Keep this hook local to this runner instance.
        if self.head is not None:
            original_profile_run = self.runner.profile_run

            def profile_with_head(*args, **kwargs):
                result = original_profile_run(*args, **kwargs)
                with torch.inference_mode():
                    self.payload.zero_()
                    self.head(self.logits.zero_(), list(range(n)))
                    torch.cuda.synchronize()
                return result

            self.runner.profile_run = profile_with_head
        self.available_memory = self.worker.determine_available_memory()
        page_bytes = sum(s.page_size_bytes for s in self.specs.values())
        tokenizer = AutoTokenizer.from_pretrained(model)
        token_contract = {
            "vocab": tokenizer.get_vocab(),
            "special": tokenizer.special_tokens_map,
        }
        self.metadata = {
            "vllm": vllm.__version__,
            "runner": self.runner.__class__.__module__,
            "model": model,
            "checkpoint_revision": getattr(
                self.config.model_config.hf_config, "_commit_hash", None
            )
            or model,
            "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
            "tokenizer_digest": hashlib.sha256(
                json.dumps(token_contract, sort_keys=True).encode()
            ).hexdigest(),
            "vocab": self.vocab,
            "teacher_projection_revision": getattr(teacher_config, "_commit_hash", None)
            or settings.teacher_model,
            "page_bytes": page_bytes,
            "feature_contract": {
                "layer": "final",
                "normalization": "post_norm",
                "dtype": str(self.payload.dtype),
                "width": self.teacher_width,
            },
            "teacher_projection_bytes": 0
            if self.teacher_projection is None
            else self.teacher_projection.numel()
            * self.teacher_projection.element_size(),
            "feasible_slots": max(0, self.available_memory // page_bytes - 1)
            * settings.scheduler.block_size,
            "load_profile_seconds": time.perf_counter() - started,
        }
        self.histories, self.computed, self.blocks, self.prompts = {}, {}, {}, {}
        self.block_digests = {}
        self.history_digests = {}
        self.decision_ready = set()
        self.manifest = None
        self.phase = "IDLE"
        self.last_retired = None

    def init(self, capacity):
        from vllm.v1.core.kv_cache_utils import get_kv_cache_configs

        started = time.perf_counter()
        self.config.cache_config.num_gpu_blocks_override = (
            capacity // self.settings.scheduler.block_size + 1
        )
        cache = get_kv_cache_configs(
            self.config, [self.specs], [self.available_memory]
        )[0]
        cache.kv_cache_layout = self.config.cache_config.kv_cache_layout
        self.kv_cache_config = cache
        if len(cache.kv_cache_groups) != 1 or cache.needs_kv_cache_zeroing:
            raise ValueError("only one uniform full-attention cache group is supported")
        self.worker.initialize_from_config(cache)
        self.torch.cuda.synchronize()
        allocated = time.perf_counter()
        self.worker.compile_or_warm_up_model()
        self.torch.cuda.synchronize()
        self.metadata.update(
            capacity=capacity,
            kv_bytes=sum(t.size for t in cache.kv_cache_tensors),
            kv_initialization_seconds=time.perf_counter() - started,
            kv_allocation_seconds=allocated - started,
            warmup_seconds=time.perf_counter() - allocated,
        )
        return self.metadata

    def prepare(self, manifest):
        from paired_vllm.core import (
            EMPTY_HISTORY_DIGEST,
            ProtocolError,
            advance_history_digest,
        )

        if self.phase != "IDLE":
            raise ProtocolError("worker transaction has not retired")
        if (
            self.last_retired
            and (manifest.epoch, manifest.step_id) <= self.last_retired
        ):
            raise ProtocolError("stale manifest")
        rows = manifest.teacher if self.side == 0 else manifest.student
        if manifest.feature_shape != (len(manifest.ready), self.teacher_width):
            raise ProtocolError("feature shape differs from startup contract")
        if manifest.contract[2:] != (
            self.settings.policy_revision,
            "routing_only",
            str(self.payload.dtype),
        ):
            raise ProtocolError("policy or feature dtype differs from startup contract")
        if manifest.contract[self.side] != self.metadata["checkpoint_revision"]:
            raise ProtocolError("checkpoint revision mismatch")
        for row in rows:
            if row.key in self.histories:
                history = self.histories[row.key]
                prompt = self.prompts[row.key]
                if (
                    row.history
                    or self.history_digests[row.key] != row.history_digest
                    or self.computed[row.key] != row.computed
                ):
                    raise ProtocolError("worker history/computed length mismatch")
            else:
                history, prompt = row.history, row.prompt
                digest = EMPTY_HISTORY_DIGEST
                for token in history:
                    digest = advance_history_digest(digest, token)
                if digest != row.history_digest:
                    raise ProtocolError("replayed history digest mismatch")
            end = row.computed + len(row.tokens)
            expected = (
                tuple(prompt[row.computed : min(end, len(prompt))])
                if row.computed < len(prompt)
                else ()
            )
            expected += tuple(
                history[max(0, row.computed - len(prompt)) : max(0, end - len(prompt))]
            )
            if row.output_index != len(history) or row.tokens != expected:
                raise ProtocolError("manifest query does not match committed history")
            previous_blocks = self.blocks.get(row.key, ())
            if row.block_start != len(previous_blocks):
                raise ProtocolError("incremental block table offset mismatch")
            block_digest = self.block_digests.get(row.key, EMPTY_HISTORY_DIGEST)
            for page in row.blocks:
                block_digest = advance_history_digest(block_digest, page)
            if block_digest != row.block_digest:
                raise ProtocolError("incremental block table digest mismatch")
            if (
                row.block_start + len(row.blocks)
            ) * self.settings.scheduler.block_size < row.computed + len(row.tokens):
                raise ProtocolError("manifest has insufficient mapped pages")
            if any(not 0 <= token < self.vocab for token in row.tokens):
                raise ProtocolError("token outside shared vocabulary")
        self.manifest, self.phase = manifest, "PREPARED"
        return manifest.digest

    def _check(self, digest, phase):
        from paired_vllm.core import ProtocolError

        if self.phase != phase or self.manifest.digest != digest:
            raise ProtocolError(f"invalid worker message in {self.phase}")

    def execute(self, digest):
        self._check(digest, "PREPARED")
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput

        torch = self.torch
        manifest = self.manifest
        rows = manifest.teacher if self.side == 0 else manifest.student
        output = SchedulerOutput.make_empty()
        output.num_common_prefix_blocks = [0]
        output.finished_req_ids = set(manifest.finished)
        for key in manifest.finished:
            self.histories.pop(key, None)
            self.history_digests.pop(key, None)
            self.prompts.pop(key, None)
            self.computed.pop(key, None)
            self.blocks.pop(key, None)
            self.block_digests.pop(key, None)
            self.decision_ready.discard(key)
        for row in rows:
            if row.key not in self.histories:
                output.scheduled_new_reqs.append(
                    NewRequestData(
                        req_id=row.key,
                        prompt_token_ids=list(row.prompt),
                        prefill_token_ids=list(row.prompt + row.history),
                        mm_features=[],
                        sampling_params=SamplingParams(
                            temperature=0, max_tokens=row.max_tokens
                        ),
                        pooling_params=None,
                        block_ids=(list(row.blocks),),
                        num_computed_tokens=row.computed,
                        lora_request=None,
                    )
                )
                self.histories[row.key] = list(row.history)
                self.history_digests[row.key] = row.history_digest
                self.prompts[row.key] = row.prompt
            else:
                cached = output.scheduled_cached_reqs
                cached.req_ids.append(row.key)
                cached.new_block_ids.append((list(row.blocks),))
                cached.num_computed_tokens.append(row.computed)
                cached.num_output_tokens.append(row.output_index)
            self.blocks[row.key] = self.blocks.get(row.key, ()) + row.blocks
            self.block_digests[row.key] = row.block_digest
            output.num_scheduled_tokens[row.key] = len(row.tokens)
        output.total_num_scheduled_tokens = sum(output.num_scheduled_tokens.values())
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        with torch.inference_mode():
            self.worker.execute_model(output)
            if rows:
                state = self.runner.execute_model_state
                batch = state.input_batch
                # MRV2 may sort the rows. Gather by identity, never batch index.
                by_key = {row.key: row for row in rows}
                selected = [
                    i
                    for i, key in enumerate(batch.req_ids)
                    if by_key[key].computed + len(by_key[key].tokens)
                    == by_key[key].prompt_length + by_key[key].output_index
                ]
                if selected:
                    indices = torch.tensor(selected, device="cuda")
                    hidden = state.hidden_states[batch.logits_indices[indices]]
                    slots = [
                        self.runner.req_states.req_id_to_index[batch.req_ids[i]]
                        for i in selected
                    ]
                    if self.side == 0:
                        self.hidden[slots] = hidden
                    if self.side == 1 or self.comm is None:
                        self.logits[slots] = self.runner.model.compute_logits(
                            hidden
                        ).float()
                    self.decision_ready.update(batch.req_ids[i] for i in selected)
                # Advance computed lengths once; no local sampling or output update.
                self.runner.postprocess_num_computed_tokens(batch)
                self.runner.execute_model_state = None
                for row in rows:
                    self.computed[row.key] = row.computed + len(row.tokens)
        end.record()
        self.forward_start = start
        self.forward_done = end
        self.phase = "EXECUTED"
        # EXECUTE acknowledges queuing. DECIDE/COMMIT synchronize GPU completion.
        return {"digest": digest}

    def decide(self, digest):
        self._check(digest, "EXECUTED")
        torch = self.torch
        keys = [f"{rid}:{inc}" for rid, inc, _ in self.manifest.ready]
        if any(key not in self.decision_ready for key in keys):
            raise RuntimeError("routing before both final context states exist")
        n = len(keys)
        if not n:
            self._decision_fence()
            self.phase = "PROPOSED"
            return {
                "digest": digest,
                "choices": {},
                "forward_ms": self.forward_start.elapsed_time(self.forward_done),
                "feature_bytes": 0,
                "feature_transfer_ms": 0,
            }
        slots = [self.runner.req_states.req_id_to_index[k] for k in keys]
        own = self.logits[slots] if self.side == 1 or self.comm is None else None
        transfer_start, transfer_end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        with torch.inference_mode():
            if self.comm is None:
                self.route[:n].fill_(1)
                self.candidates[:n] = own.argmax(-1)
            else:
                # One fixed GPU transfer: final post-norm teacher states T -> S.
                # Both vocabulary heads and source-first sampling live on S.
                self.comm_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.comm_stream):
                    transfer_start.record()
                    if self.side == 0:
                        self.payload[:n].copy_(self.hidden[slots])
                        self.comm.send(self.payload[:n], 1, self.comm_stream)
                    else:
                        self.comm.recv(self.payload[:n], 0, self.comm_stream)
                    transfer_end.record()
                torch.cuda.current_stream().wait_stream(self.comm_stream)
                if self.side == 1:
                    self.reconstructed_logits = self.head(
                        own, [k for _, _, k in self.manifest.ready]
                    )
                    if self.target_cache is not None and self.capture_active:
                        if self.settings.target_format == "topk":
                            from paired_vllm.training_ops import capture_topk

                            self.capture_values = capture_topk(
                                own,
                                self.reconstructed_logits,
                                self.settings.target_topk,
                                self.settings.target_temperature,
                            )
                        else:
                            self.capture_values = (
                                self.payload[:n]
                                if self.settings.target_format == "hidden"
                                else self.reconstructed_logits
                            )
        if self.settings.fused_dispatch and (self.side == 1 or self.comm is None):
            # Queue both tiny readbacks before a single stream fence. Reading
            # the pinned buffers afterward avoids two synchronous .tolist()
            # device transfers on every token.
            self.choice_tokens_cpu[:n].copy_(self.candidates[:n], non_blocking=True)
            self.choice_sources_cpu[:n].copy_(self.route[:n], non_blocking=True)
        self._decision_fence()
        choices = {}
        if self.side == 1 or self.comm is None:
            tokens, sources = (
                (
                    self.choice_tokens_cpu[:n].tolist(),
                    self.choice_sources_cpu[:n].tolist(),
                )
                if self.settings.fused_dispatch
                else (self.candidates[:n].tolist(), self.route[:n].tolist())
            )
            choices = {
                rid: (token, "teacher" if source else "student")
                for (rid, _, _), token, source in zip(
                    self.manifest.ready, tokens, sources
                )
            }
        self.phase = "PROPOSED"
        return {
            "digest": digest,
            "choices": choices,
            "forward_ms": self.forward_start.elapsed_time(self.forward_done),
            "feature_transfer_ms": transfer_start.elapsed_time(transfer_end)
            if self.comm
            else 0.0,
            "feature_bytes": n * self.teacher_width * self.payload.element_size()
            if self.comm
            else 0,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        }

    def run(self, digest):
        self.execute(digest)
        return self.decide(digest)

    def _decision_fence(self):
        if self.settings.fused_dispatch:
            self.torch.cuda.current_stream().synchronize()
        else:
            self.torch.cuda.synchronize()

    def prepare_run(self, manifest):
        return self.run(self.prepare(manifest))

    def finalize(self, payload):
        if self.phase == "IDLE" and self.manifest.digest == payload[0]:
            if self.applied_events != payload[1]:
                raise RuntimeError("conflicting retired commit")
            return payload[0]
        self.commit(payload)
        return self.retire(payload[0])

    def shutdown(self):
        if self.sleeping:
            self.worker.wake_up()
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        self.torch.cuda.synchronize()
        if self.comm is not None:
            self.comm.destroy()
        cleanup_dist_env_and_memory()
        self.config_context.__exit__(None, None, None)

    def commit(self, payload):
        from paired_vllm.core import advance_history_digest

        digest, events = payload
        if self.phase == "COMMITTED" and self.manifest.digest == digest:
            if events != self.applied_events:
                raise RuntimeError("conflicting duplicate commit")
            return digest
        self._check(digest, "PROPOSED")
        torch = self.torch
        indices, tokens = [], []
        for event in events:
            if (
                event.request_id,
                event.incarnation,
                event.output_index,
            ) not in self.manifest.ready:
                raise RuntimeError("commit identity absent from ready manifest")
            if not 0 <= event.token_id < self.vocab:
                raise RuntimeError("commit token outside vocabulary")
            key = f"{event.request_id}:{event.incarnation}"
            if len(self.histories[key]) != event.output_index:
                raise RuntimeError("commit output index mismatch")
            indices.append(self.runner.req_states.req_id_to_index[key])
            tokens.append([event.token_id])
        if indices:
            with torch.inference_mode():
                n = len(indices)
                self.commit_indices_cpu.numpy()[:n] = indices
                self.commit_tokens_cpu.numpy()[:n] = tokens
                self.commit_indices[:n].copy_(
                    self.commit_indices_cpu[:n], non_blocking=True
                )
                self.commit_tokens[:n].copy_(
                    self.commit_tokens_cpu[:n], non_blocking=True
                )
                # query_start_loc=None => zero computed delta. Updates token history,
                # last_sampled_tokens, total_len and output penalty counts together.
                self.runner.postprocess_sampled(
                    self.commit_indices[:n],
                    self.commit_tokens[:n],
                    self.commit_ones[:n],
                    self.commit_zeros[:n],
                    query_start_loc=None,
                )
        cached_payload = None
        if self.target_cache is not None and self.capture_active and events:
            value = self.capture_values
            if self.settings.gpu_target_cache:
                positions = {key: i for i, key in enumerate(self.manifest.ready)}
                self.target_cache.append_batch(
                    [
                        (
                            event.request_id,
                            event.incarnation,
                            event.output_index,
                            self.manifest.contract[0],
                            positions[
                                (
                                    event.request_id,
                                    event.incarnation,
                                    event.output_index,
                                )
                            ],
                        )
                        for event in events
                    ],
                    value,
                )
            else:
                cached_payload = (
                    {k: v.cpu() for k, v in value.items()}
                    if isinstance(value, dict)
                    else value.cpu()
                )
        for event in events:
            key = f"{event.request_id}:{event.incarnation}"
            if cached_payload is not None:
                row = self.manifest.ready.index(
                    (event.request_id, event.incarnation, event.output_index)
                )
                self.target_cache.append(
                    event.request_id,
                    event.incarnation,
                    event.output_index,
                    self.manifest.contract[0],
                    {k: v[row] for k, v in cached_payload.items()}
                    if isinstance(cached_payload, dict)
                    else cached_payload[row],
                )
            self.histories[key].append(event.token_id)
            self.history_digests[key] = advance_history_digest(
                self.history_digests[key], event.token_id
            )
            self.decision_ready.discard(key)
        self.phase = "COMMITTED"
        self.applied_events = list(events)
        return digest

    def retire(self, digest):
        self._check(digest, "COMMITTED")
        # This stream contains token updates, target copies and waits for the
        # transport stream. Fence it before reusing pinned commit buffers.
        if self.settings.fused_dispatch:
            self.torch.cuda.current_stream().synchronize()
        else:
            self.torch.cuda.synchronize()
        self.last_retired = (self.manifest.epoch, self.manifest.step_id)
        self.phase = "IDLE"
        return digest

    def targets(self, requests):
        if self.target_cache is None:
            return None
        return {
            rid: self.target_cache.take(rid, inc, length, version)
            for rid, inc, length, version in requests
        }

    def sampling(self, payload):
        from dataclasses import replace

        if self.target_cache is not None and self.target_cache.rows:
            raise RuntimeError("consume targets before changing sampling")
        self.settings = replace(
            self.settings,
            policy=payload["policy"],
            temperature=payload["temperature"],
            top_k=payload["top_k"],
        )
        if self.side == 1:
            self.head.policy = payload["policy"]
            self.head.temperature = payload["temperature"]
            self.head.top_k = payload["top_k"]
        self.capture_active = payload["capture"]
        return True

    def sleep(self, _):
        if self.side == 0 and self.settings.keep_teacher_resident:
            return True
        self.worker.sleep(level=2)
        self.sleeping = True
        return True

    def wake(self, _):
        if self.side == 0 and self.settings.keep_teacher_resident:
            return True
        self.worker.wake_up()
        self.sleeping = False
        return True

    def sync(self, payload):
        from paired_vllm.training_ops import load_vllm_weights, read_weights

        if self.target_cache is not None and self.target_cache.rows:
            raise RuntimeError("consume targets before changing teacher weights")
        sync_teacher = payload.get("sync_teacher", True)
        if not sync_teacher and not (
            self.settings.keep_teacher_resident and self.teacher_synced
        ):
            raise RuntimeError(
                "skipping teacher refresh requires an initialized resident teacher"
            )
        path = payload["teacher"] if self.side == 0 else payload["student"]
        if self.side == 1 or sync_teacher:
            load_vllm_weights(self.worker, path)
        if self.side == 1 and sync_teacher:
            with self.torch.inference_mode():
                self.teacher_projection.copy_(
                    read_weights(payload["teacher"])["lm_head.weight"]
                )
        self.teacher_synced = True
        self.metadata["checkpoint_revision"] = payload["version"]
        self.torch.manual_seed(payload["seed"])
        return True

    def inspect(self, _):
        torch = self.torch
        torch.cuda.synchronize()
        rs = self.runner.req_states
        state = {}
        if self.side == 0 and self.comm is not None:
            # Debug-only comparison; production never projects teacher logits here.
            slots = list(rs.req_id_to_index.values())
            with torch.inference_mode():
                self.logits[slots] = self.runner.model.compute_logits(
                    self.hidden[slots]
                ).float()
        for key, idx in rs.req_id_to_index.items():
            total = int(rs.total_len.gpu[idx].item())
            state[key] = {
                "tokens": rs.all_token_ids.gpu[idx, :total].cpu().tolist(),
                "computed": int(rs.num_computed_tokens.gpu[idx].item()),
                "history": self.histories[key],
                "logits": self.logits[idx].cpu().tolist(),
            }
        if self.side == 1 and self.reconstructed_logits is not None:
            for i, (rid, inc, _) in enumerate(self.manifest.ready):
                state[f"{rid}:{inc}"]["teacher_logits"] = (
                    self.reconstructed_logits[i].cpu().tolist()
                )
        return state
