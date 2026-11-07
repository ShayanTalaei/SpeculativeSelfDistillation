"""Trainer-side bridge to the isolated lockstep vLLM pair (``paired_vllm.server``).

The trainer runs in one process on the student GPU. It spawns the server as a
subprocess that owns two vLLM workers: the student worker shares GPU 0 with
training and the teacher worker lives on GPU 1. Every rollout step the trainer
exports its current student weights (and, with an EMA teacher, the reference
model) as CUDA IPC handles, the server refreshes both workers, generates the
speculative rollouts and leaves the teacher's hidden-state targets resident on
the student GPU, where the loss projects them through the teacher head.
"""

import os
import shutil
import socket
import subprocess
import tempfile
import time
from multiprocessing.connection import Connection
from pathlib import Path
from types import SimpleNamespace

import torch

from ssd.generation import GenerationOutput
from ssd.paired_cache import FullTeacherCache
from paired_vllm.async_core import eos_token_ids
from paired_vllm.training_ops import write_weights


class PairedRunner:
    def __init__(
        self, trainer, model_a, model_b, scratch, max_model_len, memory_utilization
    ):
        self.trainer = trainer
        args = trainer.args
        if (
            trainer.accelerator.num_processes != 1
            or trainer.is_fsdp_enabled
            or trainer.is_deepspeed_enabled
        ):
            raise ValueError(
                "paired_vllm requires one training process and two rollout GPUs"
            )
        if (
            args.steps_per_generation != args.gradient_accumulation_steps
            or args.num_iterations != 1
        ):
            raise ValueError(
                "paired_vllm requires one rollout batch per optimizer step (steps_per_generation=GA, num_iterations=1)"
            )
        if args.routing_strategy == "jsd_switch":
            policy = f"jsd:{args.jsd_threshold}"
        elif args.routing_strategy == "kl_switch":
            policy = f"kl_{args.kl_direction}:{args.kl_threshold}"
        else:
            raise ValueError(
                "SSD routes on a divergence: routing_strategy must be jsd_switch or kl_switch"
            )
        self.tokenizer = trainer.processing_class
        self.stop_tokens = eos_token_ids(
            self.tokenizer.eos_token_id,
            getattr(
                getattr(trainer.model, "generation_config", None), "eos_token_id", None
            ),
        )
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")

        def physical_gpu(value):
            # Pydra can parse numeric CLI overrides as ints even when the
            # config default is a string; UUID overrides remain strings.
            value = str(value)
            if visible != [""] and value.isdigit():
                return visible[int(value)]
            return value

        os.makedirs(scratch, exist_ok=True)
        self.scratch = Path(tempfile.mkdtemp(prefix="paired-", dir=scratch))
        self.config = {
            "model_a": model_a,
            "model_b": model_b,
            "policy": policy,
            "scratch": str(self.scratch),
            "max_model_len": max_model_len,
            "memory_utilization": memory_utilization,
            "teacher_gpu": physical_gpu(args.paired_teacher_gpu),
            "student_gpu": physical_gpu(args.paired_student_gpu),
            "capacity": args.paired_capacity,
            "token_budget": args.paired_token_budget,
            "compile_model": args.paired_compile_model,
            "temperature": args.temperature,
            "top_k": args.top_k or 0,
            "transfer_logits": args.paired_transfer_logits,
            "rollout_group_size": args.per_device_train_batch_size
            * args.steps_per_generation,
        }
        # Training weights and optimizer state stay on the GPU during rollouts;
        # the student worker maps the live training allocations through cached
        # CUDA IPC handles instead of re-uploading weights every step.
        self.keep_student_resident = True
        self.cache_weight_handles = True
        self.process = self.conn = None
        self.maps = {}
        self.weight_staging = {}
        self.cuda_weights = {}
        self.cuda_weight_refs = {}
        self.previous_cuda_weight_refs = []
        self.sleeping = True
        self.offloaded = False
        self.device = trainer.accelerator.device
        self.version = None
        self.sync_count = 0
        self.phase_start = None
        self.phase_evaluation = False

    def _call(self, command, payload=None):
        started = time.perf_counter()
        self.conn.send((command, payload))
        if not self.conn.poll(1800):
            self.close()
            raise TimeoutError(
                f"paired bridge timed out on {command}; see paired-worker.log"
            )
        ok, result = self.conn.recv()
        if not ok:
            raise RuntimeError(result)
        if not self.phase_evaluation:
            self.trainer._metrics["train"][f"paired/{command.lower()}_seconds"].append(
                time.perf_counter() - started
            )
        return result

    def _move_training(self, device):
        # Preserve parameter objects, optimizer state dtypes and CPU step counters.
        self.trainer.model.to(device)
        self.trainer.ref_model.to(device)
        optimizer = self.trainer.optimizer
        if optimizer is not None:
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor) and key != "step":
                        state[key] = value.to(device)

    def wake_up_for_evaluation(self):
        self.wake_up(evaluation=True)

    def wake_up(self, evaluation=False):
        self.phase_evaluation = evaluation
        self.phase_start = time.perf_counter()
        # vLLM's initial memory profiling requires the training allocation to
        # be absent. Later rollouts share the 80 GB device with training
        # weights and optimizer state.
        if not self.offloaded and self.process is None:
            self._move_training("cpu")
            self.offloaded = True
        # Release cached backward temporaries even when weights stay resident.
        torch.cuda.empty_cache()
        if self.process is None:
            parent, child = socket.socketpair()
            env = os.environ.copy()
            # Training's rendezvous must not leak into isolated vLLM workers.
            for key in (
                "RANK",
                "LOCAL_RANK",
                "WORLD_SIZE",
                "MASTER_ADDR",
                "MASTER_PORT",
                "VLLM_ATTENTION_BACKEND",
            ):
                env.pop(key, None)
            root = Path(__file__).resolve().parents[2]
            env["PYTHONPATH"] = str(root)
            try:
                with open(
                    Path(self.trainer.args.output_dir) / "paired-worker.log", "a"
                ) as log:
                    self.process = subprocess.Popen(
                        [
                            self.trainer.args.paired_python,
                            "-m",
                            "paired_vllm.server",
                            "--fd",
                            str(child.fileno()),
                        ],
                        pass_fds=(child.fileno(),),
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        cwd=root,
                    )
            finally:
                child.close()
            self.conn = Connection(parent.detach())
            self._call("INIT", self.config)
        elif self.sleeping:
            self._call("WAKE")
        self.sleeping = False

    def update_named_params(self, side, named_params):
        started = time.perf_counter()
        state = dict(named_params)
        if side in self.cuda_weight_refs:
            # Exported allocations must outlive every importing CUDA context.
            # Keep the previous storage until SYNC has acknowledged any remap.
            self.previous_cuda_weight_refs.append(self.cuda_weight_refs[side])
        if "lm_head.weight" not in state:
            # Qwen tied embeddings may be deduplicated by named_parameters().
            model = self.trainer.model if side == "a" else self.trainer.ref_model
            if model.config.tie_word_embeddings:
                state["lm_head.weight"] = state["model.embed_tokens.weight"]
        # Side a: the student worker shares the training GPU and maps these
        # allocations directly. Side b: an EMA teacher stays resident on the
        # training GPU, and the resident teacher worker pulls it over the GPU
        # interconnect each step. Both skip the pinned/mmap host round trip.
        # A frozen teacher is uploaded once through the mmap path below.
        exportable = side == "a" or self.trainer.args.sync_ref_model
        if (
            exportable
            and all(tensor.is_cuda for tensor in state.values())
            and os.environ.get("PAIRED_WEIGHT_TRANSPORT", "cuda") == "cuda"
        ):
            from paired_vllm.cuda_weights import export_cuda_weights

            self.cuda_weights[side] = export_cuda_weights(state)
            self.cuda_weight_refs[side] = {
                name: tensor.detach() for name, tensor in state.items()
            }
            self._record_time(f"weight_export_{side}_seconds", started)
            return
        self.cuda_weights.pop(side, None)
        self.cuda_weight_refs.pop(side, None)
        if side not in self.maps:
            self.maps[side] = write_weights(self.scratch / side, state)
        else:
            if set(state) != set(self.maps[side]):
                raise ValueError("weight names changed across paired sync")
            if any(tensor.is_cuda for tensor in state.values()):
                # Pageable mmap destinations serialize slow device-to-host
                # transfers. Download into reusable pinned storage, then copy
                # into the shared file only after the GPU has finished writing.
                if side not in self.weight_staging:
                    self.weight_staging[side] = {
                        name: torch.empty_like(value, device="cpu", pin_memory=True)
                        for name, value in state.items()
                    }
                staging = self.weight_staging[side]
                for name, tensor in state.items():
                    staging[name].copy_(tensor.detach(), non_blocking=True)
                torch.cuda.synchronize(self.device)
                state = staging
            for name, tensor in state.items():
                self.maps[side][name].copy_(tensor.detach())
        self._record_time(f"weight_export_{side}_seconds", started)

    def _record_time(self, name, started):
        if not self.phase_evaluation:
            self.trainer._metrics["train"][f"paired/{name}"].append(
                time.perf_counter() - started
            )

    def _sync(self):
        # A frozen teacher stays on its dedicated GPU and needs only the
        # initial checkpoint upload; an EMA teacher is re-exported every step.
        if "b" not in self.maps and "b" not in self.cuda_weights:
            self.update_named_params("b", self.trainer.ref_model.state_dict().items())
        if "a" not in self.maps and "a" not in self.cuda_weights:
            raise RuntimeError("student weights must be synchronized before generation")
        self.version = str(self.trainer.state.global_step)
        result = self._call(
            "SYNC",
            {
                "student": str(self.scratch / "a"),
                "student_cuda": self.cuda_weights.get("a"),
                "teacher": str(self.scratch / "b"),
                "teacher_cuda": self.cuda_weights.get("b"),
                "version": self.version,
                "seed": self.trainer.args.seed + self.trainer.state.global_step,
                "sync_teacher": self.trainer.args.sync_ref_model
                or self.sync_count == 0,
                "cache_student_cuda_handles": self.cache_weight_handles,
                "cache_teacher_cuda_handles": self.cache_weight_handles,
            },
        )
        self.previous_cuda_weight_refs.clear()
        if not self.phase_evaluation and isinstance(result, dict):
            for timings in result.values():
                if isinstance(timings, dict):
                    for key, seconds in timings.items():
                        if key.endswith("_seconds"):
                            self.trainer._metrics["train"][f"paired/{key}"].append(
                                seconds
                            )
        self.sync_count += 1

    def _generate(self, students, teachers, max_tokens, temperature, top_k, evaluation):
        self.phase_evaluation = evaluation
        self._sync()

        def encode(prompts):
            return self.tokenizer(
                prompts,
                add_special_tokens=False,
                truncation=True,
                max_length=self.trainer.max_prompt_length,
            )["input_ids"]

        started = time.perf_counter()
        student_ids, teacher_ids = encode(students), encode(teachers)
        self._record_time("tokenize_seconds", started)
        return self._call(
            "GENERATE",
            {
                "student": student_ids,
                "teacher": teacher_ids,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_k": max(0, top_k or 0),
                "eval": evaluation,
                "stop_tokens": self.stop_tokens,
                "version": self.version,
            },
        )

    def generate_batch(
        self, student_prompts, teacher_prompts, max_new_tokens, temperature, top_k
    ):
        from paired_vllm.cuda_cache import copy_cuda_targets

        rows = self._generate(
            student_prompts, teacher_prompts, max_new_tokens, temperature, top_k, False
        )
        import_started = time.perf_counter()
        if any("cuda_cache" not in row for row in rows):
            raise RuntimeError("lockstep training requires GPU-resident hidden targets")
        resident = copy_cuda_targets(
            {str(i): row["cuda_cache"] for i, row in enumerate(rows)},
            device=self.device,
            version=self.trainer.state.global_step,
            token_ids={str(i): row["token_ids"] for i, row in enumerate(rows)},
        )
        leases = {row["cuda_cache"]["lease"] for row in rows}
        self._record_time("target_import_seconds", import_started)
        result = []
        for i, row in enumerate(rows):
            for key, value in row.get("timings", {}).items():
                self.trainer._metrics["train"][f"paired/{key}"].append(value)
                if key.startswith("generation_"):
                    self.trainer._metrics["train"][
                        key.replace("generation_", "generation/", 1)
                    ].append(value)
            output = GenerationOutput(
                tokens=[],
                token_ids=row["token_ids"],
                provenance=row["provenance"],
                student_logprobs=[],
                teacher_logprobs=[],
                kl_forward=[],
                kl_reverse=[],
                student_entropy=[],
                teacher_entropy=[],
                decisions=[],
            )
            output.full_teacher_cache = FullTeacherCache(
                "hidden",
                resident[str(i)],
                self.trainer.state.global_step,
                output.token_ids,
            )
            result.append(output)
        if leases:
            # copy_cuda_targets synchronizes the GPU copy before releasing the
            # producer lease. Trainer-owned tensors survive student vLLM sleep.
            self._call("RELEASE_TARGETS", list(leases))
        return result

    def sleep(self, level=1):
        if (
            self.process is not None
            and not self.sleeping
            and not self.keep_student_resident
        ):
            self._call("SLEEP")
        self.sleeping = not self.keep_student_resident
        if self.offloaded:
            self._move_training(self.device)
            self.offloaded = False
        if not self.phase_evaluation and self.phase_start is not None:
            self.trainer._metrics["train"]["paired/rollout_phase_seconds"].append(
                time.perf_counter() - self.phase_start
            )
        self.phase_start = None

    def eval_proxy(self):
        return self

    def generate(self, prompts, sampling_params, use_tqdm=False):
        rows = self._generate(
            prompts,
            prompts,
            sampling_params.max_tokens,
            sampling_params.temperature,
            sampling_params.top_k,
            True,
        )
        self.last_eval_metrics = rows[0].get("timings", {}) if rows else {}
        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text=self.tokenizer.decode(
                            row["token_ids"], skip_special_tokens=True
                        )
                    )
                ]
            )
            for row in rows
        ]

    def close(self):
        if self.process is not None:
            # vLLM remaps sleeping allocations during allocator teardown.
            # Release training residency first, just as before a normal wake.
            self._move_training("cpu")
            torch.cuda.empty_cache()
            try:
                self.conn.send(("CLOSE", None))
                self.process.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()
            self.conn.close()
            self.process = None
        self.maps.clear()
        self.weight_staging.clear()
        self.cuda_weights.clear()
        self.cuda_weight_refs.clear()
        self.previous_cuda_weight_refs.clear()
        shutil.rmtree(self.scratch, ignore_errors=True)
