"""Training bridge: lockstep student/teacher rollouts with GPU-resident teacher targets."""

from paired_vllm.async_core import RolloutRequest
from paired_vllm.core import PairConfig
from paired_vllm.engine import RuntimeConfig
from paired_vllm.lockstep_engine import LockstepPairedEngine


class LockstepTrainingBridge:
    bridge_commands = True

    def __init__(self, config):
        if config["policy"].partition(":")[0] not in ("jsd", "kl_forward", "kl_reverse"):
            raise ValueError("lockstep training requires JSD or KL routing")
        self.config = config
        runtime = RuntimeConfig(
            teacher_model=config["model_b"],
            student_model=config["model_a"],
            teacher_gpu=config["teacher_gpu"],
            student_gpu=config["student_gpu"],
            dtype="bfloat16",
            policy=config["policy"],
            max_model_len=config["max_model_len"],
            scheduler=PairConfig(
                capacity=config["capacity"],
                token_budget=config["token_budget"],
                max_requests=config["rollout_group_size"],
            ),
            memory_utilization=config["memory_utilization"],
            graphs=True,
            compile_model=config.get("compile_model", True),
            enable_sleep_mode=True,
            # The teacher never leaves its GPU: a frozen teacher is uploaded
            # once, an EMA teacher is refreshed in place every SYNC.
            keep_teacher_resident=True,
            capture_targets=True,
            target_format="hidden",
            temperature=config["temperature"],
            top_k=config["top_k"],
            startup_timeout=600,
            timeout=600,
        )
        self.engine = LockstepPairedEngine(
            runtime,
            readback_interval=8,
            # Logits transfer ships fp32 vocab rows per token; hidden transfer
            # ships post-norm states and projects them with the student-side
            # copy of the teacher head, which SYNC refreshes from the EMA export.
            transfer_logits=config["transfer_logits"],
            fused_jsd=True,
            resident_targets=True,
        )

    def call(self, command, payload):
        if command == "RELEASE_TARGETS":
            self.engine.release_targets(payload)
            return True
        if command != "GENERATE":
            return self.engine.training_command(command, payload)
        if payload["eval"]:
            return self.engine.evaluate_student(payload)
        capture = not payload["eval"]
        self.engine.training_command(
            "SAMPLING",
            {
                "policy": "student" if payload["eval"] else self.config["policy"],
                "temperature": payload["temperature"],
                "top_k": payload["top_k"],
                "capture": capture,
            },
        )
        if len(payload["student"]) != len(payload["teacher"]):
            raise ValueError("teacher/student rollout counts differ")
        requests = [
            RolloutRequest(
                str(i),
                teacher,
                student,
                payload["max_tokens"],
                tuple(payload["stop_tokens"]),
            )
            for i, (student, teacher) in enumerate(
                zip(payload["student"], payload["teacher"])
            )
        ]
        generated = self.engine.generate(requests)
        targets = self.engine.targets() if capture else {}
        rows = []
        for item in generated:
            row = {"token_ids": item["token_ids"], "provenance": item["sources"]}
            if capture:
                cache = targets[item["request_id"]]
                if cache["version"] != str(payload["version"]):
                    raise ValueError(
                        "rollout targets came from the wrong teacher snapshot"
                    )
                row["cuda_cache"] = cache
            rows.append(row)
        if rows:
            rows[0]["timings"] = {
                f"generation_{key}": value
                for key, value in self.engine.last_metrics.items()
            }
        return rows

    def close(self):
        self.engine.close()
