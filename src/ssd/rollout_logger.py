"""Rollout logger for the SSD trainer.

Saves per-token generation data (provenance, logprobs, KL, entropy, loss)
to JSON files that the trainer visualizer can load.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from ssd.generation import GenerationOutput


@dataclass
class TrainerRolloutToken:
    """Per-token data saved by the trainer for visualization."""
    token_str: str
    source: str              # 'student' | 'teacher' | 'fused'
    student_logprob: float
    teacher_logprob: float
    kl_forward: float
    kl_reverse: float
    student_entropy: float
    teacher_entropy: float
    loss_value: float        # this token's contribution to the loss
    decision_reason: str


@dataclass
class TrainerRollout:
    """A complete rollout saved by the trainer."""
    step: int
    student_prompt: str
    teacher_prompt: str
    tokens: list[TrainerRolloutToken]
    strategy_name: str
    strategy_config: dict
    total_loss: float
    student_token_count: int
    teacher_token_count: int


class RolloutLogger:
    """Accumulates rollouts and writes them to disk periodically."""

    def __init__(self, output_dir: str, log_interval: int = 10):
        self.output_dir = Path(output_dir) / "rollout_logs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.buffer: list[TrainerRollout] = []

    def log_rollout(
        self,
        step: int,
        student_prompt: str,
        teacher_prompt: str,
        gen_output: GenerationOutput,
        per_token_loss: list[float],
        strategy_name: str,
        strategy_config: dict,
        total_loss: float,
    ):
        """Convert a GenerationOutput into a TrainerRollout and buffer it."""
        tokens = []
        for i in range(len(gen_output.tokens)):
            tokens.append(TrainerRolloutToken(
                token_str=gen_output.tokens[i],
                source=gen_output.provenance[i],
                student_logprob=gen_output.student_logprobs[i],
                teacher_logprob=gen_output.teacher_logprobs[i],
                kl_forward=gen_output.kl_forward[i],
                kl_reverse=gen_output.kl_reverse[i],
                student_entropy=gen_output.student_entropy[i],
                teacher_entropy=gen_output.teacher_entropy[i],
                loss_value=per_token_loss[i] if i < len(per_token_loss) else 0.0,
                decision_reason=gen_output.decisions[i].get("reason", ""),
            ))

        s_count = sum(1 for p in gen_output.provenance if p == "student")
        t_count = sum(1 for p in gen_output.provenance if p == "teacher")

        rollout = TrainerRollout(
            step=step,
            student_prompt=student_prompt,
            teacher_prompt=teacher_prompt,
            tokens=tokens,
            strategy_name=strategy_name,
            strategy_config=strategy_config,
            total_loss=total_loss,
            student_token_count=s_count,
            teacher_token_count=t_count,
        )
        self.buffer.append(rollout)

    def should_flush(self, step: int) -> bool:
        return step % self.log_interval == 0 and len(self.buffer) > 0

    def flush(self, step: int):
        """Write buffered rollouts to a JSON file and clear the buffer."""
        if not self.buffer:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"rollouts_step_{step:06d}.json"
        data = [asdict(r) for r in self.buffer]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self.buffer.clear()

    def load_all(self) -> list[TrainerRollout]:
        """Load all saved rollout files (for the visualizer)."""
        rollouts = []
        for path in sorted(self.output_dir.glob("rollouts_step_*.json")):
            with open(path) as f:
                data = json.load(f)
            for item in data:
                tokens = [TrainerRolloutToken(**t) for t in item["tokens"]]
                item["tokens"] = tokens
                rollouts.append(TrainerRollout(**item))
        return rollouts
