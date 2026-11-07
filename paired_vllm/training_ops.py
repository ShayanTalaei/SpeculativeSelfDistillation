"""Full-vocabulary distillation with bounded projection memory and CPU weights."""

import json
from pathlib import Path

import numpy as np
import torch


class TargetCache:
    """Committed post-norm features, tied to a particular teacher snapshot."""

    def __init__(self):
        self.rows = {}

    def append(self, request_id, incarnation, index, version, hidden):
        key = (request_id, incarnation, version)
        rows = self.rows.setdefault(key, [])
        if index != len(rows):
            raise ValueError("target cache output order mismatch")
        rows.append(
            {k: v.detach().cpu().clone() for k, v in hidden.items()}
            if isinstance(hidden, dict)
            else hidden.detach().cpu().clone()
        )

    def take(self, request_id, incarnation, length, version):
        key = (request_id, incarnation, version)
        if key not in self.rows:
            raise ValueError("target cache version or request mismatch")
        if len(self.rows[key]) != length:
            raise ValueError("target cache completion length mismatch")
        rows = self.rows.pop(key)
        if isinstance(rows[0], dict):
            return {k: torch.stack([r[k] for r in rows]) for k in rows[0]}
        return torch.stack(rows)


class BufferedHiddenCache(TargetCache):
    """Copy each committed batch into device slabs; download only on take().

    Request/version identities remain separate from reusable scheduler slots.
    Memory scales with unconsumed targets, not the number of live KV slots.
    """

    def __init__(self, slab_tokens=4096):
        super().__init__()
        if slab_tokens < 1:
            raise ValueError("slab size must be positive")
        self.slab_tokens = slab_tokens
        self.slab = None
        self.used = 0

    def append_batch(self, entries, hidden):
        # Validate the entire transaction before changing storage or histories.
        seen = set()
        for rid, inc, index, version, row in entries:
            key = (rid, inc, version)
            if key in seen or index != len(self.rows.get(key, [])):
                raise ValueError("target cache output order mismatch")
            if not 0 <= row < len(hidden):
                raise ValueError("target cache row outside batch")
            seen.add(key)
        if not entries:
            return
        n = len(hidden)
        if self.slab is None or self.used + n > len(self.slab):
            self.slab = hidden.new_empty((max(self.slab_tokens, n), *hidden.shape[1:]))
            self.used = 0
        copied = self.slab[self.used : self.used + n]
        copied.copy_(hidden.detach())
        self.used += n
        for rid, inc, index, version, row in entries:
            self.rows.setdefault((rid, inc, version), []).append(copied[row])

    def take(self, request_id, incarnation, length, version):
        result = super().take(request_id, incarnation, length, version).cpu()
        if not self.rows:
            self.slab = None
            self.used = 0
        return result


def capture_topk(student, teacher, k, temperature):
    """Same teacher/student top-k union and full normalizer as legacy cache."""
    k = min(k, teacher.shape[-1])
    values, indices = teacher.float().topk(k, dim=-1)
    student_indices = student.float().topk(k, dim=-1).indices
    duplicates = (student_indices[..., None] == indices[:, None, :]).any(-1)
    student_values = teacher.float().gather(-1, student_indices)
    student_indices = student_indices.masked_fill(duplicates, -1)
    return {
        "indices": torch.cat((indices, student_indices), -1),
        "logits": torch.cat((values, student_values), -1),
        "lse": (teacher.float() / temperature).logsumexp(-1),
    }


def write_weights(path, state):
    """Create an mmap layout once; callers update these CPU tensors in place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    layout, offset = {}, 0
    aliases = {}
    for name, value in state.items():
        size = value.numel() * value.element_size()
        identity = (value.device, value.data_ptr(), tuple(value.shape), value.dtype)
        if identity in aliases:
            layout[name] = layout[aliases[identity]]
            continue
        aliases[identity] = name
        layout[name] = [
            list(value.shape),
            str(value.dtype).split(".")[-1],
            offset,
            size,
        ]
        offset += size
    with path.with_suffix(".bin").open("wb") as stream:
        stream.truncate(offset)
    path.with_suffix(".json").write_text(json.dumps(layout))
    mapped = read_weights(path)
    for name, value in state.items():
        mapped[name].copy_(value.detach())
    return mapped


def read_weights(path):
    path = Path(path)
    buffer = np.memmap(path.with_suffix(".bin"), dtype=np.uint8, mode="r+")
    layout = json.loads(path.with_suffix(".json").read_text())
    return {
        name: torch.from_numpy(buffer[offset : offset + size])
        .view(getattr(torch, dtype))
        .reshape(shape)
        for name, (shape, dtype, offset, size) in layout.items()
    }


class _ForwardKL(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden,
        weight,
        target_hidden,
        target_weight,
        chunk,
        sampled_ids,
        sampling_logps,
    ):
        ctx.save_for_backward(hidden, weight, target_hidden, target_weight)
        ctx.chunk = chunk
        total = hidden.new_zeros((), dtype=torch.float32)
        importance_sum = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, len(hidden), chunk):
            end = start + chunk
            teacher = (
                (target_hidden[start:end] @ target_weight.T).float().log_softmax(-1)
            )
            student = (hidden[start:end] @ weight.T).float().log_softmax(-1)
            total += (teacher.exp() * (teacher - student)).sum()
            if sampling_logps is not None:
                current = student.gather(-1, sampled_ids[start:end, None]).squeeze(-1)
                importance_sum += (
                    (current - sampling_logps[start:end]).exp().clamp(max=2).sum()
                )
        ctx.importance = (
            importance_sum / len(hidden) if sampling_logps is not None else 1.0
        )
        return total / len(hidden) * ctx.importance

    @staticmethod
    def backward(ctx, grad):
        hidden, weight, target_hidden, target_weight = ctx.saved_tensors
        dh, dw = torch.empty_like(hidden), torch.zeros_like(weight)
        for start in range(0, len(hidden), ctx.chunk):
            end = start + ctx.chunk
            teacher = (target_hidden[start:end] @ target_weight.T).float().softmax(-1)
            student = (hidden[start:end] @ weight.T).float().softmax(-1)
            delta = ((student - teacher) * (grad * ctx.importance / len(hidden))).to(
                hidden.dtype
            )
            dh[start:end] = delta @ weight
            dw.addmm_(delta.T, hidden[start:end])
        return dh, dw, None, None, None, None, None


def chunked_forward_kl(
    hidden,
    weight,
    target_hidden,
    target_weight,
    chunk_size=128,
    sampled_ids=None,
    sampling_logps=None,
):
    if not len(hidden) or len(hidden) != len(target_hidden) or chunk_size < 1:
        raise ValueError("invalid target length or projection chunk size")
    if sampling_logps is not None and (
        sampled_ids is None
        or len(sampled_ids) != len(hidden)
        or len(sampling_logps) != len(hidden)
    ):
        raise ValueError("importance correction must cover every completion token")
    return _ForwardKL.apply(
        hidden,
        weight,
        target_hidden,
        target_weight,
        chunk_size,
        sampled_ids,
        sampling_logps,
    )


def load_vllm_weights(worker, path):
    """Stock vLLM collective_rpc callable: transport a path, never GPU logits."""
    with torch.inference_mode():
        worker.model_runner.model.load_weights(
            (name, tensor.to("cuda")) for name, tensor in read_weights(path).items()
        )
        torch.cuda.synchronize()
    return True


class WeightUpdateExtension:
    """Named stock worker RPC; no arbitrary function deserialization needed."""

    def update_training_weights(self, path):
        return load_vllm_weights(self, path)
