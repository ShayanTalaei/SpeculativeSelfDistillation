"""Opt-in bitwise audit for unsharded, unquantized Qwen-style vLLM weights."""

import torch


@torch.no_grad()
def verify_colocated_weights(source, target):
    state = source.state_dict()
    packed = getattr(target, "packed_modules_mapping", {})
    checked = set()
    target_count = 0
    for name, value in target.named_parameters():
        if name in state:
            names = [name]
            expected = state[name]
        else:
            names = None
            for fused, parts in packed.items():
                marker = f".{fused}."
                if marker in name:
                    names = [name.replace(marker, f".{part}.") for part in parts]
                    break
            if not names or not all(key in state for key in names):
                raise ValueError(f"Cannot audit inference weight {name}")
            expected = torch.cat([state[key] for key in names], dim=0)
        if expected.shape != value.shape or not torch.equal(expected, value):
            raise ValueError(f"Inference weights differ from current student: {name}")
        checked.update(names)
        target_count += 1
    missing = set(state) - checked
    # named_parameters removes tied aliases; verify that any omitted state
    # entry shares exactly the same storage view with a checked source weight.
    views = {
        (state[n].data_ptr(), tuple(state[n].shape), state[n].stride()) for n in checked
    }
    for name in missing:
        tensor = state[name]
        if (tensor.data_ptr(), tuple(tensor.shape), tensor.stride()) not in views:
            raise ValueError(f"Unchecked source tensor: {name}")
    return {
        "source_tensors": len(state),
        "inference_tensors": target_count,
        "status": "passed",
    }
