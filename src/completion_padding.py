"""Remove unused completion columns without a CUDA synchronization."""


def trim_completion_padding(inputs):
    """Use CPU rollout lengths, retained through batch shuffle and splitting.

    Lengths describe generated tokens, not the loss mask: a truncated completion
    may be masked out entirely and must still retain its original attention.
    The returned dictionary shares storage and never mutates the rollout buffer.
    """
    lengths = inputs.get("completion_lengths")
    if lengths is None:
        return inputs
    width = inputs["completion_ids"].shape[1]
    if len(lengths) != inputs["completion_ids"].shape[0]:
        raise ValueError("completion lengths must match the microbatch")
    if any(length < 0 or length > width for length in lengths):
        raise ValueError("completion length exceeds the padded width")
    length = max(1, max(lengths, default=0))
    if length >= width:
        return inputs
    trimmed = inputs.copy()
    for key in (
        "completion_ids", "completion_mask", "advantages", "provenance",
        "old_per_token_logps", "ref_per_token_logps", "importance_sampling_ratio",
    ):
        value = inputs.get(key)
        if value is not None and value.ndim >= 2 and value.shape[1] == width:
            trimmed[key] = value[:, :length]
    if "token_type_ids" in inputs:
        prompt_width = inputs["prompt_ids"].shape[1]
        trimmed["token_type_ids"] = inputs["token_type_ids"][:, :prompt_width + length]
    return trimmed
