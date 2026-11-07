"""Training contracts checked against independent dense PyTorch references."""

import pytest
import torch

from paired_vllm.training_ops import (
    BufferedHiddenCache,
    TargetCache,
    chunked_forward_kl,
    read_weights,
    write_weights,
)


def test_chunked_kl_value_and_all_student_gradients_match_dense():
    torch.manual_seed(5)
    h = torch.randn(7, 5, requires_grad=True)
    w = torch.randn(11, 5, requires_grad=True)
    th, tw = torch.randn(7, 3), torch.randn(11, 3)
    teacher = (th @ tw.T).log_softmax(-1)
    reference = (teacher.exp() * (teacher - (h @ w.T).log_softmax(-1))).sum(-1).mean()
    expected = torch.autograd.grad(reference, (h, w))
    actual = chunked_forward_kl(h, w, th, tw, chunk_size=3)
    grads = torch.autograd.grad(actual, (h, w))
    torch.testing.assert_close(actual, reference)
    for got, want in zip(grads, expected):
        torch.testing.assert_close(got, want)


def test_onpolicy_correction_matches_original_detached_sequence_weight():
    torch.manual_seed(7)
    h = torch.randn(7, 5, requires_grad=True)
    w = torch.randn(11, 5, requires_grad=True)
    th, tw = torch.randn(7, 3), torch.randn(11, 3)
    ids = torch.arange(7)
    teacher = (th @ tw.T).log_softmax(-1)
    student = (h @ w.T).log_softmax(-1)
    sampled_logps = student.detach()[torch.arange(7), ids] + torch.linspace(-2, 1, 7)
    ratio = (
        (student.detach()[torch.arange(7), ids] - sampled_logps)
        .exp()
        .clamp(max=2)
        .mean()
    )
    expected = (teacher.exp() * (teacher - student)).sum(-1).mean() * ratio
    grads = torch.autograd.grad(expected, (h, w))
    actual = chunked_forward_kl(
        h, w, th, tw, chunk_size=3, sampled_ids=ids, sampling_logps=sampled_logps
    )
    torch.testing.assert_close(actual, expected)
    for got, want in zip(torch.autograd.grad(actual, (h, w)), grads):
        torch.testing.assert_close(got, want)


def test_cached_targets_are_committed_in_order_and_versioned():
    cache = TargetCache()
    cache.append("r", 3, 0, "teacher-4", torch.ones(5))
    with pytest.raises(ValueError, match="order"):
        cache.append("r", 3, 2, "teacher-4", torch.ones(5))
    with pytest.raises(ValueError, match="version"):
        cache.take("r", 3, 1, "teacher-5")
    torch.testing.assert_close(cache.take("r", 3, 1, "teacher-4"), torch.ones(1, 5))
    assert not cache.rows


def test_buffered_hidden_targets_survive_source_overwrites_and_slab_rollover():
    cache = BufferedHiddenCache(slab_tokens=3)
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    first = source.clone()
    cache.append_batch([("a", 0, 0, "v1", 2), ("b", 4, 0, "v1", 0)], source)
    source.add_(100)
    second = source.clone()
    # The source buffer is reused, scheduler batch order changes, and storage
    # crosses a slab boundary. Identity must not depend on the physical slot.
    cache.append_batch([("b", 4, 1, "v1", 2), ("a", 0, 1, "v1", 1)], source)
    source.zero_()
    with pytest.raises(ValueError, match="version"):
        cache.take("a", 0, 2, "v2")
    torch.testing.assert_close(
        cache.take("a", 0, 2, "v1"), torch.stack([first[2], second[1]])
    )
    torch.testing.assert_close(
        cache.take("b", 4, 2, "v1"), torch.stack([first[0], second[2]])
    )
    assert not cache.rows and cache.slab is None


def test_buffered_hidden_batch_validation_does_not_partially_commit():
    cache = BufferedHiddenCache()
    with pytest.raises(ValueError, match="order"):
        cache.append_batch([("a", 0, 0, "v", 0), ("b", 0, 1, "v", 1)], torch.ones(2, 3))
    assert not cache.rows and cache.slab is None


def test_shared_weight_file_updates_are_visible_without_serializing_weights(tmp_path):
    source = {"a": torch.randn(3, 2, dtype=torch.bfloat16), "b": torch.ones(2)}
    writer = write_weights(tmp_path / "weights", source)
    reader = read_weights(tmp_path / "weights")
    for key in source:
        torch.testing.assert_close(writer[key], reader[key])
    writer["a"].fill_(7)
    assert reader["a"].tolist() == [[7, 7], [7, 7], [7, 7]]


def test_tied_embedding_and_projection_keep_one_shared_weight_snapshot(tmp_path):
    weight = torch.randn(7, 3, dtype=torch.bfloat16)
    state = {"model.embed_tokens.weight": weight, "lm_head.weight": weight}
    writer = write_weights(tmp_path / "tied", state)
    writer["model.embed_tokens.weight"].fill_(5)
    torch.testing.assert_close(writer["lm_head.weight"], torch.full_like(weight, 5))
    assert (
        tmp_path / "tied.bin"
    ).stat().st_size == weight.numel() * weight.element_size()
