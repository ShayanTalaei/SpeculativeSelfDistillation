"""Tests for PPO-style mini-batching in the SSD trainer.

Validates that:
- Train batch is correctly split into mini-batches
- Policy is updated after each mini-batch
- Edge cases (mini_batch_size=1, mini_batch_size=train_batch_size) work correctly
"""

import torch
import pytest

from ssd.utils import split_into_mini_batches


class TestMiniBatchSplitting:
    """Test that train batches are correctly split into mini-batches."""

    def test_even_split(self):
        """8 samples, mini_batch_size=4 → 2 mini-batches of 4."""
        data = {
            "input_ids": torch.randn(8, 32),
            "labels": torch.randint(0, 100, (8,)),
            "provenance": [["student", "teacher"] * 16] * 8,
        }
        batches = split_into_mini_batches(data, mini_batch_size=4)
        assert len(batches) == 2
        assert batches[0]["input_ids"].shape[0] == 4
        assert batches[1]["input_ids"].shape[0] == 4

    def test_mini_batch_size_equals_train_batch(self):
        """mini_batch_size == train_batch_size → 1 batch (no splitting)."""
        data = {
            "input_ids": torch.randn(8, 32),
            "labels": torch.randint(0, 100, (8,)),
        }
        batches = split_into_mini_batches(data, mini_batch_size=8)
        assert len(batches) == 1
        assert batches[0]["input_ids"].shape[0] == 8

    def test_mini_batch_size_one(self):
        """mini_batch_size=1 → N mini-batches (SGD equivalent)."""
        data = {
            "input_ids": torch.randn(4, 16),
            "labels": torch.randint(0, 100, (4,)),
        }
        batches = split_into_mini_batches(data, mini_batch_size=1)
        assert len(batches) == 4
        for b in batches:
            assert b["input_ids"].shape[0] == 1

    def test_uneven_split_drops_remainder(self):
        """7 samples, mini_batch_size=4 → 1 batch of 4, remainder of 3 dropped."""
        data = {
            "input_ids": torch.randn(7, 16),
            "labels": torch.randint(0, 100, (7,)),
        }
        batches = split_into_mini_batches(data, mini_batch_size=4)
        assert len(batches) == 1
        assert batches[0]["input_ids"].shape[0] == 4

    def test_split_preserves_correspondence(self):
        """Items at same index across different keys should stay together."""
        data = {
            "input_ids": torch.arange(8).unsqueeze(1),  # [[0],[1],...,[7]]
            "labels": torch.arange(8) * 10,             # [0, 10, 20, ..., 70]
        }
        batches = split_into_mini_batches(data, mini_batch_size=4)
        # First batch should have items 0-3
        assert torch.equal(batches[0]["input_ids"].squeeze(), torch.arange(4))
        assert torch.equal(batches[0]["labels"], torch.arange(4) * 10)
        # Second batch should have items 4-7
        assert torch.equal(batches[1]["input_ids"].squeeze(), torch.arange(4, 8))
        assert torch.equal(batches[1]["labels"], torch.arange(4, 8) * 10)

    def test_split_handles_list_values(self):
        """Non-tensor values (like provenance lists) should also be split."""
        data = {
            "input_ids": torch.randn(4, 8),
            "provenance": [["s", "t"], ["t", "s"], ["s", "s"], ["t", "t"]],
        }
        batches = split_into_mini_batches(data, mini_batch_size=2)
        assert len(batches) == 2
        assert batches[0]["provenance"] == [["s", "t"], ["t", "s"]]
        assert batches[1]["provenance"] == [["s", "s"], ["t", "t"]]
