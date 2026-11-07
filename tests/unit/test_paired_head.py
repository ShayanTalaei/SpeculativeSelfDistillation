"""Independent mathematical checks of the source-before-token contract."""

import math

import pytest
import torch

from paired_vllm.head import DecisionHead


def test_sampling_uses_selected_source_topk_and_preserves_teacher_targets():
    features = torch.tensor([[20.0, -20.0, -20.0], [-20.0, 20.0, -20.0]])
    own = torch.tensor([[-20.0, -20.0, 20.0], [20.0, -20.0, -20.0]])
    routes, tokens = torch.empty(2, dtype=torch.int32), torch.empty(2, dtype=torch.long)
    head = DecisionHead(
        torch,
        torch.eye(3),
        features,
        routes,
        tokens,
        "alternating",
        temperature=1.0,
        top_k=1,
    )
    torch.testing.assert_close(head(own, [0, 1]), features)
    assert tokens.tolist() == [0, 0]


def probabilities(logits):
    normalizer = sum(math.exp(x) for x in logits)
    return [math.exp(x) / normalizer for x in logits]


@pytest.mark.parametrize(
    "policy",
    [
        "teacher",
        "student",
        "alternating",
        "kl_forward:0.5",
        "kl_reverse:0.5",
        "jsd:0.1",
    ],
)
def test_projection_router_and_selected_token_match_scalar_oracle(policy):
    teacher_rows = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]]
    student_rows = [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 3.0]]
    features = torch.tensor(teacher_rows)
    routes = torch.empty(3, dtype=torch.int32)
    tokens = torch.empty(3, dtype=torch.int64)
    head = DecisionHead(torch, torch.eye(3), features, routes, tokens, policy)
    rng_before = torch.random.get_rng_state().clone()
    reconstructed = head(torch.tensor(student_rows), [0, 1, 2])
    assert torch.equal(reconstructed, features)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    expected_sources, expected_tokens = [], []
    for i, (t_logits, s_logits) in enumerate(zip(teacher_rows, student_rows)):
        teacher, student = probabilities(t_logits), probabilities(s_logits)
        if policy == "teacher":
            source = True
        elif policy == "student":
            source = False
        elif policy == "alternating":
            source = i % 2 == 0
        else:
            kind, threshold = policy.split(":")
            if kind == "kl_forward":
                score = sum(p * math.log(p / q) for p, q in zip(teacher, student))
            elif kind == "kl_reverse":
                score = sum(p * math.log(p / q) for p, q in zip(student, teacher))
            else:
                mixture = [(p + q) / 2 for p, q in zip(teacher, student)]
                score = (
                    sum(
                        p * math.log(p / m) + q * math.log(q / m)
                        for p, q, m in zip(teacher, student, mixture)
                    )
                    / 2
                )
            source = score > float(threshold)
        selected = teacher if source else student
        expected_sources.append(int(source))
        expected_tokens.append(max(range(3), key=selected.__getitem__))
    assert routes.tolist() == expected_sources
    assert tokens.tolist() == expected_tokens
