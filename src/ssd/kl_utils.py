"""Shared FKL / KL helpers.

These functions are the canonical implementations used by both the
off-policy trainer (``train_off_policy.py``) and the offline measurement
tools (``offline/measure_fkl.py``). Training and measurement MUST share
the same tokenization boundaries and KL definition or comparisons across
the two become meaningless.
"""
import torch
import torch.nn.functional as F


def tokenize_for_fkl(
    row: dict,
    tokenizer,
    max_student_len: int,
    max_teacher_len: int,
    chat_template_kwargs: dict | None = None,
) -> dict | None:
    """Tokenize one distil-format row for FKL/adaptive training.

    Expected row fields (the universal distil format):
        prompt:         list[dict]  — student chat messages [system?, user]
        teacher_prompt: list[dict]  — teacher chat messages
        answer:         str          — the completion both models score on
        chat_template_kwargs: dict  — kwargs for apply_chat_template; the
            function's own ``chat_template_kwargs`` argument wins when set.

    Teacher-prompt parity between training and generation depends on the
    upstream pipeline: when the dataset was built via ``data.build
    format=distil`` from the SAME source the generator consumed, with the
    same ``context_documents``/``context_answer`` flags, the
    ``teacher_prompt`` column here is byte-identical to what the generator
    saw. Otherwise it may differ in framing, for example raw documents in the
    teacher's system message versus a ``<context>...</context>`` block.

    Returns None if, after truncation, there are no completion tokens left.
    """
    if row.get("answer") is None:
        raise ValueError(
            "row['answer'] is None — distil-format rows destined for FKL "
            "training must carry a completion target. On-policy-only rows "
            "(answer=None) should not reach tokenize_for_fkl."
        )

    ck = chat_template_kwargs or row.get("chat_template_kwargs") or {}

    student_chat = list(row["prompt"]) + [
        {"role": "assistant", "content": row["answer"]}
    ]
    teacher_chat = list(row["teacher_prompt"]) + [
        {"role": "assistant", "content": row["answer"]}
    ]

    student_prompt_ids = tokenizer.apply_chat_template(
        row["prompt"], add_generation_prompt=True, tokenize=True, return_dict=False, **ck,
    )
    student_full_ids = tokenizer.apply_chat_template(
        student_chat, tokenize=True, return_dict=False, **ck,
    )
    teacher_prompt_ids = tokenizer.apply_chat_template(
        row["teacher_prompt"], add_generation_prompt=True, tokenize=True, return_dict=False, **ck,
    )
    teacher_full_ids = tokenizer.apply_chat_template(
        teacher_chat, tokenize=True, return_dict=False, **ck,
    )

    comp_start_s = len(student_prompt_ids)
    comp_start_t = len(teacher_prompt_ids)
    num_comp = min(
        len(student_full_ids) - comp_start_s,
        len(teacher_full_ids) - comp_start_t,
    )
    if num_comp < 1:
        return None

    if len(student_full_ids) > max_student_len:
        student_full_ids = student_full_ids[:max_student_len]
        num_comp = min(num_comp, max_student_len - comp_start_s)
    if len(teacher_full_ids) > max_teacher_len:
        teacher_full_ids = teacher_full_ids[:max_teacher_len]
        num_comp = min(num_comp, max_teacher_len - comp_start_t)

    if num_comp < 1:
        return None

    return {
        "student_ids": student_full_ids,
        "teacher_ids": teacher_full_ids,
        "completion_start_student": comp_start_s,
        "completion_start_teacher": comp_start_t,
        "num_completion_tokens": num_comp,
    }


def batch_forward_slice(
    model,
    ids_list: list[list[int]],
    cs_list: list[int],
    nc_list: list[int],
    pad_id: int,
    device,
    with_grad: bool = True,
) -> torch.Tensor:
    """Run a padded batched forward and return concatenated completion-slice logits.

    Pads all sequences in ``ids_list`` on the right to the max length in the
    batch, runs one forward pass with an attention mask, then for each sample
    slices logits[i, cs_i - 1 : cs_i + nc_i - 1] and concatenates across the
    batch. Returns a [sum(nc_i), vocab] tensor ready for loss computation.
    """
    max_len = max(len(x) for x in ids_list)
    B = len(ids_list)
    batch = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    for i, x in enumerate(ids_list):
        L = len(x)
        batch[i, :L] = torch.tensor(x, dtype=torch.long, device=device)
        mask[i, :L] = 1

    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        out = model(input_ids=batch, attention_mask=mask)
    slices = []
    for i, (cs, nc) in enumerate(zip(cs_list, nc_list)):
        slices.append(out.logits[i, cs - 1: cs + nc - 1, :])
    return torch.cat(slices, dim=0)


def compute_fkl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Forward KL: KL(P_teacher || P_student) averaged over completion tokens."""
    s_logits = student_logits / temperature
    t_logits = teacher_logits / temperature

    log_p_s = F.log_softmax(s_logits, dim=-1)
    log_p_t = F.log_softmax(t_logits, dim=-1)
    p_t = log_p_t.exp()

    kl_per_token = (p_t * (log_p_t - log_p_s)).sum(dim=-1)
    return kl_per_token.mean()


def per_token_fkl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Per-token forward KL over the completion slice (no mean)."""
    s_logits = student_logits / temperature
    t_logits = teacher_logits / temperature
    log_p_s = F.log_softmax(s_logits, dim=-1)
    log_p_t = F.log_softmax(t_logits, dim=-1)
    p_t = log_p_t.exp()
    return (p_t * (log_p_t - log_p_s)).sum(dim=-1)
