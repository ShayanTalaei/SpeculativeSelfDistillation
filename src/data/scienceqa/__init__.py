"""SciKnowEval Chemistry L3 ("ScienceQA") from the upstream SDFT repository.

Reads the Arrow datasets the SDFT authors ship under
``third_party/sdft/data/science_data/{train,eval}_data`` (fetched by
``scripts/prepare_data.py scienceqa``). Training rows carry GPT-4o expert
demonstrations; evaluation rows carry the gold letter answer.
"""

from pathlib import Path

from datasets import load_from_disk

from data.core import DistillationRow

SDFT_DIR = Path(__file__).resolve().parents[3] / "third_party" / "sdft" / "data"


def load(*, split: str = "train", **_kw) -> list[DistillationRow]:
    if split not in ("train", "eval"):
        raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
    dataset = load_from_disk(str(SDFT_DIR / "science_data" / f"{split}_data"))
    rows = []
    if split == "train":
        for ex in dataset:
            rows.append(DistillationRow(
                question=ex["messages"][1]["content"],
                answer=ex["output_text"],
                system_prompt=ex["messages"][0]["content"],
                metadata={"benchmark": "scienceqa", "split": "train"},
            ))
    else:
        for ex in dataset:
            rows.append(DistillationRow(
                question=ex["prompt"][1]["content"],
                answer=ex["answer"],
                system_prompt=ex["prompt"][0]["content"],
                metadata={"benchmark": "scienceqa", "split": "eval"},
            ))
    return rows
