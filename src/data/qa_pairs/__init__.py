"""Question/answer pairs about one article (Company Memo, Wikipedia).

The directory holds exactly one article (``*.txt``) and one Q/A file
(``*_questions_answers_split_fixed.json``) of the form
``{"train": {"question": [...], "answer": [...]}, "test": {...}}``.
"""

import json
from glob import glob
from pathlib import Path

from data.core import DistillationRow


def load(*, path: str, split: str = "train", **_kw) -> list[DistillationRow]:
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    articles = sorted(glob(str(Path(path) / "*.txt")))
    qa_files = sorted(glob(str(Path(path) / "*_questions_answers_split_fixed.json")))
    if len(articles) != 1 or len(qa_files) != 1:
        raise ValueError(
            f"{path} must contain exactly one article .txt and one Q/A json, "
            f"found {len(articles)} and {len(qa_files)}"
        )
    article_text = Path(articles[0]).read_text()
    qa = json.loads(Path(qa_files[0]).read_text())[split]
    name = Path(articles[0]).stem
    return [
        DistillationRow(
            question=q,
            documents=article_text,
            answer=a,
            metadata={"doc_id": name, "split": split, "index": i},
        )
        for i, (q, a) in enumerate(zip(qa["question"], qa["answer"]))
    ]
