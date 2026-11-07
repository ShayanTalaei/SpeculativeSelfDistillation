#!/usr/bin/env python3
"""Build a training dataset once; every trainer loads the saved path.

Output schema (one HuggingFace dataset on disk plus metadata.json):
    prompt               student chat messages
    teacher_prompt       teacher chat messages (question + gold answer in <context>)
    answer               gold completion text
    chat_template_kwargs passed to apply_chat_template at training time

Examples::

    python src/data/build.py dataset=qa_pairs path=data/company_memo \
        output=scratch/data/builds/company_memo
    python src/data/build.py dataset=scienceqa output=scratch/data/builds/scienceqa

``scripts/prepare_data.py`` runs these for the three released tasks.
"""

import pydra

from data import DATASETS
from data.core import CONTEXT_KWARGS, save_built_dataset, to_distil_dataset


class BuildConfig(pydra.Config):
    def __init__(self) -> None:
        super().__init__()
        self.dataset = pydra.REQUIRED  # key into DATASETS
        self.output = pydra.REQUIRED
        self.path: str | None = None  # qa_pairs: directory with the article and Q/A json
        self.split = "train"
        self.seed = 42

        # Teacher context (the released recipe): the teacher sees the question
        # and the gold answer inside <context>, placed before the question.
        self.enable_thinking = False
        self.context_documents = False
        self.context_question = True
        self.context_answer = True
        self.context_position = "before"

    def finalize(self):
        if self.dataset not in DATASETS:
            raise ValueError(f"Unknown dataset={self.dataset!r}. Available: {sorted(DATASETS)}")
        if self.dataset == "qa_pairs" and not self.path:
            raise ValueError("dataset=qa_pairs needs path=<directory>")


@pydra.main(BuildConfig)
def main(cfg: BuildConfig) -> None:
    rows = DATASETS[cfg.dataset](path=cfg.path, split=cfg.split, seed=cfg.seed)
    context_kwargs = {k: getattr(cfg, k) for k in CONTEXT_KWARGS}
    dataset = to_distil_dataset(rows, enable_thinking=cfg.enable_thinking, **context_kwargs)
    metadata = {
        "dataset": cfg.dataset,
        "path": cfg.path,
        "split": cfg.split,
        "num_rows": len(dataset),
        "seed": cfg.seed,
        "enable_thinking": cfg.enable_thinking,
        "columns": dataset.column_names,
        **context_kwargs,
    }
    save_built_dataset(dataset, cfg.output, metadata)


if __name__ == "__main__":
    main()
