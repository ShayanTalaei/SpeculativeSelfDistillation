"""Dataset registry: name -> loader returning ``list[DistillationRow]``.

    qa_pairs   Company Memo and Wikipedia: a directory with one article .txt
               and one *_questions_answers_split_fixed.json (train/test Q/A).
    scienceqa  SciKnowEval Chemistry L3 from the upstream SDFT repository.
"""

from data import qa_pairs, scienceqa

DATASETS = {
    "qa_pairs": qa_pairs.load,
    "scienceqa": scienceqa.load,
}
