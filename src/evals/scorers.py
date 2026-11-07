"""Scorer for the ScienceQA (SciKnowEval Chemistry L3) multiple-choice evaluation.

Scorers follow the interface ``scorer(question, reference, response) -> dict``
with at least a ``score`` key in [0, 1]. They plug into the online evaluator
through the ``eval_scoring`` config value, e.g.
``eval_scoring=evals.scorers:science_xml_match_scorer``.
"""


def science_xml_match_scorer(
    question: str,
    reference_answer: str,
    model_response: str,
) -> dict:
    """Extract the last ``<answer>...</answer>`` tag and compare it exactly.

    Matches ``eval_science.py:extract_xml_answer`` in the upstream SDFT repo.
    """
    extracted = model_response.split("<answer>")[-1].split("</answer>")[0].strip()
    match = extracted == reference_answer.strip()
    return {
        "score": 1.0 if match else 0.0,
        "verdict": "CORRECT" if match else "INCORRECT",
        "extracted": extracted,
    }
