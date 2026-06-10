"""
metrics.py
==========
Accuracy metrics for GSM8K and MATH evaluation.
"""

from typing import List, Optional, Dict
from .answer_extraction import answers_are_equal


def compute_accuracy(
    predictions: List[Optional[str]],
    gold_answers: List[Optional[str]],
    tol: float = 1e-3,
) -> Dict[str, float]:
    """
    Compute exact-match accuracy between predictions and gold answers.

    Args:
        predictions  : list of model-predicted answer strings
        gold_answers : list of ground truth answer strings
        tol          : tolerance for numeric comparison

    Returns:
        dict with 'accuracy', 'correct', 'total'
    """
    assert len(predictions) == len(gold_answers), (
        f"Length mismatch: {len(predictions)} predictions vs {len(gold_answers)} gold answers"
    )

    correct = sum(
        answers_are_equal(pred, gold, tol)
        for pred, gold in zip(predictions, gold_answers)
    )
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "accuracy_pct": 100.0 * accuracy,
        "correct": correct,
        "total": total,
    }


def compute_per_category_accuracy(
    predictions: List[Optional[str]],
    gold_answers: List[Optional[str]],
    categories: List[str],
    tol: float = 1e-3,
) -> Dict[str, Dict]:
    """
    Compute accuracy broken down by category (for MATH dataset).

    Args:
        predictions  : model outputs
        gold_answers : ground truth
        categories   : list of category strings (e.g. "algebra", "geometry")
        tol          : numeric tolerance

    Returns:
        dict mapping category → accuracy metrics
    """
    from collections import defaultdict

    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)

    for pred, gold, cat in zip(predictions, gold_answers, categories):
        cat_total[cat] += 1
        if answers_are_equal(pred, gold, tol):
            cat_correct[cat] += 1

    results = {}
    for cat in cat_total:
        total = cat_total[cat]
        correct = cat_correct[cat]
        results[cat] = {
            "accuracy": correct / total if total > 0 else 0.0,
            "accuracy_pct": 100.0 * correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
        }

    return results
