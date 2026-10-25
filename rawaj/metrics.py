"""Decision metrics for a mule-risk score.

You write this file. Keep the signatures exactly as they are: the grader calls them by name.

Rules for every function:
- y_true and scores are lists or 1-D arrays. Raise ValueError if the lengths differ, the
  input is empty, a label is not 0 or 1, or a score is NaN.
- A case is flagged when score >= threshold.
- Precision is 0.0 when nothing is flagged; recall is 0.0 when there are no positives.
"""
from __future__ import annotations
import math
import numpy as np


def _validate(y_true, scores):
    y = np.asarray(y_true)
    s = np.asarray(scores)
    if y.ndim != 1 or s.ndim != 1:
        raise ValueError("y_true and scores must be 1-D")
    if len(y) != len(s):
        raise ValueError("y_true and scores must have the same length")
    if len(y) == 0:
        raise ValueError("inputs must not be empty")
    try:
        y = y.astype(float)
        s = s.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("inputs must be numeric") from exc
    if np.isnan(y).any() or not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("labels must be 0 or 1")
    if np.isnan(s).any():
        raise ValueError("scores must not contain NaN")
    return y.astype(int), s


def _counts(y, s, threshold):
    flagged = s >= threshold
    tp = int(np.sum(flagged & (y == 1)))
    fp = int(np.sum(flagged & (y == 0)))
    fn = int(np.sum(~flagged & (y == 1)))
    tn = int(np.sum(~flagged & (y == 0)))
    return tp, fp, tn, fn


def confusion_at(y_true, scores, threshold: float) -> dict:
    y, s = _validate(y_true, scores)
    tp, fp, tn, fn = _counts(y, s, threshold)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def precision_recall_at(y_true, scores, threshold: float) -> tuple[float, float]:
    y, s = _validate(y_true, scores)
    tp, fp, _tn, fn = _counts(y, s, threshold)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(precision), float(recall)


def precision_at_k(y_true, scores, k: int) -> float:
    y, s = _validate(y_true, scores)
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(int(k), len(y))
    order = np.argsort(-s, kind="stable")  # stable: ties keep input order
    return float(y[order[:k]].sum() / k)


def expected_cost(y_true, scores, threshold: float, *, cost_fp: float, cost_fn: float) -> float:
    y, s = _validate(y_true, scores)
    _tp, fp, _tn, fn = _counts(y, s, threshold)
    return float(cost_fp * fp + cost_fn * fn)


def best_threshold(y_true, scores, *, cost_fp: float, cost_fn: float) -> float:
    y, s = _validate(y_true, scores)
    candidates = [math.inf] + sorted(set(s.tolist()), reverse=True)  # highest first
    best_t, best_c = candidates[0], None
    for t in candidates:
        _tp, fp, _tn, fn = _counts(y, s, t)
        cost = cost_fp * fp + cost_fn * fn
        if best_c is None or cost < best_c:  # strict <  -> ties keep the higher threshold
            best_c, best_t = cost, t
    return float(best_t)


def threshold_for_precision(y_true, scores, min_precision: float):
    y, s = _validate(y_true, scores)
    for t in sorted(set(s.tolist())):  # lowest first
        tp, fp, _tn, _fn = _counts(y, s, t)
        flagged = tp + fp
        if flagged > 0 and tp / flagged >= min_precision:
            return float(t)
    return None