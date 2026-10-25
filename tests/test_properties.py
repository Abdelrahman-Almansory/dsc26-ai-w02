"""Property-style tests: random inputs (seeded numpy, no extra dependencies)
checked against simple brute-force definitions or invariants."""

import math

import numpy as np
import pandas as pd
import pytest

from rawaj.audit import audit
from rawaj.metrics import (
    best_threshold,
    confusion_at,
    expected_cost,
    precision_at_k,
    precision_recall_at,
    threshold_for_precision,
)

SEEDS = range(200)


def rand_case(seed, max_n=30, n_levels=6):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, max_n))
    y = rng.integers(0, 2, n)
    s = rng.integers(0, n_levels, n).astype(float)  # few levels -> many ties
    return rng, y, s


def brute_counts(y, s, t):
    tp = sum(1 for a, b in zip(y, s) if b >= t and a == 1)
    fp = sum(1 for a, b in zip(y, s) if b >= t and a == 0)
    fn = sum(1 for a, b in zip(y, s) if b < t and a == 1)
    tn = sum(1 for a, b in zip(y, s) if b < t and a == 0)
    return tp, fp, tn, fn


# ---------------------------------------------------------------- metrics

@pytest.mark.parametrize("seed", SEEDS)
def test_confusion_invariants(seed):
    rng, y, s = rand_case(seed)
    t = float(rng.integers(-1, 8))
    c = confusion_at(y.tolist(), s.tolist(), t)
    assert (c["tp"], c["fp"], c["tn"], c["fn"]) == brute_counts(y, s, t)
    assert sum(c.values()) == len(y)
    assert c["tp"] + c["fn"] == y.sum()
    # lists and arrays give the same answer
    assert c == confusion_at(y, s, t)
    # extremes
    assert confusion_at(y, s, -math.inf)["fn"] == 0
    assert confusion_at(y, s, math.inf)["tp"] == 0


@pytest.mark.parametrize("seed", SEEDS)
def test_precision_recall_matches_confusion(seed):
    rng, y, s = rand_case(seed)
    t = float(rng.integers(-1, 8))
    tp, fp, tn, fn = brute_counts(y, s, t)
    p, r = precision_recall_at(y, s, t)
    assert p == (tp / (tp + fp) if tp + fp else 0.0)
    assert r == (tp / (tp + fn) if tp + fn else 0.0)
    assert 0.0 <= p <= 1.0 and 0.0 <= r <= 1.0


@pytest.mark.parametrize("seed", SEEDS)
def test_order_invariance(seed):
    rng, y, s = rand_case(seed)
    perm = rng.permutation(len(y))
    t = float(rng.integers(0, 6))
    assert confusion_at(y, s, t) == confusion_at(y[perm], s[perm], t)
    assert best_threshold(y, s, cost_fp=1.0, cost_fn=3.0) == best_threshold(
        y[perm], s[perm], cost_fp=1.0, cost_fn=3.0
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_precision_at_k_matches_stable_brute_force(seed):
    rng, y, s = rand_case(seed)
    k = int(rng.integers(1, len(y) + 5))
    ranked = sorted(range(len(y)), key=lambda i: -s[i])  # python sort is stable
    kk = min(k, len(y))
    expected = sum(y[i] for i in ranked[:kk]) / kk
    assert precision_at_k(y, s, k) == pytest.approx(expected)
    assert precision_at_k(y, s, len(y) + 10) == precision_at_k(y, s, len(y))


@pytest.mark.parametrize("seed", SEEDS)
def test_expected_cost_and_best_threshold(seed):
    rng, y, s = rand_case(seed)
    cfp, cfn = float(rng.integers(0, 5)), float(rng.integers(0, 5))
    cands = [math.inf] + sorted(set(s.tolist()), reverse=True)
    costs = {t: expected_cost(y, s, t, cost_fp=cfp, cost_fn=cfn) for t in cands}
    for t, c in costs.items():
        tp, fp, tn, fn = brute_counts(y, s, t)
        assert c == cfp * fp + cfn * fn
    best = best_threshold(y, s, cost_fp=cfp, cost_fn=cfn)
    assert best in costs
    m = min(costs.values())
    assert costs[best] == m
    # tie rule: highest threshold among all minimisers
    assert best == max(t for t, c in costs.items() if c == m)


@pytest.mark.parametrize("seed", SEEDS)
def test_threshold_for_precision(seed):
    rng, y, s = rand_case(seed)
    target = float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0]))

    def prec(t):
        tp, fp, _tn, _fn = brute_counts(y, s, t)
        return tp / (tp + fp) if tp + fp else 0.0

    ok = [t for t in sorted(set(s.tolist())) if prec(t) >= target and brute_counts(y, s, t)[0] + brute_counts(y, s, t)[1] > 0]
    got = threshold_for_precision(y, s, target)
    if ok:
        assert got == ok[0]  # the lowest qualifying threshold
    else:
        assert got is None


@pytest.mark.parametrize("seed", range(40))
def test_bad_inputs_always_raise(seed):
    rng, y, s = rand_case(seed)
    fns = [
        lambda a, b: confusion_at(a, b, 0.5),
        lambda a, b: precision_recall_at(a, b, 0.5),
        lambda a, b: precision_at_k(a, b, 1),
        lambda a, b: expected_cost(a, b, 0.5, cost_fp=1, cost_fn=1),
        lambda a, b: best_threshold(a, b, cost_fp=1, cost_fn=1),
        lambda a, b: threshold_for_precision(a, b, 0.5),
    ]
    bad_s = s.copy()
    bad_s[0] = np.nan
    bad_y = y.copy()
    bad_y[0] = 2
    for f in fns:
        with pytest.raises(ValueError):
            f(y[:-1], s) if len(y) > 1 else f([], [])
        with pytest.raises(ValueError):
            f([], [])
        with pytest.raises(ValueError):
            f(y, bad_s)
        with pytest.raises(ValueError):
            f(bad_y, s)


# ------------------------------------------------------------------ audit

def make_df(seed, n=None):
    rng = np.random.default_rng(seed)
    n = n or int(rng.integers(40, 120))
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, size=max(3, n // 5), replace=False)] = 1
    return rng, n, y, pd.DataFrame(
        {
            "id": [f"t{i}" for i in range(n)],
            "t": pd.date_range("2026-09-01", periods=n, freq="h").strftime("%Y-%m-%d %H:%M"),
            "y": y,
            "honest": rng.normal(size=n),
            "noise_text": rng.choice(list("abcde"), size=n),
        }
    )


@pytest.mark.parametrize("seed", range(60))
def test_audit_basic_invariants(seed):
    rng, n, y, df = make_df(seed)
    # random blanks of different kinds in a text column
    idx = rng.choice(n, size=5, replace=False)
    df.loc[idx[0], "noise_text"] = None
    df.loc[idx[1], "noise_text"] = ""
    df.loc[idx[2], "noise_text"] = "   "
    df.loc[idx[3], "noise_text"] = "\t\n"
    r = audit(df, target="y", id_col="id", time_col="t")
    assert r["n_rows"] == n
    assert r["missing"]["noise_text"] == pytest.approx(4 / n)
    assert all(0.0 <= v <= 1.0 for v in r["missing"].values())
    assert set(r["missing"]) == set(df.columns)
    assert r["duplicate_rows"] == 0 and r["duplicate_ids"] == []
    assert r["unlabelled_rows"] == 0
    assert r["positive_rate"] == pytest.approx(y.mean())
    lo, hi = r["time_range"]
    assert pd.Timestamp(lo) <= pd.Timestamp(hi)
    # honest features are not flagged
    assert "honest" not in r["leakage_suspects"]
    assert "noise_text" not in r["leakage_suspects"]


@pytest.mark.parametrize("seed", range(60))
def test_audit_duplicates_match_brute_force(seed):
    rng, n, y, df = make_df(seed)
    extra = df.iloc[rng.integers(0, n, size=int(rng.integers(1, 8)))]
    df2 = pd.concat([df, extra], ignore_index=True)
    r = audit(df2, target="y", id_col="id", time_col="t")
    assert r["duplicate_rows"] == int(df2.duplicated().sum())
    counts = df2["id"].value_counts()
    assert r["duplicate_ids"] == sorted(counts[counts > 1].index)
    assert r["n_rows"] == len(df2)


@pytest.mark.parametrize("seed", range(60))
def test_audit_leaks_are_flagged(seed):
    rng, n, y, df = make_df(seed)
    df["copy"] = y.astype(float)
    df["inverse"] = 1 - y
    df["flag_bool"] = y.astype(bool)
    df["only_pos"] = np.where(y == 1, "closed", "")
    df["blank_for_pos"] = np.where(y == 1, None, "x")
    df["const"] = "EGP"
    r = audit(df, target="y", id_col="id", time_col="t")
    for c in ("copy", "inverse", "flag_bool", "only_pos", "blank_for_pos"):
        assert c in r["leakage_suspects"], c
    assert "const" not in r["leakage_suspects"]
    assert "const" in r["constant_columns"]
    assert "y" not in r["constant_columns"]
    assert r["leakage_suspects"] == sorted(r["leakage_suspects"])


@pytest.mark.parametrize("seed", range(40))
def test_audit_row_order_does_not_change_answer(seed):
    rng, n, y, df = make_df(seed)
    df["copy"] = y.astype(float)
    shuffled = df.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)
    a = audit(df, target="y", id_col="id", time_col="t")
    b = audit(shuffled, target="y", id_col="id", time_col="t")
    assert a == b


@pytest.mark.parametrize("seed", range(40))
def test_audit_minimums_and_single_class(seed):
    rng, n, y, df = make_df(seed, n=19)  # fewer than 20 labelled rows
    df["copy"] = y.astype(float)
    assert audit(df, target="y", id_col="id", time_col="t")["leakage_suspects"] == []
    rng, n, y, df = make_df(seed, n=60)
    df["copy"] = y.astype(float)
    df["y"] = 0  # one class only
    assert audit(df, target="y", id_col="id", time_col="t")["leakage_suspects"] == []
    # fewer than 20 non-missing values -> value check does not apply
    rng, n, y, df = make_df(seed, n=60)
    col = np.full(n, np.nan)
    pos = np.flatnonzero(y == 1)[:5]
    col[pos] = 1.0
    df["sparse"] = col  # filled only for some positives: missingness AUC < 0.98
    r = audit(df, target="y", id_col="id", time_col="t")
    assert r["n_rows"] == 60


@pytest.mark.parametrize("seed", range(40))
def test_audit_timestamps_and_unlabelled(seed):
    rng, n, y, df = make_df(seed)
    y_f = df["y"].astype(float)
    un = rng.choice(n, size=4, replace=False)
    y_f.iloc[un[:2]] = np.nan
    df["y"] = y_f
    df.loc[un[2], "y"] = np.nan
    df.loc[un[3], "y"] = np.nan
    bad = rng.choice(n, size=3, replace=False)
    df.loc[bad[0], "t"] = "2026-02-30 10:00"
    df.loc[bad[1], "t"] = "not a date"
    df.loc[bad[2], "t"] = "  "
    r = audit(df, target="y", id_col="id", time_col="t")
    assert r["unlabelled_rows"] == 4
    assert r["bad_timestamps"] == 2  # blank one is missing, not bad
    labelled = df["y"].dropna()
    assert r["positive_rate"] == pytest.approx((labelled == 1).mean())


@pytest.mark.parametrize("col", ["target", "id_col", "time_col"])
def test_audit_missing_required_column_raises_keyerror(col):
    _, _, _, df = make_df(0)
    kwargs = dict(target="y", id_col="id", time_col="t")
    kwargs[col] = "does_not_exist"
    with pytest.raises(KeyError):
        audit(df, **kwargs)