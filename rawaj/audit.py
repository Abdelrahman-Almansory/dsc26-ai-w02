from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _missing_mask(s: pd.Series) -> pd.Series:
    na = s.isna()
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return na
    blank = s.map(lambda v: isinstance(v, str) and v.strip() == "")
    return na | blank.astype(bool)


def _sorted(values):
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=str)


def _parse_times(values: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values):
        return values
    strs = values.astype(str).str.strip()
    try:
        return pd.to_datetime(strs, format="ISO8601", errors="coerce")
    except (ValueError, TypeError):
        return pd.to_datetime(strs, format="ISO8601", errors="coerce", utc=True)


def _auc_flag(y, x) -> bool:
    if len(np.unique(y)) < 2:
        return False
    auc = roc_auc_score(y, x)
    return bool(auc >= 0.98 or auc <= 0.02)


def audit(df: pd.DataFrame, *, target: str, id_col: str, time_col: str) -> dict:
    for c in (target, id_col, time_col):
        if c not in df.columns:
            raise KeyError(c)

    n = len(df)
    masks = {c: _missing_mask(df[c]) for c in df.columns}
    missing = {c: (float(masks[c].mean()) if n else 0.0) for c in df.columns}

    ids = df[id_col][~masks[id_col]]
    vc = ids.value_counts()
    duplicate_ids = _sorted(vc[vc > 1].index.tolist())
    duplicate_rows = int(df.duplicated().sum())

    constant_columns = _sorted(
        c for c in df.columns
        if c != target and df[c][~masks[c]].nunique() <= 1
    )

    labelled = ~masks[target]
    unlabelled_rows = int((~labelled).sum())
    y = (pd.to_numeric(df[target][labelled], errors="coerce") == 1).astype(int).to_numpy()
    positive_rate = float(y.mean()) if len(y) else 0.0

    tvals = df[time_col][~masks[time_col]]
    if len(tvals):
        parsed = _parse_times(tvals)
        ok = parsed.notna()
        bad_timestamps = int((~ok).sum())
        good = parsed[ok]
    else:
        bad_timestamps, good = 0, tvals
    time_range = [good.min().isoformat(), good.max().isoformat()] if len(good) else [None, None]

    suspects = []
    if len(y) >= 20 and len(np.unique(y)) == 2:
        lab = df[labelled]
        for c in df.columns:
            if c in (target, id_col, time_col):
                continue
            s = lab[c]
            m = masks[c][labelled]
            # single distinct value, with missing counted as a value -> never a suspect
            if s.where(~m).nunique(dropna=False) <= 1:
                continue
            flagged = False
            # (a) value leak: numeric/bool, >= 20 non-missing values, both classes present
            if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
                nm = ~m.to_numpy()
                if nm.sum() >= 20:
                    vals = pd.to_numeric(s, errors="coerce").astype("float64").to_numpy()
                    flagged = _auc_flag(y[nm], vals[nm])
            # (b) missingness leak: works for every column type
            if not flagged:
                flagged = _auc_flag(y, m.to_numpy().astype(int))
            if flagged:
                suspects.append(c)

    return {
        "n_rows": int(n),
        "missing": missing,
        "duplicate_ids": duplicate_ids,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "unlabelled_rows": unlabelled_rows,
        "positive_rate": positive_rate,
        "bad_timestamps": bad_timestamps,
        "time_range": time_range,
        "leakage_suspects": _sorted(suspects),
    }