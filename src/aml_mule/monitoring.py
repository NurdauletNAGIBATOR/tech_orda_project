"""Feature-drift monitoring via Population Stability Index (PSI).

PSI < 0.1  -- no meaningful shift
0.1-0.2    -- moderate shift, worth watching
>= 0.2     -- significant shift, model should be reviewed/retrained
"""

import numpy as np
import pandas as pd


def _psi_for_feature(reference: pd.Series, current: pd.Series, buckets: int = 10) -> float:
    reference = reference.dropna()
    current = current.dropna()
    if len(reference) == 0 or len(current) == 0:
        return float("nan")
    quantiles = np.linspace(0, 1, buckets + 1)
    breakpoints = np.unique(reference.quantile(quantiles).values)
    if len(breakpoints) < 3:
        return 0.0
    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-4, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-4, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame,
                          feature_cols: list[str], warn: float = 0.1, critical: float = 0.2) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        psi = _psi_for_feature(reference_df[col], current_df[col])
        status = "critical" if psi >= critical else ("warning" if psi >= warn else "ok")
        rows.append({"feature": col, "psi": psi, "status": status})
    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values("psi", ascending=False).reset_index(drop=True)
    return report
