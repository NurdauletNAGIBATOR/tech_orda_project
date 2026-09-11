import numpy as np
import pandas as pd

from aml_mule.monitoring import compute_drift_report


def test_psi_near_zero_for_identical_distribution():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 5000)})
    cur = pd.DataFrame({"x": rng.normal(0, 1, 5000)})
    report = compute_drift_report(ref, cur, ["x"])
    assert report.loc[0, "psi"] < 0.05
    assert report.loc[0, "status"] == "ok"


def test_psi_flags_a_real_shift():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 5000)})
    cur = pd.DataFrame({"x": rng.normal(3, 1, 5000)})
    report = compute_drift_report(ref, cur, ["x"])
    assert report.loc[0, "psi"] > 0.2
    assert report.loc[0, "status"] == "critical"
