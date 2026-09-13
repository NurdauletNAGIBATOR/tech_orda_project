import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture(scope="module")
def trained_artifact_path(tmp_path_factory):
    import yaml
    from generate_synthetic_data import generate

    from aml_mule.train import train

    workdir = tmp_path_factory.mktemp("mlops_test")
    data_dir = workdir / "data"
    data_dir.mkdir()
    data_path = data_dir / "data.csv"
    generate(n_cust=2000, n_mule=150, n_trader=500, seed=1).to_csv(data_path, index=False)

    cfg = {
        "data": {"raw_path": str(data_path), "raw_num_cols": [
            "Account_Age_Days", "Transaction_Volume_USD", "Avg_Time_Between_Trans_Min",
            "Flags_Last_6M", "Avg_End_Day_Balance"]},
        "split": {"test_size": 0.3, "val_size_of_temp": 0.5, "random_state": 42},
        "model": {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.1,
                  "primary_max_features": 0.5,
                  "dominant_features": ["Account_Age_Days", "Avg_Time_Between_Trans_Min",
                                        "Trans_Per_Hour", "Daily_Volume", "log_Daily_Volume"]},
        "artifacts": {"dir": str(workdir / "artifacts"), "filename_prefix": "test_pipeline",
                      "reference_sample_size": 500},
    }
    cfg_path = workdir / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    train(str(cfg_path))
    return workdir / "artifacts" / "test_pipeline_latest.joblib"


def test_artifact_is_created(trained_artifact_path):
    assert trained_artifact_path.exists()


def test_inference_on_saved_artifact(trained_artifact_path):
    import pandas as pd

    from aml_mule.inference import MuleDetector

    detector = MuleDetector(str(trained_artifact_path))
    sample = pd.DataFrame([
        {"Account_Age_Days": 20, "Transaction_Volume_USD": 40000,
         "Avg_Time_Between_Trans_Min": 8, "Flags_Last_6M": 2, "Avg_End_Day_Balance": 100},
        {"Account_Age_Days": 1500, "Transaction_Volume_USD": 2000,
         "Avg_Time_Between_Trans_Min": 5000, "Flags_Last_6M": 0, "Avg_End_Day_Balance": 5000},
    ])
    proba = detector.predict_proba(sample)
    pred = detector.predict(sample)
    assert proba.shape == (2,)
    assert set(pred.tolist()).issubset({0, 1})
    assert (proba >= 0).all() and (proba <= 1).all()
