import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import yaml
    from generate_synthetic_data import generate

    from aml_mule.train import train

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    generate(n_cust=1000, n_mule=80, n_trader=250, seed=2).to_csv(tmp_path / "data" / "data.csv", index=False)

    cfg = {
        "data": {"raw_path": "data/data.csv", "raw_num_cols": [
            "Account_Age_Days", "Transaction_Volume_USD", "Avg_Time_Between_Trans_Min",
            "Flags_Last_6M", "Avg_End_Day_Balance"]},
        "split": {"test_size": 0.3, "val_size_of_temp": 0.5, "random_state": 42},
        "model": {"n_estimators": 30, "max_depth": 4, "learning_rate": 0.1,
                  "primary_max_features": 0.5,
                  "dominant_features": ["Account_Age_Days", "Avg_Time_Between_Trans_Min",
                                        "Trans_Per_Hour", "Daily_Volume", "log_Daily_Volume"]},
        "artifacts": {"dir": "artifacts", "filename_prefix": "aml_mule_pipeline",
                      "reference_sample_size": 300},
    }
    with open("config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
    train("config.yaml")

    from fastapi.testclient import TestClient

    import api.main as api_main
    api_main.detector = api_main._load_detector()
    return TestClient(api_main.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_returns_probability_and_factors(client):
    payload = {"records": [{
        "Account_Age_Days": 25, "Transaction_Volume_USD": 42000,
        "Avg_Time_Between_Trans_Min": 9, "Flags_Last_6M": 2, "Avg_End_Day_Balance": 120,
    }]}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    pred = r.json()["predictions"][0]
    assert "is_mule" in pred and "probability" in pred and "top_factors" in pred
    assert 0.0 <= pred["probability"] <= 1.0


def test_drift_endpoint(client):
    payload = {"records": [{
        "Account_Age_Days": 1500, "Transaction_Volume_USD": 2000,
        "Avg_Time_Between_Trans_Min": 5000, "Flags_Last_6M": 0, "Avg_End_Day_Balance": 5000,
    }] * 20}
    r = client.post("/monitor/drift", json=payload)
    assert r.status_code == 200
    assert "drift_report" in r.json()
