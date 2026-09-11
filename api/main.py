"""FastAPI service exposing the trained mule-detection pipeline.

GET  /health           -- liveness + whether a model is loaded
POST /predict          -- score a batch of transactions, with per-record top SHAP factors
POST /monitor/drift    -- compare a batch against the training reference sample (PSI)
"""
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from aml_mule.explain import explain_instance
from aml_mule.features import add_missing_flags, engineer_features
from aml_mule.inference import MuleDetector
from aml_mule.monitoring import compute_drift_report

ARTIFACT_PATH = Path("artifacts/aml_mule_pipeline_latest.joblib")

detector: MuleDetector | None = None


class Transaction(BaseModel):
    Account_Age_Days: float
    Transaction_Volume_USD: float
    Avg_Time_Between_Trans_Min: float
    Flags_Last_6M: float
    Avg_End_Day_Balance: float


class PredictRequest(BaseModel):
    records: list[Transaction]


class DriftRequest(BaseModel):
    records: list[Transaction]


def _load_detector() -> MuleDetector | None:
    if ARTIFACT_PATH.exists():
        return MuleDetector(str(ARTIFACT_PATH))
    return None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global detector
    detector = _load_detector()
    yield


app = FastAPI(title="AML Mule Detection API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": detector is not None}


@app.post("/predict")
def predict(req: PredictRequest):
    if detector is None:
        raise HTTPException(status_code=503, detail="model not loaded -- run training first")
    df = pd.DataFrame([r.model_dump() for r in req.records])
    proba = detector.predict_proba(df)
    pred = detector.predict(df)

    results = []
    for i in range(len(df)):
        top_factors = explain_instance(detector, df.iloc[[i]], top_n=3)
        results.append({
            "is_mule": bool(pred[i]),
            "probability": float(proba[i]),
            "top_factors": {k: float(v) for k, v in top_factors.items()},
        })
    return {"predictions": results}


@app.post("/monitor/drift")
def monitor_drift(req: DriftRequest):
    if detector is None:
        raise HTTPException(status_code=503, detail="model not loaded -- run training first")
    df = pd.DataFrame([r.model_dump() for r in req.records])
    df = add_missing_flags(df, detector.artifact["raw_num_cols"])
    df = engineer_features(df)
    report = compute_drift_report(
        detector.artifact["reference_sample"], df, detector.artifact["feature_cols"])
    return {"drift_report": report.to_dict(orient="records")}


STATIC_DIR = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
