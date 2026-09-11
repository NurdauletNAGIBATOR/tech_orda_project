"""End-to-end training pipeline -- the scripted, reproducible version of the
research notebook. Produces a versioned artifact + a metrics json alongside it.

Usage:
    python -m aml_mule.train --config config/config.yaml
"""
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from aml_mule.config import load_config
from aml_mule.data import prepare
from aml_mule.features import Preprocessor, add_missing_flags, engineer_features


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # noqa: BLE001 -- git metadata is best-effort, never fatal
        return "unknown"


def build_datasets(cfg: dict):
    data = prepare(cfg["data"]["raw_path"])
    raw_num_cols = cfg["data"]["raw_num_cols"]
    split_cfg = cfg["split"]

    train_df, temp_df = train_test_split(
        data, test_size=split_cfg["test_size"], random_state=split_cfg["random_state"],
        stratify=data["Is_Mule"])
    val_df, test_df = train_test_split(
        temp_df, test_size=split_cfg["val_size_of_temp"], random_state=split_cfg["random_state"],
        stratify=temp_df["Is_Mule"])

    train_df = add_missing_flags(train_df, raw_num_cols)
    val_df = add_missing_flags(val_df, raw_num_cols)
    test_df = add_missing_flags(test_df, raw_num_cols)

    prep = Preprocessor(raw_num_cols)
    train_df[raw_num_cols] = prep.fit_transform(train_df)[raw_num_cols]
    val_df[raw_num_cols] = prep.transform(val_df)[raw_num_cols]
    test_df[raw_num_cols] = prep.transform(test_df)[raw_num_cols]

    train_df = engineer_features(train_df)
    val_df = engineer_features(val_df)
    test_df = engineer_features(test_df)

    drop_cols = ["Customer_ID", "Customer_type", "Is_Mule"]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    return prep, train_df, val_df, test_df, feature_cols


def best_threshold(y_true, proba, beta: float = 2.0) -> float:
    prec, rec, thr = precision_recall_curve(y_true, proba)
    fbeta = ((1 + beta**2) * prec * rec) / (beta**2 * prec + rec + 1e-9)
    return float(thr[np.argmax(fbeta[:-1])])


def train(cfg_path: str = "config/config.yaml") -> dict:
    cfg = load_config(cfg_path)
    prep, train_df, val_df, test_df, feature_cols = build_datasets(cfg)

    X_train, y_train = train_df[feature_cols], train_df["Is_Mule"]
    X_val, y_val = val_df[feature_cols], val_df["Is_Mule"]
    X_test, y_test = test_df[feature_cols], test_df["Is_Mule"]

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / max(pos, 1)
    mcfg = cfg["model"]
    rs = cfg["split"]["random_state"]

    candidates = {
        "logreg": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=mcfg["n_estimators"], class_weight="balanced", random_state=rs, n_jobs=-1),
        "xgboost": XGBClassifier(
            n_estimators=mcfg["n_estimators"], max_depth=mcfg["max_depth"],
            learning_rate=mcfg["learning_rate"], scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr", random_state=rs),
    }

    candidate_scores = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        candidate_scores[name] = {
            "pr_auc": float(average_precision_score(y_val, proba)),
            "roc_auc": float(roc_auc_score(y_val, proba)),
        }

    # defense-in-depth ensemble: a primary model regularized to not lean
    # entirely on the two dominant features, plus a fallback model that
    # never sees those features at all (see notebook section 11 for why).
    dominant = [c for c in mcfg["dominant_features"] if c in feature_cols]
    remaining_feats = [c for c in feature_cols if c not in dominant]

    primary = RandomForestClassifier(
        n_estimators=mcfg["n_estimators"], max_features=mcfg["primary_max_features"],
        class_weight="balanced", random_state=rs, n_jobs=-1)
    primary.fit(X_train[feature_cols], y_train)

    fallback = RandomForestClassifier(
        n_estimators=mcfg["n_estimators"], class_weight="balanced", random_state=rs, n_jobs=-1)
    fallback.fit(X_train[remaining_feats], y_train)

    proba_val = np.maximum(
        primary.predict_proba(X_val[feature_cols])[:, 1],
        fallback.predict_proba(X_val[remaining_feats])[:, 1],
    )
    threshold = best_threshold(y_val, proba_val)

    proba_test = np.maximum(
        primary.predict_proba(X_test[feature_cols])[:, 1],
        fallback.predict_proba(X_test[remaining_feats])[:, 1],
    )
    pred_test = (proba_test >= threshold).astype(int)

    report = classification_report(y_test, pred_test, output_dict=True)
    segment_block_rate = test_df.assign(pred=pred_test).groupby("Customer_type")["pred"].mean().to_dict()

    sample_n = min(cfg["artifacts"].get("reference_sample_size", 5000), len(train_df))
    reference_sample = train_df[feature_cols].sample(sample_n, random_state=rs).reset_index(drop=True)

    artifact = {
        "scaler": prep.scaler,
        "imputer": prep.imputer,
        "primary_model": primary,
        "fallback_model": fallback,
        "feature_cols": feature_cols,
        "remaining_feats": remaining_feats,
        "raw_num_cols": cfg["data"]["raw_num_cols"],
        "threshold": threshold,
        "reference_sample": reference_sample,
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "candidate_val_scores": candidate_scores,
            "test_report": report,
            "segment_block_rate": segment_block_rate,
        },
    }

    out_dir = Path(cfg["artifacts"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = cfg["artifacts"]["filename_prefix"]

    versioned_path = out_dir / f"{prefix}_{version}.joblib"
    latest_path = out_dir / f"{prefix}_latest.joblib"
    joblib.dump(artifact, versioned_path)
    joblib.dump(artifact, latest_path)

    with open(out_dir / f"{prefix}_{version}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(artifact["metadata"], f, indent=2, default=str)

    print(f"saved: {versioned_path}")
    print(f"saved: {latest_path}")
    print(json.dumps(artifact["metadata"], indent=2, default=str))
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    train(args.config)
