import joblib
import numpy as np
import pandas as pd

from aml_mule.features import add_missing_flags, engineer_features


class MuleDetector:
    def __init__(self, artifact_path: str):
        self.artifact = joblib.load(artifact_path)

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        raw_num_cols = self.artifact["raw_num_cols"]
        df = add_missing_flags(df, raw_num_cols)
        scaled = self.artifact["scaler"].transform(df[raw_num_cols])
        imputed = self.artifact["imputer"].transform(scaled)
        df = df.copy()
        df[raw_num_cols] = self.artifact["scaler"].inverse_transform(imputed)
        df = engineer_features(df)
        return df

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        prepared = self._prepare(df)
        feature_cols = self.artifact["feature_cols"]
        remaining_feats = self.artifact["remaining_feats"]
        proba_primary = self.artifact["primary_model"].predict_proba(prepared[feature_cols])[:, 1]
        proba_fallback = self.artifact["fallback_model"].predict_proba(prepared[remaining_feats])[:, 1]
        return np.maximum(proba_primary, proba_fallback)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(df)
        return (proba >= self.artifact["threshold"]).astype(int)
