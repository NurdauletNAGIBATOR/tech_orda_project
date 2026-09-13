import pandas as pd
import shap

from aml_mule.inference import MuleDetector


def _positive_class_shap(shap_values):
    if isinstance(shap_values, list):
        return shap_values[1]
    if hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


def explain_instance(detector: MuleDetector, row_df: pd.DataFrame, top_n: int = 5) -> pd.Series:
    prepared = detector._prepare(row_df)
    feature_cols = detector.artifact["feature_cols"]
    explainer = shap.TreeExplainer(detector.artifact["primary_model"])
    raw_shap = explainer.shap_values(prepared[feature_cols])
    vals = _positive_class_shap(raw_shap)
    explanation = pd.Series(vals[0], index=feature_cols).sort_values(key=abs, ascending=False)
    return explanation.head(top_n)
