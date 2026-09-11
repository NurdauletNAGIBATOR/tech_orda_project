import numpy as np
import pandas as pd

from aml_mule.features import Preprocessor, add_missing_flags, engineer_features

RAW_NUM_COLS = ["Account_Age_Days", "Transaction_Volume_USD",
                "Avg_Time_Between_Trans_Min", "Flags_Last_6M", "Avg_End_Day_Balance"]


def _toy_df(n=50, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Account_Age_Days": rng.uniform(1, 3000, n),
        "Transaction_Volume_USD": rng.uniform(100, 50000, n),
        "Avg_Time_Between_Trans_Min": rng.uniform(1, 9000, n),
        "Flags_Last_6M": rng.integers(0, 3, n).astype(float),
        "Avg_End_Day_Balance": rng.uniform(0, 20000, n),
    })
    df.loc[rng.choice(n, 5, replace=False), "Avg_End_Day_Balance"] = np.nan
    return df


def test_missing_flags_added_and_no_nan_leaks_into_flag():
    df = _toy_df()
    out = add_missing_flags(df, RAW_NUM_COLS)
    assert "was_missing_Avg_End_Day_Balance" in out.columns
    assert out["was_missing_Avg_End_Day_Balance"].sum() == df["Avg_End_Day_Balance"].isna().sum()


def test_preprocessor_fit_on_train_only_then_transform_val():
    train = _toy_df(seed=1)
    val = _toy_df(seed=2)
    prep = Preprocessor(RAW_NUM_COLS)
    train_out = prep.fit_transform(train)
    val_out = prep.transform(val)
    assert train_out[RAW_NUM_COLS].isna().sum().sum() == 0
    assert val_out[RAW_NUM_COLS].isna().sum().sum() == 0
    # transform() must not be usable before fit()
    fresh = Preprocessor(RAW_NUM_COLS)
    try:
        fresh.transform(val)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_engineer_features_creates_expected_columns():
    df = _toy_df().fillna(0)
    out = engineer_features(df)
    for col in ["Balance_to_Volume_Ratio", "Volume_to_Balance_Ratio", "Daily_Volume",
                "Trans_Per_Hour", "Flags_per_Day", "log_Transaction_Volume_USD"]:
        assert col in out.columns
        assert out[col].isna().sum() == 0
