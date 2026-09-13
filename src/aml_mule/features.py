import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


def add_missing_flags(df: pd.DataFrame, raw_num_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in raw_num_cols:
        df[f"was_missing_{c}"] = df[c].isna().astype(int)
    return df


class Preprocessor:
    
    def __init__(self, raw_num_cols: list[str], n_neighbors: int = 5):
        self.raw_num_cols = raw_num_cols
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=n_neighbors)
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        scaled = self.scaler.fit_transform(df[self.raw_num_cols])
        self.imputer.fit(scaled)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fit() before transform()")
        df = df.copy()
        scaled = self.scaler.transform(df[self.raw_num_cols])
        imputed = self.imputer.transform(scaled)
        df[self.raw_num_cols] = self.scaler.inverse_transform(imputed)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Balance_to_Volume_Ratio"] = df["Avg_End_Day_Balance"] / (df["Transaction_Volume_USD"] + 1)
    df["Volume_to_Balance_Ratio"] = df["Transaction_Volume_USD"] / (df["Avg_End_Day_Balance"] + 1)
    df["Daily_Volume"] = df["Transaction_Volume_USD"] / (df["Account_Age_Days"] + 1)
    df["Trans_Per_Hour"] = 60 / (df["Avg_Time_Between_Trans_Min"] + 1)
    df["Flags_per_Day"] = df["Flags_Last_6M"] / (df["Account_Age_Days"] + 1)
    for c in ["Transaction_Volume_USD", "Avg_End_Day_Balance", "Daily_Volume"]:
        df[f"log_{c}"] = np.log1p(df[c])
    return df
