import pandas as pd


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def fix_labels(data: pd.DataFrame) -> pd.DataFrame:
    data = data.dropna(subset=["Customer_ID"]).copy()
    data["Is_Mule"] = data["Customer_ID"].str.contains("MULE").astype(int)
    data["Customer_type"] = data["Customer_ID"].str.split("_").str[0]
    return data


def drop_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop_duplicates(subset=["Customer_ID"]).reset_index(drop=True)


def prepare(path: str) -> pd.DataFrame:
    data = load_raw(path)
    data = fix_labels(data)
    data = drop_duplicates(data)
    return data
