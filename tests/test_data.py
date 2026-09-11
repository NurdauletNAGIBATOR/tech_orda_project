import numpy as np
import pandas as pd

from aml_mule.data import drop_duplicates, fix_labels


def test_rows_without_customer_id_are_dropped_not_defaulted():
    df = pd.DataFrame({
        "Customer_ID": ["MULE_1", "CUST_1", None],
        "Is_Mule": [np.nan, np.nan, np.nan],
    })
    out = fix_labels(df)
    assert len(out) == 2
    assert out["Is_Mule"].tolist() == [1, 0]


def test_customer_type_extracted_from_prefix():
    df = pd.DataFrame({"Customer_ID": ["TRADER_5", "CUST_9"], "Is_Mule": [np.nan, np.nan]})
    out = fix_labels(df)
    assert set(out["Customer_type"]) == {"TRADER", "CUST"}


def test_drop_duplicates_keeps_first_occurrence_only():
    df = pd.DataFrame({"Customer_ID": ["CUST_1", "CUST_1", "CUST_2"], "Is_Mule": [0, 0, 0]})
    out = drop_duplicates(df)
    assert len(out) == 2
