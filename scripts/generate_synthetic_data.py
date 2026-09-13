
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(n_cust=38000, n_mule=2500, n_trader=9495, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def block(n, age_mu, age_sd, vol_mu, vol_sd, time_mu, time_sd, flag_p, bal_mu, bal_sd, prefix):
        age = np.clip(rng.normal(age_mu, age_sd, n), 1, 4000)
        vol = np.clip(rng.lognormal(np.log(vol_mu), vol_sd, n), 1, None)
        tmin = np.clip(rng.normal(time_mu, time_sd, n), 0.5, None)
        flags = rng.binomial(3, flag_p, n)
        bal = np.clip(rng.normal(bal_mu, bal_sd, n), 0, None)
        ids = [f"{prefix}_{i}" for i in rng.choice(200000, n, replace=False)]
        return pd.DataFrame({
            "Customer_ID": ids, "Account_Age_Days": age, "Transaction_Volume_USD": vol,
            "Avg_Time_Between_Trans_Min": tmin, "Flags_Last_6M": flags, "Avg_End_Day_Balance": bal,
        })

    cust = block(n_cust, 1550, 800, 2000, 0.6, 5000, 2500, 0.02, 5000, 3000, "CUST")
    mule = block(n_mule, 32, 25, 45000, 0.5, 8, 5, 0.75, 150, 200, "MULE")
    trader = block(n_trader, 1170, 700, 50000, 0.5, 22, 15, 0.25, 15000, 8000, "TRADER")

    data = pd.concat([cust, mule, trader], ignore_index=True)
    data["Is_Mule"] = data["Customer_ID"].str.startswith("MULE").astype(int)

    for col in ["Account_Age_Days", "Transaction_Volume_USD", "Avg_Time_Between_Trans_Min",
                "Flags_Last_6M", "Avg_End_Day_Balance"]:
        idx = rng.choice(data.index, size=int(0.01 * len(data)), replace=False)
        data.loc[idx, col] = np.nan

    bad_idx = rng.choice(data.index, size=480, replace=False)
    data.loc[bad_idx, "Customer_ID"] = np.nan
    data.loc[bad_idx, "Is_Mule"] = np.nan

    return data.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/data.csv")
    args = parser.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    generate().to_csv(args.out, index=False)
    print(f"written: {args.out}")
