
import argparse
import sys

import joblib
import pandas as pd

from aml_mule.features import add_missing_flags, engineer_features
from aml_mule.monitoring import compute_drift_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--warn", type=float, default=0.1)
    parser.add_argument("--critical", type=float, default=0.2)
    args = parser.parse_args()

    artifact = joblib.load(args.artifact)
    batch = pd.read_csv(args.batch)
    batch = add_missing_flags(batch, artifact["raw_num_cols"])
    batch = engineer_features(batch)

    report = compute_drift_report(
        artifact["reference_sample"], batch, artifact["feature_cols"], args.warn, args.critical)
    print(report.to_string(index=False))

    if (report["status"] == "critical").any():
        print("\nCRITICAL DRIFT DETECTED -- review/retrain recommended")
        sys.exit(1)


if __name__ == "__main__":
    main()
