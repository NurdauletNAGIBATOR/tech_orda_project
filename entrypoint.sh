#!/bin/sh
set -e

# demo convenience: if no trained artifact exists yet and no real data.csv
# was mounted either, generate synthetic data and train on it so the
# container is runnable out of the box. Mount a real data/data.csv to use
# real data instead.
if [ ! -f artifacts/aml_mule_pipeline_latest.joblib ]; then
    if [ ! -f data/data.csv ]; then
        echo "no data/data.csv found -- generating synthetic demo data"
        python scripts/generate_synthetic_data.py --out data/data.csv
    fi
    echo "no trained artifact found -- running training pipeline"
    python -m aml_mule.train --config config/config.yaml
fi

exec uvicorn api.main:app --host 0.0.0.0 --port 8000
