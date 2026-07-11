#!/bin/sh
set -e

# Train model if not present
if [ ! -f /app/models/model_v1.pkl ]; then
    echo "No model found. Training initial model..."
    cd /app/training && python data_prep.py && python train.py
    cd /app
fi

echo "Starting API..."
exec uvicorn api.main:app --host 0.0.0.0 --port 8000
