#!/bin/bash
set -e

WEIGHTS_PATH="./src/model/data/weights.pt"

echo "=== [1/3] Fetching Data & Setting Up Schemas ==="
python -m src.setup

echo "=== [2/3] Checking Model Weights ==="
if [ ! -f "$WEIGHTS_PATH" ]; then
    echo "Weights missing at $WEIGHTS_PATH. Training model..."
    python -m src.train
else
    echo "Found weights at $WEIGHTS_PATH. Skipping training."
fi

echo "=== [3/3] Generating Embeddings & Populating Database ==="
python -m src.embed

echo "=== Initialization Complete! Database and weights are ready. ==="