#!/bin/bash

# Configuration
VENV_PYTHON="./.venv/bin/python"

# Datasets to process
DATASETS=("video_games" "cell_phones_and_accessories")

for ds in "${DATASETS[@]}"; do
    echo "===================================================="
    echo "🚀 STARTING PIPELINE FOR: $ds"
    echo "===================================================="
    
    # 1. Data Preparation
    echo "[1/3] Preparing stratified training data..."
    $VENV_PYTHON src/prepare_xgboost_data.py --dataset "$ds"
    if [ $? -ne 0 ]; then echo "❌ Data prep failed for $ds"; exit 1; fi
    
    # 2. Grid Search (Tuning)
    echo "[2/3] Running hyperparameter grid search..."
    $VENV_PYTHON src/xgboost_grid_search.py --dataset "$ds"
    if [ $? -ne 0 ]; then echo "❌ Grid search failed for $ds"; exit 1; fi
    
    # 3. Scientific Evaluation
    echo "[3/3] Running full-catalog evaluation..."
    $VENV_PYTHON src/evaluate_xgboost.py --dataset "$ds"
    if [ $? -ne 0 ]; then echo "❌ Evaluation failed for $ds"; exit 1; fi
    
    echo "✅ FINISHED PIPELINE FOR: $ds"
    echo ""
done

echo "🎉 ALL XGBOOST BASELINES COMPLETED!"
