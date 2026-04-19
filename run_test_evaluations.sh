#!/bin/bash
# Run best model configs on TEST set for all datasets
# This provides the final unbiased evaluation metrics

# Change to repo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "TEST SET EVALUATION - Best Models"
echo "Running from: $(pwd)"
echo "=========================================="

# ===== VIDEO GAMES =====
echo ""
echo "🎮 VIDEO GAMES DATASET"
echo "=========================================="

# SASRec Best: hidden_dim=64, num_heads=2, num_layers=3, dropout=0.2, max_seq_len=25, lr=0.0005
echo "Evaluating SASRec (Best Config)..."
.venv/bin/python src/sasrec_evaluate.py \
    --dataset video_games \
    --split test \
    --hidden_dim 64 \
    --num_heads 2 \
    --num_layers 3 \
    --dropout 0.2 \
    --max_seq_len 25 \
    --checkpoint src/sasrec_video_games.pth \
    2>&1 | tee test_eval_video_games_sasrec.txt

# BPR Best: hidden_dim=128, lr=0.0005, weight_decay=1e-5
echo ""
echo "Evaluating BPR (Best Config)..."
# Need to train with test_split flag to get test results
.venv/bin/python src/train_bpr.py \
    --dataset video_games \
    --hidden_dim 128 \
    --learning_rate 0.0005 \
    --weight_decay 1e-5 \
    --epochs 1 \
    --checkpoint src/bpr_video_games.pth \
    --test_split \
    2>&1 | tee test_eval_video_games_bpr.txt

# XGBoost Best: max_depth=8, n_estimators=300
echo ""
echo "Evaluating XGBoost (Best Config)..."
.venv/bin/python src/evaluate_xgboost.py \
    --dataset video_games \
    --split test \
    2>&1 | tee test_eval_video_games_xgboost.txt

# ===== INDUSTRIAL & SCIENTIFIC =====
echo ""
echo ""
echo "🔬 INDUSTRIAL & SCIENTIFIC DATASET"
echo "=========================================="

# SASRec Best: hidden_dim=64, num_heads=4, num_layers=2, dropout=0.3, max_seq_len=25, lr=0.001
echo "Evaluating SASRec (Best Config)..."
.venv/bin/python src/sasrec_evaluate.py \
    --dataset industrial_and_scientific \
    --split test \
    --hidden_dim 64 \
    --num_heads 4 \
    --num_layers 2 \
    --dropout 0.3 \
    --max_seq_len 25 \
    --checkpoint src/sasrec_industrial_and_scientific.pth \
    2>&1 | tee test_eval_industrial_sasrec.txt

# BPR Best: hidden_dim=64, lr=0.0005, weight_decay=1e-5
echo ""
echo "Evaluating BPR (Best Config)..."
.venv/bin/python src/train_bpr.py \
    --dataset industrial_and_scientific \
    --hidden_dim 64 \
    --learning_rate 0.0005 \
    --weight_decay 1e-5 \
    --epochs 1 \
    --checkpoint src/bpr_industrial_and_scientific_best.pth \
    --test_split \
    2>&1 | tee test_eval_industrial_bpr.txt

# XGBoost Best: max_depth=8, n_estimators=100
echo ""
echo "Evaluating XGBoost (Best Config)..."
.venv/bin/python src/evaluate_xgboost.py \
    --dataset industrial_and_scientific \
    --split test \
    2>&1 | tee test_eval_industrial_xgboost.txt

# ===== CELL PHONES & ACCESSORIES =====
echo ""
echo ""
echo "📱 CELL PHONES & ACCESSORIES DATASET"
echo "=========================================="

# SASRec Best: hidden_dim=64, num_heads=2, num_layers=2, dropout=0.2, max_seq_len=25, lr=0.0005
echo "Evaluating SASRec (Best Config)..."
.venv/bin/python src/sasrec_evaluate.py \
    --dataset cell_phones_and_accessories \
    --split test \
    --hidden_dim 64 \
    --num_heads 2 \
    --num_layers 2 \
    --dropout 0.2 \
    --max_seq_len 25 \
    --sample_users 50000 \
    --checkpoint src/sasrec_cell_phones_and_accessories.pth \
    2>&1 | tee test_eval_cell_phones_sasrec.txt

# BPR Best: hidden_dim=128, lr=0.0005, weight_decay=0.001
echo ""
echo "Evaluating BPR (Best Config)..."
.venv/bin/python src/train_bpr.py \
    --dataset cell_phones_and_accessories \
    --hidden_dim 128 \
    --learning_rate 0.0005 \
    --weight_decay 0.001 \
    --epochs 1 \
    --checkpoint src/bpr_cell_phones_and_accessories.pth \
    --test_split \
    2>&1 | tee test_eval_cell_phones_bpr.txt

# XGBoost Best: max_depth=8, n_estimators=300 (evaluated on 15K stratified sample)
echo ""
echo "Evaluating XGBoost (Best Config)..."
.venv/bin/python src/evaluate_xgboost.py \
    --dataset cell_phones_and_accessories \
    --split test \
    --sample_size 15000 \
    --use_stratified \
    2>&1 | tee test_eval_cell_phones_xgboost.txt

echo ""
echo "=========================================="
echo "✅ All test evaluations complete!"
echo "Results saved to test_eval_*.txt files"
echo "=========================================="
