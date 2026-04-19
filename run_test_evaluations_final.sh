#!/bin/bash
# Run best model configs on TEST set for all datasets
# Uses grid search checkpoints - NO retraining needed

# Change to repo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "TEST SET EVALUATION - Best Models"
echo "Running from: $(pwd)"
echo "=========================================="
echo ""
echo "⚠️  Using grid search checkpoints (no retraining)"
echo "=========================================="

# ===== VIDEO GAMES =====
echo ""
echo "🎮 VIDEO GAMES DATASET"
echo "=========================================="

# SASRec Best: hidden_dim=64, num_heads=2, num_layers=3, dropout=0.2, max_seq_len=25, lr=0.0005
# Grid Search Trial 1 -> grid_search_video_games_1.pth (14.7MB, 3 layers)
echo "Evaluating SASRec (Best Config)..."
.venv/bin/python src/sasrec_evaluate.py \
    --dataset video_games \
    --split test \
    --hidden_dim 64 \
    --num_heads 2 \
    --num_layers 3 \
    --dropout 0.2 \
    --max_seq_len 25 \
    --checkpoint src/grid_search_video_games_1.pth \
    2>&1 | tee test_eval_video_games_sasrec.txt

# BPR Best: hidden_dim=128, lr=0.0005, weight_decay=1e-5
# Grid Search Trial 1 -> grid_search_bpr_video_games_1.pth (30MB, dim=128)
echo ""
echo "Evaluating BPR (Best Config)..."
.venv/bin/python src/bpr_evaluate_standalone.py \
    --dataset video_games \
    --split test \
    --hidden_dim 128 \
    --checkpoint src/grid_search_bpr_video_games_1.pth \
    2>&1 | tee test_eval_video_games_bpr.txt

# XGBoost Best: max_depth=8, n_estimators=300
echo ""
echo "Evaluating XGBoost (Best Config)..."
.venv/bin/python src/evaluate_xgboost.py \
    --dataset video_games \
    --split test \
    --sample_size 50000 \
    2>&1 | tee test_eval_video_games_xgboost.txt

# ===== INDUSTRIAL & SCIENTIFIC =====
echo ""
echo ""
echo "🔬 INDUSTRIAL & SCIENTIFIC DATASET"
echo "=========================================="

# SASRec Best: hidden_dim=64, num_heads=4, num_layers=2, dropout=0.3, max_seq_len=25, lr=0.001
# Grid Search Trial 0 -> grid_search_industrial_and_scientific_0.pth (6.8MB, 2 layers)
echo "Evaluating SASRec (Best Config)..."
.venv/bin/python src/sasrec_evaluate.py \
    --dataset industrial_and_scientific \
    --split test \
    --hidden_dim 64 \
    --num_heads 4 \
    --num_layers 2 \
    --dropout 0.3 \
    --max_seq_len 25 \
    --checkpoint src/grid_search_industrial_and_scientific_0.pth \
    2>&1 | tee test_eval_industrial_sasrec.txt

# BPR Best: hidden_dim=64, lr=0.0005, weight_decay=1e-5
# Grid Search Trial 0 -> grid_search_bpr_industrial_and_scientific_0.pth (19MB, dim=64)
echo ""
echo "Evaluating BPR (Best Config)..."
.venv/bin/python src/bpr_evaluate_standalone.py \
    --dataset industrial_and_scientific \
    --split test \
    --hidden_dim 64 \
    --checkpoint src/grid_search_bpr_industrial_and_scientific_0.pth \
    2>&1 | tee test_eval_industrial_bpr.txt

# XGBoost Best: max_depth=8, n_estimators=100
echo ""
echo "Evaluating XGBoost (Best Config)..."
.venv/bin/python src/evaluate_xgboost.py \
    --dataset industrial_and_scientific \
    --split test \
    --sample_size 50000 \
    2>&1 | tee test_eval_industrial_xgboost.txt

# ===== CELL PHONES & ACCESSORIES =====
echo ""
echo ""
echo "📱 CELL PHONES & ACCESSORIES DATASET"
echo "=========================================="

# SASRec Best: hidden_dim=64, num_heads=2, num_layers=2, dropout=0.2, max_seq_len=25, lr=0.0005
# Grid Search Trial 0 -> grid_search_cell_phones_and_accessories_0.pth (28MB, 2 layers)
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
    --checkpoint src/grid_search_cell_phones_and_accessories_0.pth \
    2>&1 | tee test_eval_cell_phones_sasrec.txt

# BPR Best: hidden_dim=128, lr=0.0005, weight_decay=0.001
# Grid Search Trial 0 -> grid_search_bpr_cell_phones_and_accessories_0.pth (125MB, dim=128)
echo ""
echo "Evaluating BPR (Best Config)..."
.venv/bin/python src/bpr_evaluate_standalone.py \
    --dataset cell_phones_and_accessories \
    --split test \
    --hidden_dim 128 \
    --checkpoint src/grid_search_bpr_cell_phones_and_accessories_0.pth \
    2>&1 | tee test_eval_cell_phones_bpr.txt

# XGBoost Best: max_depth=8, n_estimators=300
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
