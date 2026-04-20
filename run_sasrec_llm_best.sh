#!/bin/bash
# Run SASRec with LLM embeddings using BEST configs from baseline
# Fair comparison: same hyperparams, only initialization differs

# Change to repo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "SASRec + LLM - Best Configs Only"
echo "Using same hyperparams as baseline for fair comparison"
echo "=========================================="

# ===== VIDEO GAMES =====
# Best: hidden_dim=64, num_heads=2, num_layers=3, dropout=0.2, max_seq_len=25, lr=0.0005
echo ""
echo "🎮 VIDEO GAMES - LLM Enhanced"
echo "=========================================="
.venv/bin/python src/sasrec_train.py \
    --dataset video_games \
    --hidden_dim 64 \
    --num_heads 2 \
    --num_layers 3 \
    --dropout 0.2 \
    --max_seq_len 25 \
    --learning_rate 0.0005 \
    --epochs 50 \
    --use_llm_embeddings \
    --freeze_emb_epochs 5 \
    --checkpoint src/sasrec_video_games_llm.pth \
    2>&1 | tee train_video_games_llm.log

echo ""
echo "Evaluating LLM SASRec on test set..."
.venv/bin/python src/sasrec_evaluate.py \
    --dataset video_games \
    --split test \
    --hidden_dim 64 \
    --num_heads 2 \
    --num_layers 3 \
    --dropout 0.2 \
    --max_seq_len 25 \
    --use_llm_embeddings \
    --checkpoint src/sasrec_video_games_llm.pth \
    2>&1 | tee test_video_games_llm.txt

# ===== INDUSTRIAL & SCIENTIFIC =====
# Best: hidden_dim=64, num_heads=4, num_layers=2, dropout=0.3, max_seq_len=25, lr=0.001
echo ""
echo ""
echo "🔬 INDUSTRIAL & SCIENTIFIC - LLM Enhanced"
echo "=========================================="
.venv/bin/python src/sasrec_train.py \
    --dataset industrial_and_scientific \
    --hidden_dim 64 \
    --num_heads 4 \
    --num_layers 2 \
    --dropout 0.3 \
    --max_seq_len 25 \
    --learning_rate 0.001 \
    --epochs 50 \
    --use_llm_embeddings \
    --freeze_emb_epochs 5 \
    --checkpoint src/sasrec_industrial_llm.pth \
    2>&1 | tee train_industrial_llm.log

echo ""
echo "Evaluating LLM SASRec on test set..."
.venv/bin/python src/sasrec_evaluate.py \
    --dataset industrial_and_scientific \
    --split test \
    --hidden_dim 64 \
    --num_heads 4 \
    --num_layers 2 \
    --dropout 0.3 \
    --max_seq_len 25 \
    --use_llm_embeddings \
    --checkpoint src/sasrec_industrial_llm.pth \
    2>&1 | tee test_industrial_llm.txt

# ===== CELL PHONES & ACCESSORIES =====
# Best: hidden_dim=64, num_heads=2, num_layers=2, dropout=0.2, max_seq_len=25, lr=0.0005
echo ""
echo ""
echo "📱 CELL PHONES & ACCESSORIES - LLM Enhanced"
echo "=========================================="
.venv/bin/python src/sasrec_train.py \
    --dataset cell_phones_and_accessories \
    --hidden_dim 64 \
    --num_heads 2 \
    --num_layers 2 \
    --dropout 0.2 \
    --max_seq_len 25 \
    --learning_rate 0.0005 \
    --epochs 50 \
    --use_llm_embeddings \
    --freeze_emb_epochs 5 \
    --checkpoint src/sasrec_cell_phones_llm.pth \
    2>&1 | tee train_cell_phones_llm.log

echo ""
echo "Evaluating LLM SASRec on test set..."
.venv/bin/python src/sasrec_evaluate.py \
    --dataset cell_phones_and_accessories \
    --split test \
    --hidden_dim 64 \
    --num_heads 2 \
    --num_layers 2 \
    --dropout 0.2 \
    --max_seq_len 25 \
    --sample_users 50000 \
    --use_llm_embeddings \
    --checkpoint src/sasrec_cell_phones_llm.pth \
    2>&1 | tee test_cell_phones_llm.txt

echo ""
echo "=========================================="
echo "✅ All LLM SASRec training complete!"
echo "Compare with baseline results for scientific analysis"
echo "=========================================="
