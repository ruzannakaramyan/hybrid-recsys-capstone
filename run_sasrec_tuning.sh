#!/bin/bash
DATASET="industrial_and_scientific"

echo "========================================="
echo "Starting SASRec Hyperparameter Tuning"
echo "Dataset: $DATASET"
echo "========================================="

# Config 1: Baseline Architecture + Convergence Epochs
echo "--- Testing Config 1: dim=64, len=25, dropout=0.2, epochs=50 ---"
CKPT_1="src/sasrec_${DATASET}_c1.pth"
./.venv/bin/python src/train.py --dataset $DATASET --hidden_dim 64 --max_seq_len 25 --dropout 0.2 --epochs 50 --checkpoint $CKPT_1
echo "Evaluating Config 1..."
./.venv/bin/python src/evaluate.py --dataset $DATASET --hidden_dim 64 --max_seq_len 25 --dropout 0.2 --checkpoint $CKPT_1 > tuning_c1_results.txt
cat tuning_c1_results.txt | grep -E 'Hit@10|NDCG@10'

echo "========================================="

# Config 2: High Capacity Architecture
echo "--- Testing Config 2: dim=128, len=50, dropout=0.3, epochs=50 ---"
CKPT_2="src/sasrec_${DATASET}_c2.pth"
./.venv/bin/python src/train.py --dataset $DATASET --hidden_dim 128 --max_seq_len 50 --dropout 0.3 --epochs 50 --checkpoint $CKPT_2
echo "Evaluating Config 2..."
./.venv/bin/python src/evaluate.py --dataset $DATASET --hidden_dim 128 --max_seq_len 50 --dropout 0.3 --checkpoint $CKPT_2 > tuning_c2_results.txt
cat tuning_c2_results.txt | grep -E 'Hit@10|NDCG@10'

echo "========================================="
echo "Tuning Sweep Complete."
