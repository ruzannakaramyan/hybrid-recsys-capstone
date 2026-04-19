#!/bin/bash
datasets=("video_games" "cell_phones_and_accessories")

echo "========================================="
echo "Running Optimal Config (dim=64, len=25, drop=0.2, epochs=50)"
echo "========================================="

for DATASET in "${datasets[@]}"; do
    echo "--- Training $DATASET ---"
    CKPT="src/sasrec_${DATASET}_optimized.pth"
    ./.venv/bin/python src/train.py --dataset $DATASET --hidden_dim 64 --max_seq_len 25 --dropout 0.2 --epochs 50 --checkpoint $CKPT
    
    echo "--- Evaluating $DATASET ---"
    ./.venv/bin/python src/evaluate.py --dataset $DATASET --hidden_dim 64 --max_seq_len 25 --dropout 0.2 --checkpoint $CKPT > ${DATASET}_optimized_results.txt
    cat ${DATASET}_optimized_results.txt | grep -E 'Hit@10|NDCG@10'
done

echo "========================================="
echo "All Optimized Baselines Complete!"
