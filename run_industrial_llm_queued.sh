#!/bin/bash
# Queue Industrial LLM to run after Video Games v2 completes
# Uses improved architecture (v2)

cd /Users/rkaramyan/hybrid_recsys_capstone

echo "Waiting for Video Games LLM v2 to complete..."
echo "Monitoring: src/sasrec_video_games_llm_v2.pth"

# Wait for Video Games v2 training to finish (check every 5 minutes)
while pgrep -f "sasrec_video_games_llm_v2.pth" > /dev/null; do
    echo "$(date): Video Games v2 still running..."
    sleep 300  # Check every 5 minutes
done

echo ""
echo "✅ Video Games v2 complete! Starting Industrial LLM..."
echo "$(date)"
echo ""

# Run Industrial with improved v2 settings
PYTHONUNBUFFERED=1 .venv/bin/python -u src/sasrec_train.py \
    --dataset industrial_and_scientific \
    --hidden_dim 64 \
    --num_heads 4 \
    --num_layers 2 \
    --dropout 0.3 \
    --max_seq_len 25 \
    --learning_rate 0.001 \
    --epochs 50 \
    --use_llm_embeddings \
    --freeze_emb_epochs 10 \
    --llm_lr_factor 0.5 \
    --checkpoint src/sasrec_industrial_llm_v2.pth \
    2>&1 | tee train_industrial_llm_v2.log

echo ""
echo "✅ Industrial LLM training complete!"
echo "$(date)"
