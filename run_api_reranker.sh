#!/bin/bash
# Run API-based LLM Reranker Evaluation

cd /Users/rkaramyan/hybrid_recsys_capstone

echo "=========================================="
echo "API LLM Reranker Evaluation (GPT-4o-mini)"
echo "=========================================="
echo ""

if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY environment variable is not set."
    echo "Please set your API key by running:"
    echo "export OPENAI_API_KEY='sk-your-key-here'"
    exit 1
fi

DATASET=${1:-"video_games"}
MAX_SAMPLES=${2:-"2000"}  

if [ "$DATASET" = "video_games" ]; then
    HIDDEN_DIM=64
    NUM_HEADS=2
    NUM_LAYERS=3
    DROPOUT=0.2
    MAX_SEQ_LEN=25
    CHECKPOINT="src/sasrec_video_games_llm_v3_correct.pth"
elif [ "$DATASET" = "industrial_and_scientific" ]; then
    HIDDEN_DIM=64
    NUM_HEADS=4
    NUM_LAYERS=2
    DROPOUT=0.3
    MAX_SEQ_LEN=25
    CHECKPOINT="src/sasrec_industrial_llm_simple.pth"
elif [ "$DATASET" = "cell_phones_and_accessories" ]; then
    HIDDEN_DIM=64
    NUM_HEADS=2
    NUM_LAYERS=2
    DROPOUT=0.2
    MAX_SEQ_LEN=25
    CHECKPOINT="src/sasrec_cell_phones_llm_simple.pth"
else
    echo "Unknown dataset: $DATASET"
    exit 1
fi

.venv/bin/python src/llm_api_reranker.py \
    --dataset $DATASET \
    --base_checkpoint $CHECKPOINT \
    --hidden_dim $HIDDEN_DIM \
    --num_heads $NUM_HEADS \
    --num_layers $NUM_LAYERS \
    --dropout $DROPOUT \
    --max_seq_len $MAX_SEQ_LEN \
    --use_llm_embeddings \
    --rerank_topk 30 \
    --max_samples $MAX_SAMPLES \
    2>&1 | tee "test_api_reranker_${DATASET}.log"

echo "Done! Results logged to test_api_reranker_${DATASET}.log"
