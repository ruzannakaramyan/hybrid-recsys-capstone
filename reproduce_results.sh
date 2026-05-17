#!/bin/bash

# ==============================================================================
# One-Command Reproduction Script
# Hybrid RecSys: LLM-Augmented Recommendation Systems
# ==============================================================================
# This script evaluates all pre-trained models on the test sets to ensure
# 100% reproducibility of the metrics presented in the thesis.
# ==============================================================================

set -e

# Setup logging
LOGFILE="reproduction_log.txt"
echo "======================================================" > $LOGFILE
echo "🚀 HYBRID RECSYS RESULTS REPRODUCTION LOG 🚀" >> $LOGFILE
echo "Date: $(date)" >> $LOGFILE
echo "======================================================" >> $LOGFILE

# Determine python executable
if [ -d ".venv" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

DATASETS=("industrial_and_scientific" "video_games" "cell_phones_and_accessories")

# Mapping datasets to their respective SASRec checkpoints
get_sasrec_ckpt() {
    if [ "$1" == "industrial_and_scientific" ]; then
        echo "src/sasrec_industrial_llm_simple.pth"
    elif [ "$1" == "video_games" ]; then
        echo "src/sasrec_video_games_llm_v3_correct.pth"
    elif [ "$1" == "cell_phones_and_accessories" ]; then
        echo "src/sasrec_cell_phones_llm_simple.pth"
    else
        echo ""
    fi
}

echo "Starting one-command reproduction sequence..." | tee -a $LOGFILE

for DATASET in "${DATASETS[@]}"; do
    echo "" | tee -a $LOGFILE
    echo "======================================================" | tee -a $LOGFILE
    echo "📊 DATASET: $DATASET" | tee -a $LOGFILE
    echo "======================================================" | tee -a $LOGFILE
    
    # ---------------------------------------------------------
    # 1. TopPop (Global Popularity Baseline)
    # ---------------------------------------------------------
    echo "[1/5] Evaluating TopPop..." | tee -a $LOGFILE
    $PYTHON src/evaluate_baselines.py --model_type toppop --dataset $DATASET --split test >> $LOGFILE 2>&1 || echo "Failed TopPop for $DATASET" | tee -a $LOGFILE
    
    # ---------------------------------------------------------
    # 2. BPR (Bayesian Personalized Ranking)
    # ---------------------------------------------------------
    BPR_CKPT="src/bpr_${DATASET}_best.pth"
    if [ -f "$BPR_CKPT" ]; then
        echo "[2/5] Evaluating BPR..." | tee -a $LOGFILE
        $PYTHON src/evaluate_baselines.py --model_type bpr --dataset $DATASET --split test --checkpoint $BPR_CKPT >> $LOGFILE 2>&1 || echo "Failed BPR for $DATASET" | tee -a $LOGFILE
    else
        echo "[2/5] Skipping BPR... Checkpoint not found: $BPR_CKPT" | tee -a $LOGFILE
    fi

    # ---------------------------------------------------------
    # 3. XGBoost
    # ---------------------------------------------------------
    XGB_CKPT="src/xgboost_pure_${DATASET}_best.json"
    if [ -f "$XGB_CKPT" ]; then
        echo "[3/5] Evaluating XGBoost..." | tee -a $LOGFILE
        $PYTHON src/evaluate_xgboost.py --dataset $DATASET --split test >> $LOGFILE 2>&1 || echo "Failed XGBoost for $DATASET" | tee -a $LOGFILE
    else
        echo "[3/5] Skipping XGBoost... Checkpoint not found: $XGB_CKPT" | tee -a $LOGFILE
    fi

    # ---------------------------------------------------------
    # 4. SASRec (LLM-Initialized)
    # ---------------------------------------------------------
    SASREC_CKPT=$(get_sasrec_ckpt $DATASET)
    if [ -f "$SASREC_CKPT" ]; then
        echo "[4/5] Evaluating SASRec + LLM Embeddings..." | tee -a $LOGFILE
        $PYTHON src/evaluate_sasrec.py --dataset $DATASET --split test --use_llm_embeddings --checkpoint $SASREC_CKPT >> $LOGFILE 2>&1 || echo "Failed SASRec for $DATASET" | tee -a $LOGFILE
    else
        echo "[4/5] Skipping SASRec... Checkpoint not found: $SASREC_CKPT" | tee -a $LOGFILE
    fi

    # ---------------------------------------------------------
    # 5. Generative Rerankers (Conditional)
    # ---------------------------------------------------------
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "[5/5] Skipping LLM Rerankers... OPENAI_API_KEY environment variable is not set." | tee -a $LOGFILE
    else
        echo "[5/5] Evaluating LLM Rerankers (2000 samples for Phase 3, 250 samples for Phase 4)..." | tee -a $LOGFILE
        # Ensure we have the base model to rerank
        if [ -f "$SASREC_CKPT" ]; then
            echo "   -> Running API Listwise Reranker..." | tee -a $LOGFILE
            $PYTHON src/llm_api_reranker.py --dataset $DATASET --base_checkpoint $SASREC_CKPT --use_llm_embeddings --max_samples 2000 >> $LOGFILE 2>&1 || echo "Failed API Reranker for $DATASET" | tee -a $LOGFILE
            
            echo "   -> Running Profile-Augmented Reranker..." | tee -a $LOGFILE
            $PYTHON src/llm_profile_reranker.py --dataset $DATASET --base_checkpoint $SASREC_CKPT --use_llm_embeddings --max_samples 250 >> $LOGFILE 2>&1 || echo "Failed Profile Reranker for $DATASET" | tee -a $LOGFILE
        else
            echo "   -> Skipping rerankers. Base SASRec model not found." | tee -a $LOGFILE
        fi
    fi
done

echo "" | tee -a $LOGFILE
echo "✅ Reproduction complete! Full metrics have been saved to $LOGFILE"
