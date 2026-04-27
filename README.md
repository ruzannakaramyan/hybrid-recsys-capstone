# Hybrid RecSys: LLM-Augmented Recommendation Systems

This repository contains the graduate research project: **"Hybrid Recommendation Systems: Leveraging Generative LLM Reasoning and Semantic Embeddings for Sequential Prediction."**

The project is structured into four distinct research phases, moving from traditional collaborative filtering to state-of-the-art identity-aware generative reranking.

---

## 🛠️ Repository structure

- **`src/`**: Core modeling and training logic (SASrec, BPR, XGBoost, and LLM Rerankers).
- **`results/`**: Organized directory containing all final experimental logs, grid search CSVs, and performance metrics.
- **`data/`**: Processed interaction data and pre-computed S-BERT embeddings.
- **`notebooks/`**: EDA and visualization scratchpads.

---

## 🚀 Research Phases

### Phase 1: Traditional Baselines
Evaluates non-LLM models including TopPop, Bayesian Personalized Ranking (BPR), and Gradient Boosted Trees (XGBoost).
- **Run:** `./run_xgboost_pipeline.sh`
- **Result:** Found that ID-only models struggle with extreme item density (Cell Phones).

### Phase 2: LLM Embedding Initialization
Initializes the SASRec transformer model items using **Sentence-BERT (S-BERT)** embeddings generated from product metadata.
- **Run:** `./run_sasrec_llm_best.sh`
- **Impact:** Achieved a **12,000% Hit Rate improvement** over traditional BPR for large datasets.

### Phase 3: Generative Listwise Reranking
Implements a two-stage pipeline: SASRec retrieval followed by **GPT-4o-mini** listwise reranking with Chain-of-Thought (CoT).
- **Run:** `./run_api_reranker.sh <dataset>`
- **Prerequisite:** Set `EXPORT OPENAI_API_KEY='your-key'`

### Phase 4: Profile-Augmented Reasoning (SOTA)
An advanced reranking strategy where the LLM is forced to identify a **User Persona** before performing the ranking task.
- **Run:** `./run_profile_reranker.sh <dataset>`
- **Impact:** Achieved a **+61% NDCG boost** for technical Industrial & Scientific domains.

---

## 📦 Reproducibility Instructions

1. **Environment Setup:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Data Preparation:**
   Raw Amazon Review Data (2023) should be placed in the `data/` directory.
   Files required: `Video_Games.train.csv.gz`, `Industrial_and_Scientific.train.csv.gz`, etc.
   Run `src/generate_embeddings.py` to recreate LLM item vectors.

3. **Reproducing Measurements:**
   To verify the metrics in the `results/` folder, run these commands:

   - **TopPop:** `.venv/bin/python src/evaluate_baselines.py --model_type toppop --dataset <name> --split test`
   - **BPR:** `.venv/bin/python src/evaluate_baselines.py --model_type bpr --dataset <name> --split test --checkpoint src/bpr_<name>_best.pth`
   - **XGBoost:** `.venv/bin/python src/evaluate_xgboost.py --dataset <name> --split test`
   - **SASRec Baseline:** `.venv/bin/python src/evaluate_sasrec.py --dataset <name> --checkpoint src/sasrec_<name>_baseline_best.pth`
   - **SASRec + LLM:** `.venv/bin/python src/evaluate_sasrec.py --dataset <name> --checkpoint src/sasrec_<name>_llm_best.pth --use_llm_embeddings`

---

## 📖 Key Findings
Full metrics table available in `results/PHASE4_PROFILE/` and summarized in the final thesis report. 
Identity-aware reasoning (Phase 4) proved to be the most powerful LLM application for technical product discovery.
