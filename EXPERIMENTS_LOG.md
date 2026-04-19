# Hybrid Recommender System - Experiments Documentation

## Datasets Used

| Dataset | Users | Items | Notes |
|---------|-------|-------|-------|
| **Video Games** | 94,762 | 25,528 | Smallest dataset, used for initial testing |
| **Industrial & Scientific** | 50,985 | 25,755 | Medium dataset |
| **Cell Phones & Accessories** | 380,999 | 110,784 | **Largest dataset** - stratified eval sample: 50K users |

---

## 1. SASRec (Self-Attentive Sequential Recommendation)

### Hyperparameter Search Space
- **hidden_dim**: [64, 128]
- **num_heads**: [2, 4]
- **num_layers**: [2, 3]
- **dropout**: [0.1, 0.2, 0.3]
- **max_seq_len**: [25, 50]
- **learning_rate**: [0.0005, 0.001]

### Training Settings
- **Epochs**: 50 (with early stopping, patience=5)
- **Evaluation**: Every 1 epoch
- **Batch size**: 128
- **Device**: CPU (MPS unstable for transformers)
- **Cell Phones**: Stratified evaluation on 50K users (not full 381K)

---

### 1.1 Video Games Dataset (6 Trials)

| Trial | hidden_dim | num_heads | num_layers | dropout | max_seq_len | lr | Hit@10 |
|-------|-----------|-----------|------------|---------|-------------|-------|--------|
| 1 | 64 | 2 | 2 | 0.2 | 50 | 0.001 | **0.1028** |
| 2 | 64 | 2 | 3 | 0.2 | 25 | 0.0005 | **0.1032** |
| 3 | 128 | 4 | 3 | 0.2 | 50 | 0.0005 | 0.0953 |
| 4 | 64 | 4 | 3 | 0.1 | 50 | 0.001 | **0.1029** |
| 5 | 64 | 4 | 2 | 0.1 | 25 | 0.001 | 0.1021 |
| 6 | 128 | 2 | 2 | 0.3 | 25 | 0.001 | 0.1010 |

**Best Config:** hidden_dim=64, num_heads=2, num_layers=3, dropout=0.2, max_seq_len=25, lr=0.0005  
**Best Hit@10:** 0.1032

---

### 1.2 Industrial & Scientific Dataset (6 Trials)

| Trial | hidden_dim | num_heads | num_layers | dropout | max_seq_len | lr | Hit@10 |
|-------|-----------|-----------|------------|---------|-------------|-------|--------|
| 1 | 64 | 2 | 2 | 0.3 | 25 | 0.001 | 0.0509 |
| 2 | 128 | 4 | 3 | 0.1 | 50 | 0.0005 | 0.0406 |
| 3 | 64 | 4 | 2 | 0.3 | 25 | 0.001 | **0.0517** |
| 4 | 128 | 4 | 3 | 0.2 | 50 | 0.001 | 0.0475 |
| 5 | 128 | 2 | 3 | 0.2 | 50 | 0.0005 | 0.0392 |
| 6 | 64 | 4 | 2 | 0.3 | 50 | 0.001 | 0.0506 |

**Best Config:** hidden_dim=64, num_heads=4, num_layers=2, dropout=0.3, max_seq_len=25, lr=0.001  
**Best Hit@10:** 0.0517

---

### 1.3 Cell Phones & Accessories Dataset (6 Trials)

| Trial | hidden_dim | num_heads | num_layers | dropout | max_seq_len | lr | Hit@10 |
|-------|-----------|-----------|------------|---------|-------------|-------|--------|
| 1 | 64 | 2 | 2 | 0.2 | 25 | 0.0005 | **0.0475** |
| 2 | 64 | 2 | 3 | 0.2 | 50 | 0.001 | 0.0402 |
| 3 | 64 | 4 | 2 | 0.2 | 50 | 0.001 | 0.0410 |
| 4 | 64 | 4 | 2 | 0.3 | 25 | 0.0005 | 0.0447 |
| 5 | 64 | 2 | 2 | 0.3 | 25 | 0.001 | 0.0386 |
| 6 | 64 | 4 | 2 | 0.3 | 50 | 0.001 | 0.0395 |

**Best Config:** hidden_dim=64, num_heads=2, num_layers=2, dropout=0.2, max_seq_len=25, lr=0.0005  
**Best Hit@10:** **0.0475**

---

## 2. BPR-MF (Bayesian Personalized Ranking - Matrix Factorization)

### Hyperparameter Search Space
- **hidden_dim**: [32, 64, 128]
- **learning_rate**: [0.0005, 0.001]
- **weight_decay**: [1e-5, 1e-4, 1e-3]

### Training Settings
- **Epochs**: 50 (with early stopping, patience=5)
- **Evaluation**: Every 1 epoch
- **Device**: MPS (Apple Silicon GPU)

---

### 2.1 Video Games Dataset (6 Trials)

| Trial | hidden_dim | lr | weight_decay | Hit@10 |
|-------|-----------|-------|--------------|--------|
| 1 | 32 | 0.0005 | 0.0001 | 0.0006 |
| 2 | 64 | 0.0005 | 0.0001 | 0.0007 |
| 3 | 32 | 0.001 | 0.001 | 0.0043 |
| 4 | 128 | 0.0005 | 1e-05 | **0.0331** |
| 5 | 32 | 0.0005 | 1e-05 | 0.0317 |
| 6 | 32 | 0.0005 | 0.001 | 0.0025 |

**Best Config:** hidden_dim=128, lr=0.0005, weight_decay=1e-5  
**Best Hit@10:** 0.0331

---

### 2.2 Industrial & Scientific Dataset (6 Trials)

| Trial | hidden_dim | lr | weight_decay | Hit@10 |
|-------|-----------|-------|--------------|--------|
| 1 | 64 | 0.0005 | 1e-05 | **0.0274** |
| 2 | 32 | 0.001 | 1e-05 | 0.0268 |
| 3 | 128 | 0.0005 | 0.0001 | 0.0024 |
| 4 | 32 | 0.0005 | 1e-05 | 0.0268 |
| 5 | 32 | 0.001 | 0.0001 | 0.0007 |
| 6 | 64 | 0.001 | 1e-05 | 0.0267 |

**Best Config:** hidden_dim=64, lr=0.0005, weight_decay=1e-5  
**Best Hit@10:** 0.0274

---

### 2.3 Cell Phones & Accessories Dataset (6 Trials)

| Trial | hidden_dim | lr | weight_decay | Hit@10 |
|-------|-----------|-------|--------------|--------|
| 1 | 64 | 0.0005 | 1e-05 | 0.0003 |
| 2 | 128 | 0.001 | 1e-05 | 0.0002 |
| 3 | 128 | 0.0005 | 0.0001 | 0.0 |
| 4 | 64 | 0.0005 | 0.001 | 0.0004 |
| 5 | 128 | 0.0005 | 0.001 | **0.0007** |
| 6 | 64 | 0.001 | 0.0001 | 0.0004 |

**Best Config:** hidden_dim=128, lr=0.0005, weight_decay=0.001  
**Best Hit@10:** 0.0007

**Note:** BPR performed poorly on this dataset - loss stuck at 0.6931 (no learning).

---

## 3. XGBoost (Pure Content-Based)

### Hyperparameter Search Space (6 trials)
- **max_depth**: [4, 6, 8]
- **learning_rate**: [0.1]
- **n_estimators**: [100, 300]

### Training Settings
- **Metric**: AUC (not Hit@10/NDCG - different evaluation paradigm)
- **Features**: Item metadata (not sequential)

---

### 3.1 Video Games Dataset (6 Trials)

| Trial | max_depth | lr | n_estimators | AUC |
|-------|-----------|-------|--------------|-----|
| 1 | 4 | 0.1 | 100 | 0.9665 |
| 2 | 4 | 0.1 | 300 | 0.9668 |
| 3 | 6 | 0.1 | 100 | 0.9668 |
| 4 | 6 | 0.1 | 300 | 0.9671 |
| 5 | 8 | 0.1 | 100 | 0.9670 |
| 6 | 8 | 0.1 | 300 | **0.9672** |

**Best Config:** max_depth=8, lr=0.1, n_estimators=300  
**Best AUC:** 0.9672  
**Ranking Evaluation (Best Model):** Hit@10: 0.0248, NDCG@10: 0.0127

---

### 3.2 Industrial & Scientific Dataset (6 Trials)

| Trial | max_depth | lr | n_estimators | AUC |
|-------|-----------|-------|--------------|-----|
| 1 | 4 | 0.1 | 100 | 0.9843 |
| 2 | 4 | 0.1 | 300 | 0.9846 |
| 3 | 6 | 0.1 | 100 | 0.9845 |
| 4 | 6 | 0.1 | 300 | 0.9846 |
| 5 | 8 | 0.1 | 100 | **0.9846** |
| 6 | 8 | 0.1 | 300 | 0.9845 |

**Best Config:** max_depth=8, lr=0.1, n_estimators=100  
**Best AUC:** 0.9846  
**Ranking Evaluation (Best Model):** Hit@10: 0.0177, NDCG@10: 0.0097

---

### 3.3 Cell Phones & Accessories Dataset (6 Trials)

| Trial | max_depth | lr | n_estimators | AUC |
|-------|-----------|-------|--------------|-----|
| 1 | 4 | 0.1 | 100 | 0.9833 |
| 2 | 4 | 0.1 | 300 | 0.9835 |
| 3 | 6 | 0.1 | 100 | 0.9835 |
| 4 | 6 | 0.1 | 300 | 0.9837 |
| 5 | 8 | 0.1 | 100 | 0.9836 |
| 6 | 8 | 0.1 | 300 | **0.9837** |

**Best Config:** max_depth=8, lr=0.1, n_estimators=300  
**Best AUC:** 0.9837  
**Ranking Evaluation:** In Progress (ETA ~40 hours)

**Note:** XGBoost evaluated on AUC (classification). Ranking metrics (Hit@10/NDCG) available for Video Games and Industrial datasets.

---

## Summary of Best Results

| Dataset | SASRec Best Hit@10 | BPR Best Hit@10 | XGBoost Best AUC |
|---------|-------------------|-----------------|------------------|
| Video Games | **0.1032** | 0.0331 | **0.9672** (Hit@10: 0.0248, NDCG@10: 0.0127) |
| Industrial & Scientific | **0.0517** | 0.0274 | **0.9846** (Hit@10: 0.0177, NDCG@10: 0.0097) |
| Cell Phones & Accessories | **0.0475** | 0.0007 | 0.9837 |

---

## Key Findings

1. **SASRec consistently outperforms BPR-MF** across all datasets
2. **BPR failed on Cell Phones dataset** - model did not learn (loss stuck at 0.6931)
3. **Smaller hidden_dim (64) generally better** for both models
4. **XGBoost AUC scores very high** but not comparable to ranking metrics

---

**Key totals (6 trials per model per dataset):**
- SASRec: 6 + 6 + 6 = **18 trials** ✓
- BPR: 6 + 6 + 6 = **18 trials** ✓
- XGBoost: 6 + 6 + 6 = **18 trials** ✓

**Total: 54 trials for fair scientific comparison**

---

## Files Generated

### Results CSVs
- `grid_search_video_games.csv` - SASRec results
- `grid_search_industrial_and_scientific.csv` - SASRec results
- `grid_search_cell_phones_and_accessories.csv` - SASRec results
- `grid_search_bpr_video_games.csv` - BPR results
- `grid_search_bpr_industrial_and_scientific.csv` - BPR results
- `grid_search_bpr_cell_phones_and_accessories.csv` - BPR results
- `data/xgboost_grid_results_video_games.csv` - XGBoost results
- `data/xgboost_grid_results_cell_phones_and_accessories.csv` - XGBoost results

### Log Files
- `grid_search_logs.txt` - SASRec all datasets
- `grid_search_bpr_*_logs.txt` - BPR per dataset
- `xgboost_grid_logs.txt` - XGBoost results (Industrial)
- `evaluate_xgboost_logs.txt` - XGBoost evaluation with Hit@10/NDCG

### Model Checkpoints
- `src/sasrec_*_best.pth` - Best SASRec models
- `src/bpr_*_best.pth` - Best BPR models
- `src/grid_search_*_[N].pth` - Grid search checkpoints

---

*Document generated: April 15, 2026*
