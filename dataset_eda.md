# Optimized Dataset EDA: Comparative Report

This report provides a detailed breakdown of the three Amazon datasets used in this project. Analyzing these descriptive factors is essential for understanding the performance delta between sequential models (SASRec) and tabular models (XGBoost).

## 1. Interaction Scale & Sparsity

The following table compares the physical dimensions and density of each dataset.

| Dataset | Total Interactions | Unique Users | Unique Items | Sparsity |
| :--- | :--- | :--- | :--- | :--- |
| **Industrial & Scientific** | 259,992 | 50,985 | 25,689 | **99.9801%** |
| **Video Games** | 530,300 | 94,762 | 25,482 | **99.9780%** |
| **Cell Phones** | 1,609,788 | 380,999 | 110,373 | **99.9961%** |

> [!NOTE]
> **Sparsity Insight**: All three datasets are extremely sparse (>99.9%). The **Video Games** dataset is the "densest," which correlates with why SASRec performs exceptionally well there—there are more relative interactions per item to learn from.

---

## 2. Ratings Class Balance

We analyzed the distribution of star ratings (1.0 to 5.0) to check for label imbalance.

### Side-by-Side Distribution (%)

| Rating | Industrial & Scientific | Video Games | Cell Phones |
| :---: | :---: | :---: | :---: |
| **1.0 ⭐** | 4.36% | 5.72% | 7.75% |
| **2.0 ⭐** | 2.88% | 4.36% | 4.86% |
| **3.0 ⭐** | 5.64% | 8.94% | 7.84% |
| **4.0 ⭐** | 14.23% | 17.26% | 13.99% |
| **5.0 ⭐** | **72.90%** | **63.72%** | **65.55%** |

> [!IMPORTANT]
> **Dominancy of 5-Star Ratings**: Across all three domains, roughly **65% to 73%** of all interactions are 5-star ratings. This indicates that users primarily interact with (and review) products they like, or that the Amazon ecosystem is heavily "positive-only." For a recommender system, this means predicting "Positive Interaction" is highly dependent on identifying 5-star potential.

---

## 3. Sequential Characteristics

Since we are optimizing **Sequential Recommenders (SASRec)**, the length of user histories is a defining factor.

| Dataset | Average Sequence Length | Max Sequence Length (Target) |
| :--- | :---: | :---: |
| **Industrial & Scientific** | 6.70 | 25 |
| **Video Games** | 8.87 | 25 |
| **Cell Phones** | 4.56 | 25 |

> [!WARNING]
> **Cold-Start Risk**: The **Cell Phones** dataset has the shortest average sequence length (4.56). This makes it the hardest dataset for SASRec, as there is less "contextual history" to learn the user's intent compared to Video Games (8.87).

---

## 4. Scientific Conclusion

1.  **Video Games is the "Goldilocks" Dataset**: It has the highest interaction density and the longest average sequences. This is where we expect the highest absolute Hit@10 scores.
2.  **Industrial is a "Micro-Recommender" Challenge**: With only 260k interactions, the model must be extremely efficient with item embeddings.
3.  **Cell Phones is the Big Data Challenge**: With 1.6M interactions but very short sequences, the model must rely more on global item popularity and cross-user trends than deep individual sequential patterns.
