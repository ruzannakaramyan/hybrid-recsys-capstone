# LLM-Enhanced SASRec Architecture

## Overview
Transformer-based sequential recommender with LLM semantic embeddings for items.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INPUT: User Interaction Sequence                    │
│                    [ItemID_1, ItemID_2, ItemID_3, ..., ItemID_n]            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ITEM EMBEDDING LAYER                              │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    LLM EMBEDDINGS (Frozen→Trainable)                │   │
│   │                                                                     │   │
│   │   Item Text: "The Legend of Zelda: Breath of the Wild"              │   │
│   │           ↓                                                       │   │
│   │   SentenceTransformer (all-MiniLM-L6-v2)                          │   │
│   │           ↓                                                       │   │
│   │   LLM Vector: 384-dim semantic embedding                          │   │
│   │           ↓                                                       │   │
│   │   ┌─────────────────┐    ┌─────────────────┐                       │   │
│   │   │   PROJECTION    │    │  SIMPLE LINEAR │                       │   │
│   │   │    384 → 64     │    │  384 → 64      │                       │   │
│   │   │  (Single Layer) │    │  + LayerNorm    │                       │   │
│   │   └─────────────────┘    └─────────────────┘                       │   │
│   │                                                                     │   │
│   │   Output: 64-dim item embedding (matches SASRec hidden_dim)         │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   Alternative: Random learned embeddings (vanilla SASRec, no LLM)             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      POSITIONAL EMBEDDING (Learned)                         │
│                    Position: [0, 1, 2, ..., max_seq_len]                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EMBEDDING DROPOUT (p=0.2)                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER ENCODER STACK (SASRec Core)                  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    Self-Attention Mechanism                         │   │
│   │                                                                     │   │
│   │   Input: Item embeddings + Position embeddings                    │   │
│   │                                                                     │   │
│   │   Multi-Head Attention (num_heads=2 or 4)                           │   │
│   │   ┌─────────────────┐    ┌─────────────────┐                       │   │
│   │   │   Head 1        │    │   Head 2        │                       │   │
│   │   │  (64/2=32 dim)  │    │  (64/2=32 dim)  │                       │   │
│   │   │  Captures       │    │  Captures       │                       │   │
│   │   │  category       │    │  temporal       │                       │   │
│   │   │  similarity     │    │  patterns       │                       │   │
│   │   └─────────────────┘    └─────────────────┘                       │   │
│   │           ↓                       ↓                               │   │
│   │           └───────────┬───────────┘                               │   │
│   │                   Concatenate                                     │   │
│   │                       ↓                                           │   │
│   │                   64-dim output                                   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │              Feed-Forward Network (Position-wise)                 │   │
│   │                                                                     │   │
│   │   64 → 256 → 64 (with ReLU, Dropout, LayerNorm)                   │   │
│   │                                                                     │   │
│   │   Stack: num_layers=2 or 3 transformer layers                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      OUTPUT: Next Item Prediction                             │
│                                                                             │
│   Final Hidden State ──→ Linear ──→ Logits over all items                   │
│                                                                             │
│   Loss: Binary Cross-Entropy (BCEWithLogitsLoss)                            │
│   Negative Sampling: Random negatives + Hard negatives                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Two-Phase Training Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: WARM-UP (Epochs 1-5)                            │
│                                                                             │
│   LLM Embeddings: FROZEN (requires_grad=False)                              │
│   Transformer Params: TRAINABLE                                             │
│   Learning Rate: 0.001 (high)                                               │
│                                                                             │
│   Goal: Learn how to USE LLM knowledge without changing it                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: FINE-TUNING (Epochs 6+)                         │
│                                                                             │
│   LLM Embeddings: UNFROZEN (requires_grad=True)                             │
│   Transformer Params: TRAINABLE                                           │
│   Learning Rate: 0.0005 (lower, llm_lr_factor=0.5)                         │
│                                                                             │
│   Goal: Adapt LLM embeddings to recommendation patterns                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Innovations

### 1. Simple Projection (vs Complex Multi-Layer)

**BEFORE (Failed):**
```
384 → 256 → 128 → 64 (3 layers, ReLU, Dropout)
Result: Overfitting, Test Hit@10 = 0.035-0.086
```

**AFTER (Success):**
```
384 → 64 (1 layer, LayerNorm only)
Result: Better generalization, Test Hit@10 = 0.047-0.092
```

**Why it works:**
- Preserves LLM semantic structure
- Less parameters = less overfitting
- Direct knowledge transfer

### 2. Semantic Knowledge Injection

**Vanilla SASRec:**
- Item 1: [random vector learned from scratch]
- Item 2: [random vector learned from scratch]
- No prior knowledge about item relationships

**LLM-Enhanced SASRec:**
- Item 1: "Zelda" → LLM → [0.2, -0.5, 0.8, ...] (knows it's "adventure game")
- Item 2: "Skyrim" → LLM → [0.3, -0.4, 0.9, ...] (knows it's "open-world RPG")
- Similar items have similar embeddings from the start!

### 3. Dataset-Specific Benefits

| Dataset | Sparsity | LLM Benefit |
|---------|----------|-------------|
| **Industrial** | Very sparse | Large improvement (+16%) - semantic similarity helps cold items |
| **Video Games** | Dense | Moderate improvement - sequential patterns already strong |
| **Cell Phones** | Very sparse | Pending - expected similar to Industrial |

## Technical Specifications

```python
# Model Configuration
hidden_dim = 64          # Embedding dimension
num_heads = 2 or 4       # Attention heads (4 for Industrial, 2 for Video Games)
num_layers = 2 or 3      # Transformer layers
max_seq_len = 25         # Maximum sequence length
dropout = 0.2 or 0.3     # Regularization

# LLM Configuration
model_name = "sentence-transformers/all-MiniLM-L6-v2"
llm_dim = 384            # Output dimension from LLM
projection = "linear"    # 384 → 64 simple linear projection

# Training Configuration
freeze_epochs = 5        # Phase 1: LLM frozen
lr_factor = 0.5          # Phase 2: LR reduced by half
batch_size = 128
optimizer = Adam
```

## Results Summary

| Dataset | Baseline Hit@10 | LLM Hit@10 | Improvement |
|---------|----------------|------------|-------------|
| Industrial | 0.0410 | **0.0476** | **+16.1%** ✅ |
| Video Games | 0.0887 | 0.0920 (prev) | Pending correct run |
| Cell Phones | 0.0377 | TBD | Not started |

## Architecture Visualization Code

See `visualize_llm_architecture.py` for generating publication-quality diagrams.
