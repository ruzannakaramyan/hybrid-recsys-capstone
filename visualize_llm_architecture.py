"""
Generate publication-quality architecture diagram for LLM-Enhanced SASRec.
Run: python visualize_llm_architecture.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(1, 1, figsize=(14, 16))
ax.set_xlim(0, 14)
ax.set_ylim(0, 16)
ax.axis('off')

def draw_box(ax, x, y, width, height, text, color, fontsize=9, text_color='white', alpha=0.9):
    """Draw a rounded box with text."""
    box = FancyBboxPatch((x, y), width, height, 
                         boxstyle="round,pad=0.1", 
                         facecolor=color, 
                         edgecolor='black', 
                         linewidth=2,
                         alpha=alpha)
    ax.add_patch(box)
    
    # Wrap text if too long
    words = text.split('\n')
    line_height = height / (len(words) + 1)
    for i, word in enumerate(words):
        ax.text(x + width/2, y + height - (i+1)*line_height, word, 
                ha='center', va='center', fontsize=fontsize, 
                color=text_color, weight='bold')
    return box

def draw_arrow(ax, x1, y1, x2, y2, color='black', style='->'):
    """Draw arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=2))

# Title
ax.text(7, 15.5, 'LLM-Enhanced SASRec Architecture', 
        ha='center', va='center', fontsize=18, weight='bold')
ax.text(7, 15, 'Sequential Recommendation with Semantic Knowledge Injection', 
        ha='center', va='center', fontsize=12, style='italic', color='gray')

# Input Layer
draw_box(ax, 4, 13.5, 6, 0.8, 'User Sequence\n[Item₁, Item₂, Item₃, ..., Itemₙ]', '#2C3E50', 10)

# Item Embedding Layer (main block)
item_emb_box = FancyBboxPatch((1, 10.5), 12, 2.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#E8F4F8', 
                               edgecolor='#3498DB', 
                               linewidth=3)
ax.add_patch(item_emb_box)
ax.text(7, 12.7, 'ITEM EMBEDDING LAYER', ha='center', fontsize=12, weight='bold', color='#2C3E50')

# LLM Processing
draw_box(ax, 1.5, 11, 3.5, 1.2, 'Item Text\nDescription', '#9B59B6', 9)
draw_arrow(ax, 3.25, 11, 4.5, 11, '#9B59B6')

# LLM Model
draw_box(ax, 4.5, 10.8, 2.5, 1.2, 'SentenceTransformer\nall-MiniLM-L6-v2\n384-dim', '#8E44AD', 8)
draw_arrow(ax, 6.75, 11.4, 8, 11.4, '#8E44AD')

# LLM Vector
draw_box(ax, 8, 11, 2, 0.8, 'LLM Vector\n[384 dims]', '#9B59B6', 9)
draw_arrow(ax, 9.5, 11, 10.5, 11.2, '#9B59B6')

# Simple Projection (KEY INNOVATION)
proj_box = draw_box(ax, 10.5, 10.7, 2.5, 1.3, 'PROJECTION\n384 → 64\nLinear + LayerNorm', '#E74C3C', 9)
# Highlight this box
proj_rect = Rectangle((10.3, 10.5), 2.9, 1.7, fill=False, edgecolor='red', linewidth=3, linestyle='--')
ax.add_patch(proj_rect)
ax.text(11.75, 12.3, 'KEY INNOVATION', ha='center', fontsize=8, color='red', weight='bold')

# Output embedding
draw_box(ax, 11, 9.3, 1.5, 0.7, '64-dim\nEmbedding', '#27AE60', 9)

# Position Embedding
draw_box(ax, 1.5, 9, 3, 0.8, 'Positional Encoding\n[0, 1, 2, ..., 25]', '#F39C12', 9)
draw_arrow(ax, 3, 9, 4, 9.5, '#F39C12')

# Dropout
draw_box(ax, 5.5, 9, 3, 0.7, 'Embedding Dropout (p=0.2)', '#95A5A6', 9)

# Transformer Block
trans_box = FancyBboxPatch((2, 5.5), 10, 3, 
                            boxstyle="round,pad=0.1", 
                            facecolor='#FDEDEC', 
                            edgecolor='#E74C3C', 
                            linewidth=3)
ax.add_patch(trans_box)
ax.text(7, 8.2, 'TRANSFORMER ENCODER STACK', ha='center', fontsize=12, weight='bold', color='#C0392B')

# Multi-Head Attention
draw_box(ax, 2.5, 6.8, 3.5, 1, 'Multi-Head Self-Attention\nnum_heads=2 or 4', '#E74C3C', 9)

# Feed Forward
draw_box(ax, 6.5, 6.8, 3.5, 1, 'Feed-Forward Network\n64 → 256 → 64', '#E74C3C', 9)

# Layer Stack
draw_box(ax, 10.5, 6.8, 1.5, 1, 'Stack\n2-3 layers', '#E74C3C', 9)

# Output
draw_box(ax, 5.5, 4.5, 3, 0.8, 'Final Hidden State', '#3498DB', 9)
draw_arrow(ax, 7, 5.5, 7, 5.3, '#3498DB')

# Prediction
draw_box(ax, 5, 3.3, 4, 0.9, 'Linear → Logits\nBCE Loss + Neg Sampling', '#2980B9', 9, text_color='white')

# Output
draw_box(ax, 5.5, 2, 3, 0.8, 'Next Item Prediction', '#2C3E50', 10)

# Two-Phase Training Box (side panel)
phase_box = FancyBboxPatch((0.2, 0.3), 4, 2.2, 
                          boxstyle="round,pad=0.1", 
                          facecolor='#E8F8F5', 
                          edgecolor='#27AE60', 
                          linewidth=2)
ax.add_patch(phase_box)
ax.text(2.2, 2.2, 'Two-Phase Training', ha='center', fontsize=10, weight='bold', color='#27AE60')
ax.text(2.2, 1.8, 'Phase 1: LLM FROZEN (epochs 1-5)', ha='center', fontsize=8, color='#2C3E50')
ax.text(2.2, 1.5, 'LR=0.001, learn to USE LLM', ha='center', fontsize=8, color='#7F8C8D')
ax.text(2.2, 1.1, 'Phase 2: LLM UNFROZEN (epochs 6+)', ha='center', fontsize=8, color='#2C3E50')
ax.text(2.2, 0.8, 'LR=0.0005, adapt to rec patterns', ha='center', fontsize=8, color='#7F8C8D')

# Simple vs Complex box (side panel)
simple_box = FancyBboxPatch((10, 0.3), 3.8, 2.2, 
                            boxstyle="round,pad=0.1", 
                            facecolor='#FEF5E7', 
                            edgecolor='#F39C12', 
                            linewidth=2)
ax.add_patch(simple_box)
ax.text(11.9, 2.2, 'Why Simple Works', ha='center', fontsize=10, weight='bold', color='#D68910')
ax.text(11.9, 1.7, '❌ Complex: 384→256→128→64', ha='center', fontsize=8, color='#C0392B')
ax.text(11.9, 1.4, '   Overfitting, many params', ha='center', fontsize=7, color='#7F8C8D')
ax.text(11.9, 1.0, '✅ Simple: 384→64 (single)', ha='center', fontsize=8, color='#27AE60')
ax.text(11.9, 0.7, '   Better generalization', ha='center', fontsize=7, color='#7F8C8D')

# Arrows from components to dropout
ax.annotate('', xy=(5.5, 9.35), xytext=(11.75, 9.3),
            arrowprops=dict(arrowstyle='->', color='#27AE60', lw=2))
ax.annotate('', xy=(5.5, 9.35), xytext=(3, 9),
            arrowprops=dict(arrowstyle='->', color='#F39C12', lw=2))

# Arrows to transformer
ax.annotate('', xy=(7, 8.5), xytext=(7, 9),
            arrowprops=dict(arrowstyle='->', color='#95A5A6', lw=2))

# Main flow arrows
for i in range(2):
    y_pos = 7.3 - i*0.4
    ax.annotate('', xy=(6.5, y_pos), xytext=(6, y_pos),
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5))
    ax.annotate('', xy=(10.5, y_pos), xytext=(10, y_pos),
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5))

plt.tight_layout()
plt.savefig('llm_sasrec_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('llm_sasrec_architecture.pdf', bbox_inches='tight', facecolor='white')
print("✅ Architecture diagrams saved:")
print("   - llm_sasrec_architecture.png (300 DPI)")
print("   - llm_sasrec_architecture.pdf (vector)")
print("\nFor your capstone, use the PDF for crisp printing!")
