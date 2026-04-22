#!/usr/bin/env python3
"""
Fast stratified LLM training for SASRec.
Uses history-length stratified sampling for faster, better training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
import argparse
import os
import pandas as pd
import numpy as np
from sasrec_model import SASRec
from dataset import SequentialDataset
from sasrec_evaluate import evaluate, EvalDataset

class StratifiedUserSampler(Sampler):
    """Stratified sampler by user history length."""
    def __init__(self, dataset, num_samples=None, buckets=5, seed=42):
        self.dataset = dataset
        self.seed = seed
        np.random.seed(seed)
        
        # Get all user indices and their sequence lengths
        self.indices = list(range(len(dataset)))
        lengths = []
        for i in self.indices:
            seq, _ = dataset[i]
            # Count non-zero elements (actual items, not padding)
            length = (seq != 0).sum().item()
            lengths.append(length)
        
        lengths = np.array(lengths)
        
        # Create stratified buckets
        percentiles = np.linspace(0, 100, buckets + 1)
        bucket_edges = np.percentile(lengths, percentiles)
        
        # Assign users to buckets
        bucket_labels = np.digitize(lengths, bucket_edges[1:-1])
        
        # Sample proportionally from each bucket
        samples_per_bucket = (num_samples // buckets) if num_samples else None
        
        self.sampled_indices = []
        for b in range(buckets):
            bucket_indices = [i for i, label in enumerate(bucket_labels) if label == b]
            if num_samples and len(bucket_indices) > samples_per_bucket:
                # Sample from this bucket
                sampled = np.random.choice(bucket_indices, samples_per_bucket, replace=False)
                self.sampled_indices.extend(sampled.tolist())
            else:
                # Take all from this bucket
                self.sampled_indices.extend(bucket_indices)
        
        print(f"Stratified sampling: {len(self.sampled_indices)} users from {len(self.indices)}")
        print(f"  Buckets: {buckets}, maintaining history length distribution")
        
    def __iter__(self):
        np.random.seed(self.seed)
        indices = self.sampled_indices.copy()
        np.random.shuffle(indices)
        return iter(indices)
    
    def __len__(self):
        return len(self.sampled_indices)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)  # Higher for less overfitting
    parser.add_argument("--max_seq_len", type=int, default=25)
    parser.add_argument("--learning_rate", type=float, default=0.0005)  # Lower for stability
    parser.add_argument("--epochs", type=int, default=30)  # Fewer epochs
    parser.add_argument("--batch_size", type=int, default=256)  # Larger batches
    parser.add_argument("--patience", type=int, default=3)  # Stop faster
    parser.add_argument("--weight_decay", type=float, default=1e-3)  # Stronger regularization
    parser.add_argument("--use_llm_embeddings", action="store_true")
    parser.add_argument("--freeze_emb_epochs", type=int, default=5)
    parser.add_argument("--llm_lr_factor", type=float, default=0.3)
    parser.add_argument("--sample_train", type=int, default=50000, help="Stratified sample size for training")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load dataset
    train_file = os.path.join(script_dir, "..", "data", f"train_{args.dataset}_merged.csv")
    print(f"Loading dataset: {args.dataset}")
    train_dataset = SequentialDataset(train_file, max_seq_len=args.max_seq_len)
    
    # Stratified sampling for training
    if args.sample_train and args.sample_train < len(train_dataset):
        sampler = StratifiedUserSampler(train_dataset, num_samples=args.sample_train, buckets=5)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Model setup
    vocab_size = len(train_dataset.item_vocab) + 1
    print(f"Vocabulary Size: {vocab_size}")
    
    llm_embeds = None
    if args.use_llm_embeddings:
        pt_file = os.path.join(script_dir, "..", "data", f"item_embeddings_{args.dataset}.pt")
        print(f"Loading LLM embeddings from {pt_file}...")
        llm_embeds = torch.load(pt_file, map_location=device)
    
    model = SASRec(
        vocab_size=vocab_size,
        max_seq_len=args.max_seq_len,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout_rate=args.dropout,
        llm_embeds=llm_embeds
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Training loop
    print(f"\n=== FAST STRATIFIED LLM TRAINING ===")
    print(f"Settings: dropout={args.dropout}, weight_decay={args.weight_decay}, patience={args.patience}")
    print(f"Training on stratified sample of {args.sample_train} users")
    
    best_hit = 0.0
    patience_counter = 0
    
    valid_file = os.path.join(script_dir, "..", "data", f"valid_{args.dataset}_merged.csv")
    valid_dataset = EvalDataset(valid_file, item_vocab=train_dataset.item_vocab, max_seq_len=args.max_seq_len)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    
    for epoch in range(args.epochs):
        # Freeze/unfreeze LLM
        if args.use_llm_embeddings and args.freeze_emb_epochs > 0:
            if epoch == 0:
                print(f"Phase 1: Freezing LLM for {args.freeze_emb_epochs} epochs...")
                model.item_embedding.weight.requires_grad_(False)
            elif epoch == args.freeze_emb_epochs:
                print(f"Phase 2: Unfreezing LLM with LR x {args.llm_lr_factor}...")
                model.item_embedding.weight.requires_grad_(True)
                for pg in optimizer.param_groups:
                    pg['lr'] = args.learning_rate * args.llm_lr_factor
        
        # Training
        model.train()
        total_loss = 0.0
        for batch_idx, (seqs, targets) in enumerate(train_loader):
            seqs, targets = seqs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(seqs)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} ---")
        
        # Validation
        hit_k, ndcg_k, _, _ = evaluate(model, valid_loader, device=device, k=10)
        print(f"Validation | Hit@10: {hit_k:.4f} | NDCG@10: {ndcg_k:.4f}")
        
        if hit_k > best_hit:
            best_hit = hit_k
            patience_counter = 0
            save_path = args.checkpoint or f"sasrec_{args.dataset}_stratified_llm.pth"
            torch.save(model.state_dict(), save_path)
            print(f"🏆 New Best! Saved to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{args.patience}")
        
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"\nFinal Best Hit@10: {best_hit:.4f}")

if __name__ == "__main__":
    main()
