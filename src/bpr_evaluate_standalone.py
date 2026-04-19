#!/usr/bin/env python3
"""
Standalone BPR evaluation script for test set.
Loads a trained checkpoint and evaluates on test data.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader

from dataset import SequentialDataset
from bpr_model import BPRMF, BPRDataset, BPREvalDataset, build_user_vocab
from bpr_evaluate import evaluate_bpr


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained BPR model on test set.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset} | Split: {args.split}")
    print(f"Checkpoint: {args.checkpoint}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(script_dir, "..", "data", f"train_{args.dataset}_merged.csv")
    eval_file = os.path.join(script_dir, "..", "data", f"{args.split}_{args.dataset}_merged.csv")

    # Load vocab
    print(f"Loading vocab for {args.dataset}...")
    temp_seq_dataset = SequentialDataset(train_file, max_seq_len=25)
    item_vocab = temp_seq_dataset.item_vocab
    user_vocab = build_user_vocab(train_file)

    num_items, num_users = len(item_vocab) + 1, len(user_vocab)
    print(f"Users: {num_users} | Items: {num_items}")

    # Load model
    print("Loading model...")
    model = BPRMF(num_users=num_users, num_items=num_items, hidden_dim=args.hidden_dim).to(device)
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    print(f"✅ Loaded checkpoint from {args.checkpoint}")

    # Load eval dataset
    if not os.path.exists(eval_file):
        raise FileNotFoundError(f"Eval file not found: {eval_file}")
    
    eval_dataset = BPREvalDataset(eval_file, user_vocab, item_vocab)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Evaluating on {len(eval_dataset)} users...")

    # Evaluate
    hit_k, ndcg_k, total, hit_count = evaluate_bpr(model, eval_loader, device=device, k=10)
    
    print(f"\n{'='*50}")
    print(f"✅ TEST SET RESULTS - BPR on {args.dataset}")
    print(f"{'='*50}")
    print(f"Hit@10:  {hit_k:.8f}")
    print(f"NDCG@10: {ndcg_k:.8f}")
    print(f"Total users evaluated: {total}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
