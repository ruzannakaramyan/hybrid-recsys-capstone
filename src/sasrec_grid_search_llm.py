#!/usr/bin/env python3
"""
Grid search for SASRec with LLM embeddings.
Runs 6 trials with different hyperparameters using LLM-initialized embeddings.
"""

import subprocess
import argparse
import random
import os
import pandas as pd
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Grid search for SASRec with LLM embeddings.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=6, help="Number of random configurations (default: 6)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--freeze_epochs", type=int, default=5, help="Freeze LLM embeddings for first N epochs")
    args = parser.parse_args()

    # Hyperparameter Search Space (focused on training dynamics with LLM)
    space = {
        "hidden_dim": [64],  # LLM is 384-dim, projected to 64
        "num_heads": [2, 4],
        "num_layers": [2, 3],
        "dropout": [0.1, 0.2, 0.3],
        "max_seq_len": [25, 50],
        "learning_rate": [0.0005, 0.001],
    }

    output_csv = f"grid_search_llm_{args.dataset}.csv"
    results = []

    # Ensure embeddings exist
    embed_file = f"data/item_embeddings_{args.dataset}.pt"
    if not os.path.exists(embed_file):
        print(f"ERROR: LLM embeddings not found: {embed_file}")
        print("Run: python src/generate_embeddings.py --dataset " + args.dataset)
        return

    print(f"="*60)
    print(f"LLM GRID SEARCH - {args.dataset}")
    print(f"Embeddings: {embed_file}")
    print(f"Trials: {args.num_samples}")
    print(f"Freeze LLM for first {args.freeze_epochs} epochs")
    print(f"="*60)

    for i in range(args.num_samples):
        config = {k: random.choice(v) for k, v in space.items()}
        print(f"\n{'='*60}")
        print(f"LLM Trial {i+1}/{args.num_samples}")
        print(f"Config: {config}")
        print(f"{'='*60}")

        cmd = [
            "python", "src/sasrec_train.py",
            "--dataset", args.dataset,
            "--epochs", str(args.epochs),
            "--hidden_dim", str(config["hidden_dim"]),
            "--num_heads", str(config["num_heads"]),
            "--num_layers", str(config["num_layers"]),
            "--dropout", str(config["dropout"]),
            "--max_seq_len", str(config["max_seq_len"]),
            "--learning_rate", str(config["learning_rate"]),
            "--use_llm_embeddings",
            "--freeze_emb_epochs", str(args.freeze_epochs),
            "--checkpoint", f"src/grid_search_llm_{args.dataset}_{i}.pth",
            "--patience", "5",
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            hit_score = 0.0
            for line in process.stdout:
                print(line, end='', flush=True)
                if "Final Best Hit@10:" in line:
                    hit_score = float(line.split(":")[1].strip())

            process.wait()
            if process.returncode == 0:
                config["best_hit_at_10"] = hit_score
                results.append(config)
                pd.DataFrame(results).to_csv(output_csv, index=False)
                print(f"✅ LLM Trial {i+1} complete - Hit@10: {hit_score:.4f}")
            else:
                print(f"❌ Error in LLM Trial {i+1}: Exit code {process.returncode}")

        except Exception as e:
            print(f"❌ Exception in LLM Trial {i+1}: {e}")

    print(f"\n{'='*60}")
    print(f"LLM Grid Search Complete!")
    print(f"Results: {output_csv}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
