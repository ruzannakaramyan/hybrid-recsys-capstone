import subprocess
import argparse
import random
import os
import pandas as pd
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Grid search for SASRec.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=6, help="Number of random configurations to try (default: 6 for fair comparison)")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    # Hyperparameter Search Space
    space = {
        "hidden_dim": [64],  # Focused on 64 as it was consistently better in previous runs
        "num_heads": [2, 4],
        "num_layers": [2, 3],
        "dropout": [0.1, 0.2, 0.3],
        "max_seq_len": [25, 50],
        "learning_rate": [0.0005, 0.001]
    }

    # Scale settings based on dataset
    is_cell_phones = (args.dataset == "cell_phones_and_accessories")
    sample_eval = 50000 if is_cell_phones else None
    
    results = []
    output_csv = f"grid_search_{args.dataset}.csv"
    
    configs = []
    for _ in range(args.num_samples):
        config = {k: random.choice(v) for k, v in space.items()}
        if config not in configs:
            configs.append(config)

    print(f"Starting Grid Search for {args.dataset} | {len(configs)} trials")
    if sample_eval:
        print(f"Using Stratified Evaluation Sample: {sample_eval} users")

    for i, config in enumerate(configs):
        print(f"\n--- Testing Config {i+1}/{len(configs)}: {config} ---")
        
        ckpt_path = f"src/grid_search_{args.dataset}_{i}.pth"
        
        cmd = [
            ".venv/bin/python", "src/train.py",
            "--dataset", args.dataset,
            "--epochs", str(args.epochs),
            "--hidden_dim", str(config["hidden_dim"]),
            "--num_heads", str(config["num_heads"]),
            "--num_layers", str(config["num_layers"]),
            "--dropout", str(config["dropout"]),
            "--max_seq_len", str(config["max_seq_len"]),
            "--learning_rate", str(config["learning_rate"]),
            "--checkpoint", ckpt_path,
            "--eval_every", "1",
            "--patience", "5"
        ]
        
        if sample_eval:
            cmd += ["--sample_eval", str(sample_eval)]

        # Run training
        try:
            # Using Popen to stream output in real-time
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1  # Line buffered
            )
            
            hit_score = 0.0
            for line in process.stdout:
                print(line, end='', flush=True)  # Stream to grid_search.py's own stdout
                if "Final Best Hit@10:" in line:
                    hit_score = float(line.split(":")[1].strip())
            
            process.wait()
            if process.returncode != 0:
                print(f"Error in Trial {i+1}: Process exited with code {process.returncode}")
            else:
                config["best_hit_at_10"] = hit_score
                results.append(config)
                
                # Save periodic results to CSV
                pd.DataFrame(results).to_csv(output_csv, index=False)
            
        except Exception as e:
            print(f"Exception in Trial {i+1}: {e}")

    print(f"\nGrid Search Complete. Results saved to {output_csv}")

if __name__ == "__main__":
    main()
