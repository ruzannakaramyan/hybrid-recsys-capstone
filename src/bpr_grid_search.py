import subprocess
import argparse
import random
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Grid search for BPR-MF.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=6, help="Number of random configurations to try (default: 6 for fair comparison)")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    # Hyperparameter Search Space
    space = {
        "hidden_dim": [32, 64, 128],
        "learning_rate": [0.0005, 0.001],
        "weight_decay": [1e-5, 1e-4, 1e-3]
    }

    results = []
    output_csv = f"grid_search_bpr_{args.dataset}.csv"
    
    configs = []
    for _ in range(args.num_samples):
        config = {k: random.choice(v) for k, v in space.items()}
        if config not in configs:
            configs.append(config)

    print(f"Starting BPR Grid Search for {args.dataset} | {len(configs)} trials")

    for i, config in enumerate(configs):
        print(f"\n--- Testing Config {i+1}/{len(configs)}: {config} ---")
        
        ckpt_path = f"src/grid_search_bpr_{args.dataset}_{i}.pth"
        
        cmd = [
            ".venv/bin/python", "-u", "src/train_bpr.py",
            "--dataset", args.dataset,
            "--epochs", str(args.epochs),
            "--hidden_dim", str(config["hidden_dim"]),
            "--learning_rate", str(config["learning_rate"]),
            "--weight_decay", str(config["weight_decay"]),
            "--checkpoint", ckpt_path,
            "--eval_every", "1",
            "--patience", "5"
        ]

        # Using Popen to stream output in real-time
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
            else:
                print(f"Error in Trial {i+1}: Process exited with code {process.returncode}")
                
        except Exception as e:
            print(f"Exception in Trial {i+1}: {e}")

    print(f"\nBPR Grid Search Complete. Results saved to {output_csv}")

if __name__ == "__main__":
    main()
