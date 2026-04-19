import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from dataset import SequentialDataset
from bpr_model import BPRMF, build_user_vocab

class BPREvalDataset(Dataset):
    def __init__(self, csv_file, user_vocab, item_vocab):
        self.data = pd.read_csv(csv_file)
        self.user_vocab = user_vocab
        self.item_vocab = item_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        user_idx = self.user_vocab.get(str(row["user_id"]), 0)
        target_item_idx = self.item_vocab.get(str(row["parent_asin"]), 0)
        return torch.tensor(user_idx, dtype=torch.long), torch.tensor(target_item_idx, dtype=torch.long)

def evaluate_bpr(model, loader, device, k=10):
    model.eval()
    total = 0
    hit_count = 0.0
    ndcg_sum = 0.0
    with torch.no_grad():
        for users, targets in loader:
            users, targets = users.to(device), targets.to(device)
            logits = model.predict_all_items(users)
            logits[:, 0] = float("-inf")
            topk = torch.topk(logits, k=k, dim=1).indices
            for i in range(targets.size(0)):
                target = targets[i].item()
                if target == 0: continue
                total += 1
                pred_items = topk[i].tolist()
                if target in pred_items:
                    hit_count += 1.0
                    rank = pred_items.index(target)
                    ndcg_sum += 1.0 / torch.log2(torch.tensor(rank + 2.0)).item()
    return hit_count / total if total > 0 else 0.0, ndcg_sum / total if total > 0 else 0.0, total, hit_count

def evaluate_toppop(train_csv, eval_csv, item_vocab, k=10):
    freqs = Counter()
    train_data = pd.read_csv(train_csv)
    for idx, row in train_data.iterrows():
        target_idx = item_vocab.get(str(row["parent_asin"]), 0)
        if target_idx != 0: freqs[target_idx] += 1
        hist_str = str(row["history"])
        if hist_str and hist_str.lower() != "nan":
            for item in hist_str.split():
                i_idx = item_vocab.get(item, 0)
                if i_idx != 0: freqs[i_idx] += 1
    topk_items = [item for item, count in freqs.most_common(k)]
    
    eval_data = pd.read_csv(eval_csv)
    total = 0
    hit_count = 0.0
    ndcg_sum = 0.0
    for idx, row in eval_data.iterrows():
        target_idx = item_vocab.get(str(row["parent_asin"]), 0)
        if target_idx == 0: continue
        total += 1
        if target_idx in topk_items:
            hit_count += 1.0
            rank = topk_items.index(target_idx)
            ndcg_sum += 1.0 / torch.log2(torch.tensor(rank + 2.0)).item()
    return hit_count / total if total > 0 else 0.0, ndcg_sum / total if total > 0 else 0.0, total, hit_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=["toppop", "bpr"])
    parser.add_argument("--dataset", type=str, default="industrial_and_scientific")
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_csv = os.path.join(script_dir, "..", "data", f"train_{args.dataset}_merged.csv")
    eval_csv = os.path.join(script_dir, "..", "data", f"{args.split}_{args.dataset}_merged.csv")

    temp_seq = SequentialDataset(train_csv, max_seq_len=25)
    item_vocab = temp_seq.item_vocab

    if args.model_type == "toppop":
        hit_k, ndcg_k, evaluated_samples, hit_count = evaluate_toppop(train_csv, eval_csv, item_vocab, k=args.k)
    elif args.model_type == "bpr":
        user_vocab = build_user_vocab(train_csv)
        num_items, num_users = len(item_vocab) + 1, len(user_vocab)
        checkpoint_path = args.checkpoint or os.path.join(script_dir, f"bpr_{args.dataset}.pth")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model = BPRMF(num_users=num_users, num_items=num_items, hidden_dim=64).to(device)
        model.load_state_dict(state_dict)

        eval_dataset = BPREvalDataset(eval_csv, user_vocab, item_vocab)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
        hit_k, ndcg_k, evaluated_samples, hit_count = evaluate_bpr(model, eval_loader, device=device, k=args.k)

    print(f"Evaluated samples: {evaluated_samples}")
    print(f"Hit@{args.k}:  {hit_k:.8f}")
    print(f"NDCG@{args.k}: {ndcg_k:.8f}")

if __name__ == "__main__":
    main()
