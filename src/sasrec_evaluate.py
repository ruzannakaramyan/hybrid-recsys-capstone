import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from model import SASRec


def build_vocab_from_train(train_csv_path):
    data = pd.read_csv(train_csv_path)
    all_items = set(data["parent_asin"].astype(str).unique())
    for hist in data["history"]:
        for token in str(hist).split():
            if token and token.lower() != "nan":
                all_items.add(token)
    sorted_items = sorted(list(all_items))
    return {item: idx + 1 for idx, item in enumerate(sorted_items)}


class EvalDataset(Dataset):
    def __init__(self, csv_file, item_vocab, max_seq_len=25, max_item_id=None, sample_users=None):
        self.data = pd.read_csv(csv_file)
        self.item_vocab = item_vocab
        self.max_seq_len = max_seq_len
        self.max_item_id = max_item_id

        if sample_users and sample_users < len(self.data):
            print(f"Applying stratified sampling: {sample_users} / {len(self.data)} users")
            # Stratify by sequence length
            self.data['seq_len'] = self.data['history'].apply(lambda x: len(str(x).split()))
            # Create bins for seq_len (using 5 quantiles or unique values if small)
            num_bins = min(5, self.data['seq_len'].nunique())
            self.data['bin'] = pd.qcut(self.data['seq_len'], q=num_bins, labels=False, duplicates='drop')
            
            # Stratified sample
            self.data = self.data.groupby('bin', group_keys=False).apply(
                lambda x: x.sample(n=int(len(x) * sample_users / len(self.data)), random_state=42)
            )
            print(f"Sampled size: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        target_item_idx = self.item_vocab.get(str(row["parent_asin"]), 0)
        if self.max_item_id is not None and target_item_idx >= self.max_item_id:
            target_item_idx = 0

        history_tokens = []
        for token in str(row["history"]).split():
            if token and token.lower() != "nan":
                history_tokens.append(token)
        history_idx = [self.item_vocab.get(item, 0) for item in history_tokens]
        if self.max_item_id is not None:
            history_idx = [idx if idx < self.max_item_id else 0 for idx in history_idx]

        if len(history_idx) >= self.max_seq_len:
            seq = history_idx[-self.max_seq_len :]
        else:
            seq = [0] * (self.max_seq_len - len(history_idx)) + history_idx

        return torch.tensor(seq, dtype=torch.long), torch.tensor(target_item_idx, dtype=torch.long)


def evaluate(model, loader, device, k=10):
    model.eval()
    total = 0
    hit_count = 0.0
    ndcg_sum = 0.0

    with torch.no_grad():
        for seqs, targets in loader:
            seqs = seqs.to(device)
            targets = targets.to(device)

            logits = model(seqs)
            logits[:, 0] = float("-inf")
            topk = torch.topk(logits, k=k, dim=1).indices

            for i in range(targets.size(0)):
                target = targets[i].item()
                if target == 0:
                    continue

                total += 1
                pred_items = topk[i].tolist()
                if target in pred_items:
                    hit_count += 1.0
                    rank = pred_items.index(target)
                    ndcg_sum += 1.0 / torch.log2(torch.tensor(rank + 2.0)).item()

    hit_k = hit_count / total if total > 0 else 0.0
    ndcg_k = ndcg_sum / total if total > 0 else 0.0
    return hit_k, ndcg_k, total, hit_count


def main():
    parser = argparse.ArgumentParser(description="Evaluate SASRec with Hit@K and NDCG@K.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="industrial_and_scientific",
        choices=["industrial_and_scientific", "cell_phones_and_accessories", "video_games"],
        help="Dataset name suffix used in data filenames.",
    )
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test"], help="Evaluation split.")
    parser.add_argument("--k", type=int, default=10, help="Top-K for metrics.")
    parser.add_argument("--batch_size", type=int, default=256, help="Evaluation batch size.")
    parser.add_argument("--max_seq_len", type=int, default=25, help="Max sequence length.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Embedding dimension.")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of attention heads.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--use_llm_embeddings", action="store_true", help="Initialize item matrix with LLM embeddings.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (.pth).")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for evaluation. 'auto' prefers CUDA, then CPU (MPS avoided for Transformer eval stability).",
    )
    parser.add_argument("--sample_users", type=int, default=None, help="If set, evaluate only on a stratified sample of users.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.join(script_dir, "..")
    train_csv = os.path.join(repo_root, "data", f"train_{args.dataset}_merged.csv")
    eval_csv = os.path.join(repo_root, "data", f"{args.split}_{args.dataset}_merged.csv")
    default_ckpt_name = f"sasrec_{args.dataset}.pth"
    checkpoint_path = args.checkpoint or os.path.join(script_dir, default_ckpt_name)

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Train file not found: {train_csv}")
    if not os.path.exists(eval_csv):
        raise FileNotFoundError(f"Eval file not found: {eval_csv}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.device == "auto":
        # MPS currently hits a nested-tensor NotImplementedError in Transformer eval on some setups.
        # Prefer CUDA when available; otherwise fall back to CPU for reliability.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset} | Split: {args.split}")
    print(f"Checkpoint: {checkpoint_path}")

    item_vocab = build_vocab_from_train(train_csv)
    state_dict = torch.load(checkpoint_path, map_location=device)
    checkpoint_vocab_size = state_dict["item_embedding.weight"].shape[0]
    eval_dataset = EvalDataset(
        eval_csv,
        item_vocab=item_vocab,
        max_seq_len=args.max_seq_len,
        max_item_id=checkpoint_vocab_size,
        sample_users=args.sample_users
    )
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    llm_embeds = None
    if args.use_llm_embeddings:
        pt_file = os.path.join(script_dir, "..", "data", f"item_embeddings_{args.dataset}.pt")
        if not os.path.exists(pt_file):
            raise FileNotFoundError(f"Missing LLM embeddings: {pt_file} (Run src/generate_embeddings.py first)")
        print(f"Loading LLM Semantic Vectors from {pt_file}...")
        llm_embeds = torch.load(pt_file, map_location=device)

    model = SASRec(
        vocab_size=checkpoint_vocab_size,
        max_seq_len=args.max_seq_len,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout_rate=args.dropout,
        llm_embeds=llm_embeds
    ).to(device)
    model.load_state_dict(state_dict)

    hit_k, ndcg_k, evaluated_samples, hit_count = evaluate(model, eval_loader, device=device, k=args.k)
    print(f"Evaluated samples: {evaluated_samples}")
    print(f"Hits@{args.k} (raw): {int(hit_count)}")
    print(f"Hit@{args.k}:  {hit_k:.8f}")
    print(f"NDCG@{args.k}: {ndcg_k:.8f}")


if __name__ == "__main__":
    main()

