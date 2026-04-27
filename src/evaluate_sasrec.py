import argparse
import os
import torch
from torch.utils.data import DataLoader
from dataset import SequentialDataset
from sasrec_model import SASRec
from sasrec_evaluate import evaluate, EvalDataset, build_vocab_from_train

def main():
    parser = argparse.ArgumentParser(description="Evaluate SASRec model performance.")
    parser.add_argument("--dataset", type=str, required=True, choices=["industrial_and_scientific", "video_games", "cell_phones_and_accessories"])
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model .pth file")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=25)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--use_llm_embeddings", action="store_true")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--sample_users", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(script_dir, "..", "data", f"train_{args.dataset}_merged.csv")
    eval_file = os.path.join(script_dir, "..", "data", f"{args.split}_{args.dataset}_merged.csv")

    print(f"Building vocab from {train_file}...")
    item_vocab = build_vocab_from_train(train_file)
    vocab_size = len(item_vocab) + 1

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

    print(f"Loading checkpoint: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    eval_dataset = EvalDataset(eval_file, item_vocab, max_seq_len=args.max_seq_len, sample_users=args.sample_users)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    hit_k, ndcg_k, total_users, hit_count = evaluate(model, eval_loader, device=device, k=10)

    print(f"\n✅ Results for {args.dataset} ({args.split} set):")
    print(f"Evaluated samples: {total_users}")
    print(f"Hit@10:  {hit_k:.8f}")
    print(f"NDCG@10: {ndcg_k:.8f}")

if __name__ == "__main__":
    main()
