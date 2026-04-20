import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SequentialDataset
from sasrec_model import SASRec
from sasrec_evaluate import evaluate, EvalDataset, build_vocab_from_train


def main():
    parser = argparse.ArgumentParser(description="Train SASRec model.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="industrial_and_scientific",
        choices=["industrial_and_scientific", "video_games", "cell_phones_and_accessories"],
        help="Dataset name suffix used in data filenames.",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Optimizer learning rate.")
    parser.add_argument("--max_seq_len", type=int, default=25, help="Maximum sequence length.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Embedding dimension.")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of attention heads.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--use_llm_embeddings", action="store_true", help="Initialize item matrix with LLM embeddings.")
    parser.add_argument("--freeze_emb_epochs", type=int, default=0, help="Freeze LLM embeddings for first N epochs, then unfreeze (two-phase training). 0 = never freeze.")
    parser.add_argument("--llm_lr_factor", type=float, default=0.5, help="LR multiplier when unfreezing LLM embeddings (default: 0.5)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 weight decay for Adam optimizer.")
    parser.add_argument("--lr_scheduler", action="store_true", help="Enable cosine annealing LR scheduler.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional output checkpoint path. Defaults to src/sasrec_<dataset>.pth.",
    )
    parser.add_argument("--eval_every", type=int, default=1, help="Run validation every N epochs.")
    parser.add_argument("--patience", type=int, default=5, help="Stop after P evaluations with no improvement.")
    parser.add_argument("--sample_eval", type=int, default=None, help="If set, evaluate only on a stratified sample of users during training.")
    args = parser.parse_args()

    # 1. Setup device (Handles Apple M-Series GPUs properly)
    # Prefer CUDA if available; fall back to CPU for Transformer stability on Mac (MPS issues)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 2. Load Dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(script_dir, "..", "data", f"train_{args.dataset}_merged.csv")
    print(f"Loading dataset: {args.dataset}")
    train_dataset = SequentialDataset(train_file, max_seq_len=args.max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 3. Initialize Model
    # vocab_size needs to be +1 to accommodate the padding index (0)
    vocab_size = len(train_dataset.item_vocab) + 1
    print(f"Vocabulary Size (including padding): {vocab_size}")

    llm_embeds = None
    if args.use_llm_embeddings:
        pt_file = os.path.join(script_dir, "..", "data", f"item_embeddings_{args.dataset}.pt")
        if not os.path.exists(pt_file):
            raise FileNotFoundError(f"Missing LLM embeddings: {pt_file} (Run src/generate_embeddings.py first)")
        print(f"Loading LLM Semantic Vectors from {pt_file}...")
        llm_embeds = torch.load(pt_file, map_location=device)
        print(f"LLM Embedding Tensor Shape: {llm_embeds.shape}")

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
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Mathematically ignores the '0' padding
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.lr_scheduler else None


    # 4. Training Loop
    print("\nStarting Training...")
    best_hit = 0.0
    patience_counter = 0

    valid_file = os.path.join(script_dir, "..", "data", f"valid_{args.dataset}_merged.csv")
    valid_dataset = EvalDataset(
        valid_file, 
        item_vocab=train_dataset.item_vocab, 
        max_seq_len=args.max_seq_len,
        sample_users=args.sample_eval
    )
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    for epoch in range(args.epochs):
        # Two-phase LLM embedding freeze/unfreeze
        if args.use_llm_embeddings and args.freeze_emb_epochs > 0:
            if epoch == 0:
                print(f"Phase 1: Freezing LLM embeddings for {args.freeze_emb_epochs} epochs...")
                model.item_embedding.weight.requires_grad_(False)
            elif epoch == args.freeze_emb_epochs:
                print(f"Phase 2: Unfreezing LLM embeddings for fine-tuning...")
                model.item_embedding.weight.requires_grad_(True)
                # Reset optimizer with adjusted LR - less aggressive than before
                new_lr = args.learning_rate * args.llm_lr_factor
                print(f"Adjusting learning rate to {new_lr:.6f} for fine-tuning phase")
                optimizer = optim.Adam(model.parameters(), lr=new_lr, weight_decay=args.weight_decay)

        model.train()
        total_loss = 0.0

        for batch_idx, (seqs, targets) in enumerate(train_loader):
            seqs, targets = seqs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(seqs)
            loss = criterion(logits, targets)

            loss.backward()
            # Gradient clipping for stability with LLM embeddings
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch + 1}/{args.epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        lr_now = optimizer.param_groups[0]['lr']
        print(f"--- Epoch {epoch + 1} Completed | Avg Loss: {avg_loss:.4f} | LR: {lr_now:.6f} ---")

        if scheduler:
            scheduler.step()

        # Validation and Early Stopping
        if (epoch + 1) % args.eval_every == 0:
            hit_k, ndcg_k, _, _ = evaluate(model, valid_loader, device=device, k=10)
            print(f"Validation @ Epoch {epoch + 1} | Hit@10: {hit_k:.4f} | NDCG@10: {ndcg_k:.4f}")

            if hit_k > best_hit:
                best_hit = hit_k
                patience_counter = 0
                # Save Best Model
                default_ckpt_name = f"sasrec_{args.dataset}_best.pth"
                save_path = args.checkpoint if args.checkpoint else os.path.join(script_dir, default_ckpt_name)
                torch.save(model.state_dict(), save_path)
                print(f"🏆 New Best Hit@10! Model saved to {save_path}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{args.patience}")

            if patience_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    print(f"\nFinal Best Hit@10: {best_hit:.4f}")


if __name__ == "__main__":
    main()