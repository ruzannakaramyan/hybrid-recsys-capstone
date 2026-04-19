import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import SequentialDataset
from bpr_model import BPRMF, BPRDataset, BPREvalDataset, build_user_vocab
from bpr_evaluate import evaluate_bpr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="industrial_and_scientific")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)  # Standard SASRec-like epochs
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4) # Critical for MF
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--test_split", action="store_true", help="Evaluate on test set after training and report final results.")
    args = parser.parse_args()

    # Device Handling
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(script_dir, "..", "data", f"train_{args.dataset}_merged.csv")
    valid_file = os.path.join(script_dir, "..", "data", f"valid_{args.dataset}_merged.csv")

    # 1. Load Vocab & Datasets
    print(f"Loading vocab for {args.dataset}...")
    temp_seq_dataset = SequentialDataset(train_file, max_seq_len=25)
    item_vocab = temp_seq_dataset.item_vocab
    user_vocab = build_user_vocab(train_file)
    
    num_items, num_users = len(item_vocab) + 1, len(user_vocab)
    print(f"Users: {num_users} | Items: {num_items}")

    train_dataset = BPRDataset(train_file, user_vocab, item_vocab)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    valid_dataset = BPREvalDataset(valid_file, user_vocab, item_vocab)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. Init Model
    model = BPRMF(num_users=num_users, num_items=num_items, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # 3. Training with Early Stopping
    print("\nStarting BPR Training...")
    best_hit = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, (user_idx, pos_item_idx, neg_item_idx) in enumerate(train_loader):
            user_idx = user_idx.to(device)
            pos_item_idx = pos_item_idx.to(device)
            neg_item_idx = neg_item_idx.to(device)
            
            optimizer.zero_grad()
            pos_scores, neg_scores = model(user_idx, pos_item_idx, neg_item_idx)
            
            # BPR Loss: -ln(sigmoid(pos_score - neg_score))
            loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 200 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} Completed | Avg Loss: {avg_loss:.4f} ---")

        # Validation
        if (epoch + 1) % args.eval_every == 0:
            hit_k, ndcg_k, total_val, _ = evaluate_bpr(model, valid_loader, device=device, k=10)
            print(f"Validation @ Epoch {epoch+1} | Hit@10: {hit_k:.4f} | NDCG@10: {ndcg_k:.4f}")

            if hit_k > best_hit:
                best_hit = hit_k
                patience_counter = 0
                save_path = args.checkpoint if args.checkpoint else os.path.join(script_dir, f"bpr_{args.dataset}_best.pth")
                torch.save(model.state_dict(), save_path)
                print(f"🏆 New Best Hit@10! Model saved to {save_path}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{args.patience}")

            if patience_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"\nFinal Best Hit@10: {best_hit:.4f}")
    
    # Test Set Evaluation (if requested)
    if args.test_split:
        print("\n" + "="*50)
        print("🧪 EVALUATING ON TEST SET")
        print("="*50)
        test_file = os.path.join(script_dir, "..", "data", f"test_{args.dataset}_merged.csv")
        if os.path.exists(test_file):
            test_dataset = BPREvalDataset(test_file, user_vocab, item_vocab)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            test_hit, test_ndcg, total_test, _ = evaluate_bpr(model, test_loader, device=device, k=10)
            print(f"\n✅ TEST SET RESULTS:")
            print(f"Hit@10:  {test_hit:.8f}")
            print(f"NDCG@10: {test_ndcg:.8f}")
            print(f"Total users evaluated: {total_test}")
        else:
            print(f"⚠️  Test file not found: {test_file}")

if __name__ == "__main__":
    main()
