import torch

def evaluate_bpr(model, loader, device, k=10):
    """
    Evaluates a BPR model using user-based dot products.
    """
    model.eval()
    total = 0
    hit_count = 0.0
    ndcg_sum = 0.0

    with torch.no_grad():
        for user_indices, target_item_indices in loader:
            # Filter out users not in training vocab (user_idx == -1)
            mask = user_indices != -1
            if not mask.any():
                continue
            
            user_indices = user_indices[mask].to(device)
            target_item_indices = target_item_indices[mask].to(device)
            
            # Get scores for ALL items for these users
            # shape: (batch_size, num_items)
            scores = model.predict_all_items(user_indices)
            
            # Mask out index 0 (padding)
            scores[:, 0] = float("-inf")
            
            # Get Top-K
            topk = torch.topk(scores, k=k, dim=1).indices
            
            for i in range(target_item_indices.size(0)):
                target = target_item_indices[i].item()
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
