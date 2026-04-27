import torch
import pandas as pd
import numpy as np
from dataset import SequentialDataset
from sasrec_model import SASRec
import argparse
import os
import json
from collections import defaultdict
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# --- 1. PROMPT ENGINEERING WITH CHAIN-OF-THOUGHT ---
def create_cot_prompt(user_history, candidate_items, item_metadata):
    history_text = "User's past purchased interactions:\n"
    for i, item in enumerate(user_history[-5:]):  
        title = item_metadata.get(item, {}).get('title', item)
        history_text += f"{i+1}. {title}\n"
    
    candidates_text = "\nCandidate items to recommend to the user next:\n"
    for i, item in enumerate(candidate_items):
        title = item_metadata.get(item, {}).get('title', item)
        category = item_metadata.get(item, {}).get('category', 'Unknown')
        candidates_text += f"ID: {i+1} | Title: {title} | Category: {category}\n"
    
    prompt = f"""You are a senior e-commerce recommendation expert. 
Based on the user's interaction history, analyze their preferences and strictly rank all candidate items from most relevant to least relevant.

{history_text}
{candidates_text}

Task limits:
1. Explain your reasoning in 2-3 sentences based on semantic compatibility, brand loyalty, or clear user needs.
2. Provide the final ranking as a comma-separated list of ONLY the Candidate IDs (1-{len(candidate_items)}) in order of highest relevance to lowest relevance.

Provide exactly this format:
REASONING: <your brief reasoning>
RANKING: <comma separated ID numbers>"""
    return prompt


# --- 2. ASYNC API CLIENT ---
async def fetch_ranking(client, user_id, prompt, num_candidates, semaphore):
    async with semaphore:  # Restrict number of parallel API calls
        for attempt in range(4):
            try:
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a perfect data parsing and recommendation system. You always follow text formatting constraints."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,  # Greedy, deterministic
                    max_tokens=400
                )
                output = response.choices[0].message.content
                
                # Parse the ranking line
                ranking_line = ""
                for line in output.split('\n'):
                    if "RANKING:" in line.upper():
                        ranking_line = line.split(":", 1)[1].strip()
                        break
                if not ranking_line:
                    ranking_line = output  # fallback if they missed the prefix
                    
                # Extract numbers safely
                numbers = [int(n) for n in ranking_line.replace(',', ' ').split() if n.isdigit()]
                valid = [n for n in numbers if 1 <= n <= num_candidates]
                
                # Remove duplicates preserving order
                seen = set()
                ranking = []
                for n in valid:
                    if n not in seen:
                        ranking.append(n)
                        seen.add(n)
                        
                # Identify missing and add them to the bottom
                for i in range(1, num_candidates + 1):
                    if i not in seen:
                        ranking.append(i)
                        
                return user_id, ranking[:num_candidates]
                
            except Exception as e:
                # Sleep on ANY error (rate limits, timeouts, loss of WiFi) before retrying
                if attempt == 3:
                     # On final failed attempt, fallback
                     return user_id, list(range(1, num_candidates + 1))
                await asyncio.sleep(4 * (attempt + 1))
        
        # If all retries failed, fallback
        return user_id, list(range(1, num_candidates + 1))


async def process_api_batch(api_requests, num_candidates, max_concurrency=50):
    client = AsyncOpenAI() # Uses OPENAI_API_KEY env var
    semaphore = asyncio.Semaphore(max_concurrency)
    
    tasks = []
    for req in api_requests:
        tasks.append(fetch_ranking(
            client=client,
            user_id=req['user_id'],
            prompt=req['prompt'],
            num_candidates=num_candidates,
            semaphore=semaphore
        ))
        
    print(f"Submitting {len(tasks)} requests to OpenAI asynchronously...")
    results = await tqdm.gather(*tasks)
    
    # Return mapping of user_id -> ranking_list
    return {user_id: ranking for user_id, ranking in results}


# --- 3. INFRASTRUCTURE ---
def stratified_sample(df, n_samples=5000, random_state=42):
    if n_samples >= len(df):
        return df
    df['history_length'] = df['history'].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)
    df['length_bin'] = pd.qcut(df['history_length'], q=4, labels=False, duplicates='drop')
    bin_counts = df['length_bin'].value_counts().sort_index()
    bin_proportions = bin_counts / len(df)
    samples_per_bin = (bin_proportions * n_samples).round().astype(int)
    
    while samples_per_bin.sum() < n_samples:
        samples_per_bin[samples_per_bin.idxmax()] += 1
    while samples_per_bin.sum() > n_samples:
        samples_per_bin[samples_per_bin[samples_per_bin > 0].idxmin()] -= 1
        
    sampled_dfs = []
    for bin_id, n in samples_per_bin.items():
        bin_df = df[df['length_bin'] == bin_id]
        if len(bin_df) > 0 and n > 0:
            sampled_dfs.append(bin_df.sample(min(n, len(bin_df)), random_state=random_state))
            
    return pd.concat(sampled_dfs).drop(columns=['history_length', 'length_bin'])


def load_item_metadata(dataset, script_dir):
    metadata_path = os.path.join(script_dir, "..", "data", f"metadata_{dataset}.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
            
    metadata = {}
    train_csv = os.path.join(script_dir, "..", "data", f"train_{dataset}_merged.csv")
    if os.path.exists(train_csv):
        df = pd.read_csv(train_csv)
        available_cols = df.columns.tolist()
        
        for _, row in df.iterrows():
            item_id = str(row['parent_asin'])
            if item_id not in metadata:
                title = str(row['title'])[:100] if 'title' in available_cols and pd.notna(row.get('title')) else item_id
                category = str(row['category']) if 'category' in available_cols and pd.notna(row.get('category')) else 'Unknown'
                metadata[item_id] = {'title': title, 'category': category}
    return metadata


# --- 4. MAIN EVALUATION ---
def evaluate_api_reranker(base_model, dataset, device, k=10, rerank_topk=10, max_samples=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_csv = os.path.join(script_dir, "..", "data", f"train_{dataset}_merged.csv")
    test_csv = os.path.join(script_dir, "..", "data", f"test_{dataset}_merged.csv")
    
    train_dataset = SequentialDataset(train_csv, max_seq_len=25)
    item_vocab = train_dataset.item_vocab
    reverse_vocab = {v: k_ for k_, v in item_vocab.items()}
    
    test_df = pd.read_csv(test_csv)
    if max_samples and max_samples < len(test_df):
        print(f"Applying stratified sampling: {max_samples} / {len(test_df)} users")
        test_df = stratified_sample(test_df, n_samples=max_samples)
        
    item_metadata = load_item_metadata(dataset, script_dir)
    user_histories = defaultdict(list)
    for _, row in pd.read_csv(train_csv).iterrows():
        user_histories[str(row['user_id'])].append(str(row['parent_asin']))
        
    api_requests = []
    
    # 1. Base Model Inference (Collect targets & candidates quickly)
    print("Running base SASRec inference...")
    base_eval_data = []  # save data for calculating metrics later
    
    for _, row in test_df.iterrows():
        user_id = str(row['user_id'])
        target_item = str(row['parent_asin'])
        if target_item not in item_vocab:
            continue
            
        history = user_histories.get(user_id, [])
        if not history:
            continue
            
        seq_indices = [item_vocab.get(item, 0) for item in history[-25:]]
        seq = torch.tensor([seq_indices], dtype=torch.long).to(device)
        
        with torch.no_grad():
            logits = base_model(seq)
            logits[:, 0] = float("-inf")
            _, top_indices = torch.topk(logits[0], rerank_topk)
            
        candidate_items = [reverse_vocab[i.item()] for i in top_indices if i.item() != 0]
        
        base_eval_data.append({
            'user_id': user_id,
            'target': target_item,
            'history': history,
            'candidates': candidate_items
        })
        
        # Prepare OpenAI Request Prompt if we have multiple candidates
        if len(candidate_items) > 1:
            api_requests.append({
                'user_id': user_id,
                'prompt': create_cot_prompt(history, candidate_items, item_metadata)
            })
            
    # 2. Async API Reranking Call
    api_rankings = {}
    if os.environ.get("OPENAI_API_KEY"):
        print("\nFiring API Requests asynchronously...")
        loop = asyncio.get_event_loop()
        api_rankings = loop.run_until_complete(process_api_batch(api_requests, rerank_topk, max_concurrency=10))
    else:
        print("\n⚠️ WARNING: NO OPENAI_API_KEY FOUND. Skipping API Reranker, calculating Base only.")

    # 3. Final Metric Calculation
    print("\nCalculating metrics...")
    total = 0
    hit_base = 0.0
    ndcg_base = 0.0
    hit_rerank = 0.0
    ndcg_rerank = 0.0
    
    for data in base_eval_data:
        target = data['target']
        candidates = data['candidates']
        user_id = data['user_id']
        total += 1
        
        # Base Evaluation
        base_top_k = candidates[:k]
        if target in base_top_k:
            hit_base += 1.0
            ndcg_base += 1.0 / np.log2(base_top_k.index(target) + 2)
            
        # Rerank Evaluation
        if user_id in api_rankings and len(candidates) > 1:
            ranking_order = api_rankings[user_id]
            reranked_items = [candidates[i-1] for i in ranking_order]
        else:
            reranked_items = candidates
            
        rerank_top_k = reranked_items[:k]
        if target in rerank_top_k:
            hit_rerank += 1.0
            ndcg_rerank += 1.0 / np.log2(rerank_top_k.index(target) + 2)
            
    print("\n==================================================")
    print(f"Evaluated samples: {total}")
    print(f"[BASE]    Hit@{k}: {hit_base/total:.8f} | NDCG@{k}: {ndcg_base/total:.8f}")
    if api_rankings:
        print(f"[API LLM] Hit@{k}: {hit_rerank/total:.8f} | NDCG@{k}: {ndcg_rerank/total:.8f}")
    print("==================================================")


def main():
    parser = argparse.ArgumentParser(description="Async API LLM Reranker")
    parser.add_argument("--dataset", type=str, default="video_games")
    parser.add_argument("--base_checkpoint", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_seq_len", type=int, default=25)
    parser.add_argument("--rerank_topk", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--use_llm_embeddings", action="store_true")
    args = parser.parse_args()
    
    device = torch.device("cpu")
    print(f"Starting API Reranker for dataset: {args.dataset}")
    
    train_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", f"train_{args.dataset}_merged.csv")
    vocab_size = len(SequentialDataset(train_csv, max_seq_len=25).item_vocab) + 1
    
    llm_embeds = None
    if args.use_llm_embeddings:
        emb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", f"item_embeddings_{args.dataset}.pt")
        if os.path.exists(emb_path):
            llm_embeds = torch.load(emb_path, map_location=device)
            
    base_model = SASRec(
        vocab_size=vocab_size, max_seq_len=args.max_seq_len, hidden_dim=args.hidden_dim,
        num_heads=args.num_heads, num_layers=args.num_layers, dropout_rate=args.dropout,
        llm_embeds=llm_embeds
    ).to(device)
    
    base_model.load_state_dict(torch.load(args.base_checkpoint, map_location=device)['model_state_dict'] if 'model_state_dict' in torch.load(args.base_checkpoint, map_location=device) else torch.load(args.base_checkpoint, map_location=device))
    base_model.eval()
    
    evaluate_api_reranker(base_model, args.dataset, device, k=10, rerank_topk=args.rerank_topk, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
