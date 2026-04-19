import os
import pandas as pd
import numpy as np
import xgboost as xgb
import argparse
from tqdm import tqdm
from prepare_xgboost_data import load_metadata

def stratified_sample(df, n_samples=10000, random_state=42):
    """
    Stratified sampling by history length to maintain distribution.
    Ensures sample represents short, medium, and long history users proportionally.
    """
    # Calculate history lengths
    df = df.copy()
    df['hist_len'] = df['history'].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)
    
    # Create stratification bins based on quartiles, handling duplicates
    try:
        df['strata'] = pd.qcut(df['hist_len'], q=4, labels=['short', 'medium-short', 'medium-long', 'long'], duplicates='drop')
    except ValueError:
        # Fallback: use rank-based binning if too many duplicates
        df['rank'] = df['hist_len'].rank(method='first')
        df['strata'] = pd.cut(df['rank'], bins=4, labels=['short', 'medium-short', 'medium-long', 'long'])
        df = df.drop('rank', axis=1)
    
    # Sample proportionally from each stratum
    sampled = df.groupby('strata', group_keys=False).apply(
        lambda x: x.sample(n=max(1, int(n_samples * len(x) / len(df))), random_state=random_state)
    )
    
    # If we got too many, randomly drop excess; if too few, sample more
    if len(sampled) > n_samples:
        sampled = sampled.sample(n=n_samples, random_state=random_state)
    elif len(sampled) < n_samples:
        # Sample more from the full set to reach target
        remaining = df.drop(sampled.index)
        additional = remaining.sample(n=min(n_samples - len(sampled), len(remaining)), random_state=random_state)
        sampled = pd.concat([sampled, additional])
    
    return sampled.drop(['hist_len', 'strata'], axis=1, errors='ignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--sample_size', type=int, default=50000, help='Number of users to sample for evaluation')
    parser.add_argument('--use_stratified', action='store_true', help='Use stratified sampling by history length (maintains distribution)')
    args = parser.parse_args()
    
    dataset_name = args.dataset
    k = 10
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    
    # Correct Metadata Mapping
    meta_mapping = {
        'industrial_and_scientific': 'meta_Industrial_and_Scientific.jsonl.gz',
        'video_games': 'meta_Video_Games.jsonl.gz',
        'cell_phones_and_accessories': 'meta_Cell_Phones_and_Accessories.jsonl.gz'
    }
    
    valid_file = os.path.join(data_dir, f'valid_{dataset_name}_merged.csv')
    meta_file = os.path.join(data_dir, meta_mapping[dataset_name])
    model_path = os.path.join(script_dir, f'xgboost_pure_{dataset_name}_best.json')
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model {model_path} not found.")
        return

    # 1. Load Model
    print(f"Loading XGBoost model for {dataset_name}...")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    # 2. Load Evaluation Data
    print("Loading validation set...")
    valid_df = pd.read_csv(valid_file)
    if len(valid_df) > args.sample_size:
        if args.use_stratified:
            print(f"Stratified sampling {args.sample_size:,} users (out of {len(valid_df):,}) by history length...")
            print("This maintains the same distribution of short/medium/long history users as the full dataset.")
            valid_df = stratified_sample(valid_df, n_samples=args.sample_size, random_state=42)
            # Show distribution
            hist_lens = valid_df['history'].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)
            print(f"Sample history length distribution: min={hist_lens.min()}, max={hist_lens.max()}, mean={hist_lens.mean():.1f}")
        else:
            print(f"Random sampling {args.sample_size:,} users (out of {len(valid_df):,}) for evaluation...")
            valid_df = valid_df.sample(n=args.sample_size, random_state=42)
    
    # 3. Prepare Item Catalog Features
    print("Preparing item catalog features...")
    meta_df = load_metadata(meta_file)
    meta_df = meta_df.drop_duplicates(subset='parent_asin')
    
    meta_df['average_rating'] = pd.to_numeric(meta_df['average_rating'], errors='coerce')
    meta_df['average_rating'] = meta_df['average_rating'].fillna(meta_df['average_rating'].median(numeric_only=True))
    meta_df['rating_number'] = pd.to_numeric(meta_df['rating_number'], errors='coerce').fillna(0)
    meta_df['price'] = pd.to_numeric(meta_df['price'], errors='coerce')
    meta_df['price'] = meta_df['price'].fillna(meta_df['price'].median(numeric_only=True))
    meta_df['main_cat_code'] = meta_df['main_category'].astype('category').cat.codes
    meta_df['store_code'] = meta_df['store'].astype('category').cat.codes
    
    train_file = os.path.join(data_dir, f'train_{dataset_name}_merged.csv')
    train_df = pd.read_csv(train_file)
    popularity_map = train_df['parent_asin'].value_counts().to_dict()
    meta_df['global_popularity'] = meta_df['parent_asin'].map(popularity_map).fillna(0)
    
    feature_names = ['history_len', 'average_rating', 'rating_number', 'price', 'main_cat_code', 'store_code', 'global_popularity']
    item_catalog = meta_df[['parent_asin'] + feature_names[1:]].copy()
    num_items = len(item_catalog)
    catalog_raw = item_catalog[feature_names[1:]].values
    
    # 4. Optimized Batch Evaluation
    print(f"Starting Highly Optimized Evaluation for {len(valid_df)} users...")
    hit_count = 0
    ndcg_sum = 0
    total_users = 0
    
    # Pre-map items for faster lookup
    item_asin_list = item_catalog['parent_asin'].values
    item_to_idx = {asin: idx for idx, asin in enumerate(item_asin_list)}
    
    user_batch_size = 50 # Using 200 users per batch for stability on 110k item catalogs
    for i in tqdm(range(0, len(valid_df), user_batch_size)):
        batch_df = valid_df.iloc[i : i + user_batch_size]
        current_batch_size = len(batch_df)
        
        # Vectorized expansion of items for batch
        # shape: (current_batch_size * num_items, features-1)
        X_batch_items = np.tile(catalog_raw, (current_batch_size, 1))
        
        # Vectorized expansion of history lengths
        h_lens = batch_df['history'].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0).values
        X_batch_user = np.repeat(h_lens, num_items).reshape(-1, 1)
        
        X_batch = np.hstack([X_batch_user, X_batch_items])
        
        # Predict in large batch
        scores_all = model.predict_proba(X_batch)[:, 1]
        scores_matrix = scores_all.reshape(current_batch_size, num_items)
        
        # Hyper-Vectorized Metrics Calculation
        # row-wise argpartition for top-k selection (extremely fast)
        top_k_matrix = np.argpartition(scores_matrix, -k, axis=1)[:, -k:]
        
        # Determine targets
        target_indices = np.array([item_to_idx.get(row['parent_asin'], -1) for _, row in batch_df.iterrows()])
        
        # Valid targets (those existing in item_catalog)
        valid_mask = target_indices != -1
        total_users += np.sum(valid_mask)
        
        # Calculate Hits
        # Check if target_index is in the top_k row-wise
        hits = np.any(top_k_matrix == target_indices[:, None], axis=1)
        hit_count += np.sum(hits & valid_mask)
        
        # Calculate NDCG for hits
        hit_indices = np.where(hits & valid_mask)[0]
        for idx in hit_indices:
            user_scores = scores_matrix[idx]
            target_idx = target_indices[idx]
            
            top_indices = top_k_matrix[idx]
            top_scores = user_scores[top_indices]
            sorted_top = top_indices[np.argsort(top_scores)[::-1]]
            
            rank = np.where(sorted_top == target_idx)[0][0]
            ndcg_sum += 1.0 / np.log2(rank + 2)
                
    hit_at_k = hit_count / total_users if total_users > 0 else 0
    ndcg_at_k = ndcg_sum / total_users
    
    print(f"\n✅ Results for {dataset_name}:")
    print(f"Hit@10:  {hit_at_k:.4f}")
    print(f"NDCG@10: {ndcg_at_k:.4f}")

if __name__ == "__main__":
    main()
