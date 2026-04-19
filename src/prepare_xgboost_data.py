import os
import pandas as pd
import numpy as np
import json
import gzip
import argparse
from collections import Counter

def load_metadata(jsonl_gz_path):
    print(f"Loading metadata from {jsonl_gz_path}...")
    items = []
    with gzip.open(jsonl_gz_path, 'rt') as f:
        for line in f:
            data = json.loads(line)
            # Keep only columns we need to save memory
            items.append({
                'parent_asin': data.get('parent_asin'),
                'main_category': data.get('main_category'),
                'average_rating': data.get('average_rating'),
                'rating_number': data.get('rating_number'),
                'price': data.get('price'),
                'store': data.get('store')
            })
    return pd.DataFrame(items)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['industrial_and_scientific', 'video_games', 'cell_phones_and_accessories'])
    args = parser.parse_args()
    
    dataset_name = args.dataset
    
    # Mapping for metadata files
    meta_mapping = {
        'industrial_and_scientific': 'meta_Industrial_and_Scientific.jsonl.gz',
        'video_games': 'meta_Video_Games.jsonl.gz',
        'cell_phones_and_accessories': 'meta_Cell_Phones_and_Accessories.jsonl.gz'
    }
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    
    train_file = os.path.join(data_dir, f'train_{dataset_name}_merged.csv')
    meta_file = os.path.join(data_dir, meta_mapping[dataset_name])
    output_file = os.path.join(data_dir, f'xgboost_train_{dataset_name}.parquet')

    if not os.path.exists(train_file):
        print(f"ERROR: Train file {train_file} not found.")
        return

    # 1. Load Original Interactions
    print(f"Processing dataset: {dataset_name}")
    print("Loading interactions...")
    train_df = pd.read_csv(train_file)
    
    # 2. Calculate Global Popularity
    print("Calculating popularity...")
    all_purchases = train_df['parent_asin'].tolist()
    # Also include history to capture 'exposure' popularity
    for hist in train_df['history']:
        if isinstance(hist, str):
            all_purchases.extend(hist.split())
    
    popularity_map = Counter(all_purchases)
    
    # 3. Load and Clean Metadata
    meta_df = load_metadata(meta_file)
    meta_df = meta_df.drop_duplicates(subset='parent_asin')
    
    # Simple Preprocessing
    meta_df['average_rating'] = pd.to_numeric(meta_df['average_rating'], errors='coerce')
    meta_df['rating_number'] = pd.to_numeric(meta_df['rating_number'], errors='coerce')
    meta_df['price'] = pd.to_numeric(meta_df['price'], errors='coerce')
    
    # Imputation using numeric_only median
    meta_df['average_rating'] = meta_df['average_rating'].fillna(meta_df['average_rating'].median(numeric_only=True))
    meta_df['rating_number'] = meta_df['rating_number'].fillna(0)
    meta_df['price'] = meta_df['price'].fillna(meta_df['price'].median(numeric_only=True))
    
    # Encoding Categorical
    meta_df['main_cat_code'] = meta_df['main_category'].astype('category').cat.codes
    meta_df['store_code'] = meta_df['store'].astype('category').cat.codes
    
    # Add popularity
    meta_df['global_popularity'] = meta_df['parent_asin'].map(popularity_map).fillna(0)
    
    # 4. Stratified Negative Sampling (1:9)
    print("Performing stratified sampling (1:9 ratio)...")
    all_item_ids = meta_df['parent_asin'].unique()
    
    final_data = []
    
    # Group by user to find user_positives efficiently
    user_positives = train_df.groupby('user_id')['parent_asin'].apply(set).to_dict()
    
    # Iterate through training rows
    for i, row in train_df.iterrows():
        user_id = row['user_id']
        pos_item = row['parent_asin']
        history_list = str(row['history']).split()
        history_len = len(history_list)
        
        # Positive Sample
        final_data.append({
            'user_id': user_id,
            'parent_asin': pos_item,
            'history_len': history_len,
            'label': 1
        })
        
        # Negative Samples (9 per positive)
        count = 0
        while count < 9:
            neg_item = np.random.choice(all_item_ids)
            if neg_item != pos_item and neg_item not in user_positives[user_id]:
                final_data.append({
                    'user_id': user_id,
                    'parent_asin': neg_item,
                    'history_len': history_len,
                    'label': 0
                })
                count += 1
        
        if i % 100000 == 0:
            print(f"Processed {i}/{len(train_df)} pos items...")

    samples_df = pd.DataFrame(final_data)
    
    # 5. Final Join
    print("Joining features...")
    features_to_keep = ['parent_asin', 'average_rating', 'rating_number', 'price', 'main_cat_code', 'store_code', 'global_popularity']
    final_df = samples_df.merge(meta_df[features_to_keep], on='parent_asin', how='left')
    
    # Final cleanup of any missing metadata
    final_df = final_df.fillna(0)
    
    print(f"Saving {len(final_df)} rows to {output_file}...")
    final_df.to_parquet(output_file, index=False)
    print(f"✅ Done with {dataset_name}!")

if __name__ == "__main__":
    main()
