import pandas as pd
import os

def check():
    dataset = "cell_phones_and_accessories"
    eval_csv = f"data/valid_{dataset}_merged.csv"
    if not os.path.exists(eval_csv):
        print("Data not found")
        return
        
    df = pd.read_csv(eval_csv)
    df['seq_len'] = df['history'].apply(lambda x: len(str(x).split()))
    
    print(f"Original Mean Seq Len: {df['seq_len'].mean():.2f}")
    print(f"Original Median Seq Len: {df['seq_len'].median():.2f}")
    
    # Simulate our stratification
    sample_users = 50000
    num_bins = 5
    df['bin'] = pd.qcut(df['seq_len'], q=num_bins, labels=False, duplicates='drop')
    sample = df.groupby('bin', group_keys=False).apply(
        lambda x: x.sample(n=int(len(x) * sample_users / len(df)), random_state=42)
    )
    
    print(f"Sample Mean Seq Len: {sample['seq_len'].mean():.2f}")
    print(f"Sample Median Seq Len: {sample['seq_len'].median():.2f}")
    print(f"Sample Size: {len(sample)}")

if __name__ == "__main__":
    check()
