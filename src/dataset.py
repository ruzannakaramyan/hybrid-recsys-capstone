import os
import torch
from torch.utils.data import Dataset
import pandas as pd

class SequentialDataset(Dataset):
    def __init__(self, csv_file, max_seq_len=25):
        self.data = pd.read_csv(csv_file)
        self.max_seq_len = max_seq_len
        self.item_vocab = self._build_vocab()
        
    def _build_vocab(self):
        # 1. Grab all unique target items
        all_items = set(self.data['parent_asin'].unique())
        
        # 2. Grab all unique history items
        for hist in self.data['history']:
            # We don't need dropna() anymore because you perfectly cleaned the data!
            all_items.update(str(hist).split())
            
        # 3. CRITICAL FIX: Sort alphabetically so IDs never randomly shuffle
        sorted_items = sorted(list(all_items))
        
        # 4. Map ID -> Integer (0 is reserved for padding)
        vocab = {item: idx + 1 for idx, item in enumerate(sorted_items)}
        return vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Target Item
        target_item_str = row['parent_asin']
        target_item_idx = self.item_vocab.get(target_item_str, 0)
        
        # History Timeline
        history_list = str(row['history']).split()
        history_idx = [self.item_vocab.get(item, 0) for item in history_list]
        
        # Pad or Truncate
        if len(history_idx) >= self.max_seq_len:
            seq = history_idx[-self.max_seq_len:]
        else:
            pad_len = self.max_seq_len - len(history_idx)
            seq = [0] * pad_len + history_idx
            
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target_item_idx, dtype=torch.long)

# --- Quick Test ---
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, '..', 'data', 'train_industrial_and_scientific_merged.csv')
    
    print("Building dataset... (this takes a few seconds to sort the vocabulary)")
    dataset = SequentialDataset(file_path, max_seq_len=25)
    
    print(f"\n✅ Success! Total vocabulary size: {len(dataset.item_vocab)}")
    
    # Let's look at the very first user (who now actually has a history!)
    sample_seq, sample_target = dataset[0]
    print(f"User 0 Input Sequence: {sample_seq}")
    print(f"User 0 Target Item:    {sample_target}")