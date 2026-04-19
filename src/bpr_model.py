import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
from collections import defaultdict
import random

def build_user_vocab(csv_path):
    data = pd.read_csv(csv_path)
    all_users = sorted(data['user_id'].astype(str).unique())
    return {u: idx for idx, u in enumerate(all_users)}

class BPRMF(nn.Module):
    def __init__(self, num_users, num_items, hidden_dim=64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, hidden_dim)
        self.item_embedding = nn.Embedding(num_items, hidden_dim, padding_idx=0)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_idx, pos_item_idx, neg_item_idx):
        u_emb = self.user_embedding(user_idx)
        pos_i_emb = self.item_embedding(pos_item_idx)
        neg_i_emb = self.item_embedding(neg_item_idx)
        pos_scores = (u_emb * pos_i_emb).sum(dim=1)
        neg_scores = (u_emb * neg_i_emb).sum(dim=1)
        return pos_scores, neg_scores
        
    def predict_all_items(self, user_idx):
        u_emb = self.user_embedding(user_idx)
        return u_emb @ self.item_embedding.weight.T

class BPRDataset(Dataset):
    def __init__(self, csv_file, user_vocab, item_vocab):
        self.data = pd.read_csv(csv_file)
        self.user_vocab = user_vocab
        self.item_vocab = item_vocab
        self.max_item_id = len(item_vocab)
        self.pairs, self.user_positives = self._build_dataset()
        
    def _build_dataset(self):
        user_positives = defaultdict(set)
        for idx, row in self.data.iterrows():
            user_str = str(row['user_id'])
            user_idx = self.user_vocab.get(user_str)
            if user_idx is None: continue
            
            target_idx = self.item_vocab.get(str(row['parent_asin']), 0)
            if target_idx != 0: user_positives[user_idx].add(target_idx)
                
            hist_str = str(row['history'])
            if hist_str and hist_str.lower() != 'nan':
                for item in hist_str.split():
                    item_idx = self.item_vocab.get(item, 0)
                    if item_idx != 0: user_positives[user_idx].add(item_idx)
                        
        pairs = [(u, i) for u, items in user_positives.items() for i in items]
        return pairs, user_positives

    def __len__(self): return len(self.pairs)
        
    def __getitem__(self, idx):
        user_idx, pos_item_idx = self.pairs[idx]
        while True:
            neg_item_idx = random.randint(1, self.max_item_id)
            if neg_item_idx not in self.user_positives[user_idx]: break
        return (torch.tensor(user_idx, dtype=torch.long),
                torch.tensor(pos_item_idx, dtype=torch.long),
                torch.tensor(neg_item_idx, dtype=torch.long))

class BPREvalDataset(Dataset):
    def __init__(self, csv_file, user_vocab, item_vocab):
        self.data = pd.read_csv(csv_file)
        self.user_vocab = user_vocab
        self.item_vocab = item_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        user_str = str(row['user_id'])
        item_str = str(row['parent_asin'])
        
        user_idx = self.user_vocab.get(user_str, -1)
        target_item_idx = self.item_vocab.get(item_str, 0)
        
        return torch.tensor(user_idx, dtype=torch.long), torch.tensor(target_item_idx, dtype=torch.long)
