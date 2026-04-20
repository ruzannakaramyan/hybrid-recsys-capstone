import torch
import torch.nn as nn

class SASRec(nn.Module):
    def __init__(self, vocab_size, max_seq_len=25, hidden_dim=64, num_heads=2, num_layers=2, dropout_rate=0.2, llm_embeds=None):
        super().__init__()

        self.use_llm = llm_embeds is not None
        if self.use_llm:
            llm_dim = llm_embeds.shape[1]
            self.item_embedding = nn.Embedding.from_pretrained(llm_embeds, freeze=False, padding_idx=0)
            # Better projection: multi-layer with residual-like structure
            self.projection = nn.Sequential(
                nn.Linear(llm_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(128, hidden_dim)
            )
        else:
            self.item_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
            
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)

        self.emb_dropout = nn.Dropout(p=dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout_rate,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        batch_size, seq_len = x.size()

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)

        items = self.item_embedding(x)
        if self.use_llm:
            items = self.projection(items)
            # Scale down to prevent variance explosion from multi-layer projection
            items = items * 0.5

        x = items + self.position_embedding(positions)
        x = self.layer_norm(x)
        x = self.emb_dropout(x)

        mask = (x.abs().sum(dim=-1) == 0)

        out = self.transformer(x, src_key_padding_mask=mask)

        # 🔥 CRITICAL: use LAST non-zero item
        lengths = (x.abs().sum(dim=-1) != 0).sum(dim=1) - 1
        lengths = torch.clamp(lengths, min=0)

        final_state = out[torch.arange(batch_size), lengths]

        if self.use_llm:
            all_projected = self.projection(self.item_embedding.weight) * 0.5
            logits = final_state @ all_projected.T
        else:
            logits = final_state @ self.item_embedding.weight.T

        return logits
