import torch.nn as nn
import torch

class _PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class _EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout, out_f):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, out_f),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_f, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.self_attn(x,x,x)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x

class _Encoder(nn.Module):
    def __init__(self, num_layers, vocab_size, d_model, n_heads, dropout, out_f):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = _PositionalEncoder(d_model)
        self.layers = nn.ModuleList([
            _EncoderBlock(d_model, n_heads, dropout, out_f)
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)

class Transformer(nn.Module):
    def __init__(self, num_layers, vocab_size, d_model, n_heads, dropout, out_f):
        super().__init__()
        self.encoder = _Encoder(num_layers, vocab_size, d_model, n_heads, dropout, out_f)

    def forward(self, x):
        return self.encoder(x)