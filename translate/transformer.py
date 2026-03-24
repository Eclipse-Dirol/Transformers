import torch.nn as nn
import torch
import config

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        d_k: int | None = None
        ):
        super().__init__()
        self.d_model = config.d_model
        self.base_num = config.base_num
        max_len: int = config.max_len
        assert self.d_model>0 and max_len>0 and self.base_num>0, 'only positive value, not negative'
        theta = self.base_num ** (-1/self.d_model)
        freqs = theta ** torch.arange(0, d_k, 2).float()
        positions = torch.arange(0, max_len)
        self.cache = torch.outer(positions, freqs)
        self.cos = self.cache.cos()
        self.sin = self.cache.sin()

    def _get_cos_sin(self, seq_len):
        return self.cos[:seq_len], self.sin[:seq_len]

    def forward(self, x: torch.Tensor):
        # x.shape(batch, seq_len, n_heads, d_k)
        seq_len = x.shape[1]
        cos, sin = self._get_cos_sin(seq_len)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        x1 = x[..., 0::2] * cos - x[..., 1::2] * sin
        x2 = x[..., 0::2] * sin + x[..., 1::2] * cos
        return torch.stack(tensors=[x1, x2], dim=-1).flatten(-2)

class MultiheadAttention_with_RoPE(nn.Module):
    def __init__(self):
        super().__init__()
        d_model: int = config.d_model
        self.n_heads: int = config.n_heads
        assert self.n_heads>0, 'only positive value, not negative'
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.d_k = d_model // self.n_heads
        self.rope = RotaryPositionalEmbedding(self.d_k)

    def forward(
        self,
        x: torch.Tensor,
        is_cross: bool = False,
        encoder_output: torch.Tensor = None,
        mask: torch.Tensor = None 
        ):
        B, S, _ = x.shape

        if is_cross is not True:
            Q = self.W_q(x)
            K = self.W_k(x)
            V = self.W_v(x)
            
        else:
            if encoder_output is None: raise ValueError('encoder_output empty for this request')
            Q = self.W_q(x)
            K = self.W_k(encoder_output)
            V = self.W_v(encoder_output)
        
        S_k = encoder_output.shape[1] if is_cross else S

        Q = Q.view(B, S, self.n_heads, self.d_k)
        K = K.view(B, S_k, self.n_heads, self.d_k)
        V = V.view(B, S_k, self.n_heads, self.d_k)
        # (batch, seq_len, n_heads, d_k)

        Q = self.rope(Q).transpose(1, 2)
        K = self.rope(K).transpose(1, 2)

        score = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)

        if mask is not None:
            mask = mask.expand(B, self.n_heads, S, S_k)
            score = score + mask

        attn = torch.softmax(score, dim=-1)
        V = V.transpose(1, 2)
        output = attn @ V

        output = output.transpose(1, 2).contiguous().view(B, S, -1)
        return output

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        d_model: int = config.d_model
        out_f: int = config.out_f
        dropout: float = config.dropout
        assert out_f>0 and dropout>0, 'only positive value, not negative'
        self.attn = MultiheadAttention_with_RoPE()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, out_f),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_f, d_model)
        )
        self.norm_for_attn = nn.LayerNorm(normalized_shape=d_model)
        self.norm_for_mlp = nn.LayerNorm(normalized_shape=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: list,
        cross_mask = None
        ):
        attn = self.attn(
            x,
            mask = cross_mask
            )
        x = self.norm_for_attn(x + self.dropout(attn))

        mlp_output = self.mlp(x)
        x = self.norm_for_mlp(x + mlp_output)

        return x

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        d_model: int = config.d_model
        vocab_size: int = config.vocab_size
        num_layers: int = config.num_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            EncoderLayer()
            for _ in range(num_layers)
        ])

    def forward(self, x, cross_mask):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(
                x,
                cross_mask = cross_mask
                )
        return x

class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        d_model: int = config.d_model
        out_f: int = config.out_f
        dropout: float = config.dropout
        self.max_len = config.max_len
        self.casual_attn = MultiheadAttention_with_RoPE()
        self.cross_attn = MultiheadAttention_with_RoPE()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, out_f),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_f, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_for_attn = nn.LayerNorm(normalized_shape=d_model)
        self.norm_for_cross_attn = nn.LayerNorm(normalized_shape=d_model)
        self.norm_for_mlp = nn.LayerNorm(normalized_shape=d_model)
        self.casual_mask = self.create_casual_mask()

    def create_casual_mask(self):
        mask = torch.triu(torch.ones(self.max_len, self.max_len),diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf')).unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        x: list,
        encoder_output: torch.Tensor = None,
        cross_mask = None
        ):
        attn = self.casual_attn(x, mask=self.casual_mask)
        x = self.norm_for_attn(x + self.dropout(attn))

        cross_attn=self.cross_attn(x, 
                                   is_cross = True, 
                                   encoder_output = encoder_output,
                                   mask = cross_mask
                                   )
        x = self.norm_for_cross_attn(x + self.dropout(cross_attn))

        mlp_out = self.mlp(x)
        x = self.norm_for_mlp(x + mlp_out)

        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        d_model: int = config.d_model
        vocab_size: int = config.vocab_size
        num_layers: int = config.num_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            DecoderLayer()
            for _ in range(num_layers)
        ])
        self.mlp = nn.Linear(d_model, vocab_size)

    def forward(
        self, 
        x: torch.Tensor, 
        encoder_output: torch.Tensor, 
        cross_mask = None
        ):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(
                x,
                encoder_output = encoder_output,
                cross_mask = cross_mask
                )
        x = self.mlp(x)
        return x

class Transformer(nn.Module):
    def __init__(
        self,
        x: dict
        ):
        super().__init__()
        self.input_ids = torch.tensor(x['input_ids']).unsqueeze(0)
        self.cross_attn_mask = torch.tensor(x['attention_mask']).unsqueeze(0)
        print(self.cross_attn_mask.shape)
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self):
        encoder_output = self.encoder(
            x = self.input_ids,
            cross_mask = self.cross_attn_mask
            )
        x = self.decoder(
            x = self.input_ids,
            encoder_output = encoder_output,
            cross_mask = self.cross_attn_mask
        )
        return x