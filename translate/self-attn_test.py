import torch.nn as nn
import torch
import config
from tokenizer import Tokenizer

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
        print(x.shape)
        seq_len = x.shape[1]
        print(seq_len)
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
        print(f'first: {x.shape}')
        B, S, _ = x.shape

        if is_cross is not True:
            Q = self.W_q(x)
            K = self.W_k(x)
            V = self.W_v(x)

            print(f'second: {Q.shape} {K.shape} {V.shape}')
            print(Q)
            print(K)
            
        else:
            if encoder_output is None: raise ValueError('encoder_output empty for this request')
            Q = self.W_q(x)
            K = self.W_k(encoder_output)
            V = self.W_v(encoder_output)
        
        print(f'third: {Q.shape} {K.shape} {V.shape}')
        
        S_k = encoder_output.shape[1] if is_cross else S

        Q = Q.view(B, S, self.n_heads, self.d_k)
        K = K.view(B, S_k, self.n_heads, self.d_k)
        V = V.view(B, S_k, self.n_heads, self.d_k)
        # (batch, seq_len, n_heads, d_k)
        print(f'four: {Q.shape} {K.shape} {V.shape}')

        Q = self.rope(Q)
        K = self.rope(K)
        print(f'five: {Q.shape} {K.shape}')
        print(Q)
        print(K)

        score = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)

        if mask is not None:
            score = score + mask

        attn = torch.softmax(score, dim=-1)
        output = attn @ V

        output = output.contiguous().view(B, S, -1)
        return output

token = Tokenizer()
text = 'Hello'
data = token.encode(text)
input_ids = torch.tensor(data['input_ids'])
cross_attn_mask = data['attention_mask']
embed = nn.Embedding(config.vocab_size, config.d_model)
x = embed(input_ids).unsqueeze(0)
print(x)
print(f'not MhA: {x.shape}')
MhA = MultiheadAttention_with_RoPE()
answer = MhA(x)
print(answer)