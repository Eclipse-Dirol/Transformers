import torch
import torch.nn as nn
import pytest
from Transformer import PositionalEncoder, EncoderBlock, Encoder

def test_positional_encoder_shape():
    d_model = 16
    max_len = 100
    pe = PositionalEncoder(d_model, max_len)
    
    x = torch.randn(2, 10, d_model)  # batch=2, seq=10
    out = pe(x)
    
    assert out.shape == (2, 10, d_model)
    assert not torch.isnan(out).any()

def test_positional_encoder_adds_info():
    d_model = 8
    pe = PositionalEncoder(d_model)
    
    x = torch.zeros(1, 5, d_model)
    out = pe(x)
    
    assert not (out == 0).all()  # должны быть ненулевые значения

def test_encoder_block_shape():
    d_model = 16
    n_heads = 4
    out_f = 32
    dropout = 0.0
    
    block = EncoderBlock(d_model, n_heads, dropout, out_f)
    x = torch.randn(2, 10, d_model)
    out = block(x)
    
    assert out.shape == (2, 10, d_model)
    assert not torch.isnan(out).any()

def test_encoder_block_attention_sums_to_one():
    d_model = 16
    n_heads = 4
    out_f = 32
    dropout = 0.0
    
    block = EncoderBlock(d_model, n_heads, dropout, out_f)
    x = torch.randn(1, 5, d_model)
    
    attn_weights = block.self_attn(x, x, x, need_weights=True)[1]
    
    assert attn_weights.shape == (1, n_heads, 5, 5)
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1)), atol=1e-5)

def test_encoder_shape():
    num_layers = 2
    vocab_size = 26
    d_model = 16
    n_heads = 4
    dropout = 0.0
    out_f = 32
    
    model = Encoder(num_layers, vocab_size, d_model, n_heads, dropout, out_f)
    x = torch.randint(0, vocab_size, (2, 8))  # batch=2, seq=8
    out = model(x)
    
    assert out.shape == (2, 8, vocab_size)
    assert not torch.isnan(out).any()

def test_encoder_training_step():
    num_layers = 2
    vocab_size = 26
    d_model = 16
    n_heads = 4
    dropout = 0.0
    out_f = 32
    
    model = Encoder(num_layers, vocab_size, d_model, n_heads, dropout, out_f)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    x = torch.randint(0, vocab_size, (4, 6))
    target = torch.randint(0, vocab_size, (4, 6))
    
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out.view(-1, vocab_size), target.view(-1))
    loss.backward()
    optimizer.step()
    
    assert loss.item() > 0
    assert not torch.isnan(loss)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
