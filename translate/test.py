from tokenizer import Tokenizer
import config
import torch
import torch.nn as nn
# import torch

token = Tokenizer()
text = 'Hello'
data = token.encode(text)
print(data.items())
input_ids = data['input_ids']
cross_attn_mask = data['attention_mask']
print_text = token.decode(input_ids)
print(input_ids)
print(cross_attn_mask)
print(print_text)
print(len(input_ids), len(cross_attn_mask), len(print_text))

# seq_len = 4
# mask = torch.triu(torch.ones(seq_len, seq_len),diagonal=1)
# mask = mask.masked_fill(mask == 1, float('-inf'))
# print(mask)