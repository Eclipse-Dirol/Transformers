import torch
from Transformer import Transformer
from torch.optim import Adam
import torch.nn as nn

num_layers = 3
vocab_size = 26
d_model = 64
n_heads = 4
dropout = 0.1
out_f = 256

class work_with_model():
    def __init__(self):
        self.model = Transformer(num_layers=num_layers, vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, dropout=dropout, out_f=out_f)

    def test(self, request):
        model.model.load_state_dict(torch.load('model_weights.pth'))
        char_to_idx = {chr(i + 97): i for i in range(26)}
        idx_to_char = {i: chr(i + 97) for i in range(26)}

        def encode(text):
            return torch.tensor([char_to_idx[c.lower()] for c in text])
        def decode(tokens):
            return ''.join([idx_to_char[i.item()] for i in tokens])

        x = encode(request).unsqueeze(0)
        with torch.no_grad():
            out = self.model(x)
            pred = out.argmax(dim=-1)

        answer = decode(pred[0])
        print(answer)

    def train(self, epoch):
        self.model.train()
        opt = Adam(self.model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        for step in range(epoch):
            x = torch.randint(0,26,(64,8))
            y = torch.roll(x, shifts=1, dims=1)

            out = self.model(x)
            loss = loss_fn(out.view(-1, 26), y.view(-1))

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                preds = out.argmax(dim=-1)
                acc = (preds == y).float().mean()

            print(f"Epoch {step} | Loss: {loss:.4f} | Acc: {acc:.4f}")
        torch.save(model.model.state_dict(), 'model_weights.pth')

model = work_with_model()
request = input('write you request\n')
model.test(request=request)