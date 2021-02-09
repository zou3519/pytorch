import torch
import torch.nn as nn
import torch.nn.functional as F

# low-level API test
weight = torch.randn(2, 2).expand(3, 2, 2).requires_grad_()
weight = torch.expanded_weights.ExpandedWeight(weight, batch_size=3)

x = torch.randn(3, 2, requires_grad=True)
y = torch.mm(x, weight)
assert y.requires_grad
y = y.sum()
y.backward()
assert weight.grad.shape == weight.weight.shape
assert x.grad.shape == x.shape


# Test on a module
vocab_size = 10

class SampleNet(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 16)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.emb(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def name(self):
        return "SampleNet"

sentence_length = 5
batch_size = 64
x = torch.randint(0, vocab_size, [batch_size, sentence_length])
t = torch.randint(0, 2, [batch_size], dtype=torch.long)

model = SampleNet(vocab_size)
criterion = nn.CrossEntropyLoss()

with model.compute_per_sample_grads(batch_size=batch_size):
    y = model(x)
    loss = criterion(y, t)
    loss.backward()

assert model.fc1.weight.grad_sample.shape == (batch_size, 16, 16)
