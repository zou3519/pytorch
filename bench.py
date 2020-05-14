import torch

x = torch.rand(1, 1, 1, 1)
y = torch.rand(1, 1, 1, 1)

for _ in range(500000):
    torch.vmap(torch.add, [0, 0])(x, y)
