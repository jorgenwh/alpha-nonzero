import torch


x = torch.tensor([0.1, 0, 0, 0.3, 0, 0, 0.6, 0, 0, 0], dtype=torch.float32)

selections = [0 for _ in range(10)]
for _ in range(10000):
    s = torch.multinomial(x, 1)
    selections[s] += 1

print([s / sum(selections) for s in selections])
