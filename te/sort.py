import torch
x = torch.randn(3, 4)
print(x)
sorted, indices = torch.sort(x)
print(sorted)
sorted, indices = torch.sort(x, 0)
print(sorted)