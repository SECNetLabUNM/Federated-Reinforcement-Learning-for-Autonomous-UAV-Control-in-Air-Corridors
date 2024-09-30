import torch.nn as nn
import torch

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define embedding layers for each type of row start
        self.embedding_for_10 = nn.Linear(3, 5)  # For rows starting with [1, 0]
        self.embedding_for_01 = nn.Linear(3, 5)  # For rows starting with [0, 1]

    def eb(self, s2):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        s2 = s2.to(device)

        # Prepare s2_p with the same size and on the same device
        s2_p = torch.zeros((s2.size(0), 5), dtype=torch.float32, device=device)

        # Identify start indices for each pattern
        start_10 = (s2[:, :2] == torch.tensor([1, 0], device=device)).all(dim=1)
        start_01 = (s2[:, :2] == torch.tensor([0, 1], device=device)).all(dim=1)

        # Apply embedding to all matching rows at once
        s2_p[start_10] = self.embedding_for_10(s2[start_10])
        s2_p[start_01] = torch.zeros(5,device=device)

        return s2_p

# Model instantiation and running
model = MyModel().cuda()  # Ensure the model is on GPU
s2 = torch.tensor([[1, 0, 2],
                   [1, 0, 3],
                   [1, 0, 2],
                   [0, 1, 1],
                   [0, 1, 0]], dtype=torch.float32)

# Process the tensor
s2_p = model.eb(s2)
print(s2_p.cpu().detach().numpy())

s3 = torch.tensor([[0, 1, 1],[1, 0, 2],
                   [1, 0, 3],
                   [1, 0, 2],
                   [0, 1, 1],
                   [0, 1, 0]], dtype=torch.float32)

# Process the tensor
s3_p = model.eb(s3)
print(s3_p.cpu().detach().numpy())

print("Processed s2_p:")
print(s2_p)