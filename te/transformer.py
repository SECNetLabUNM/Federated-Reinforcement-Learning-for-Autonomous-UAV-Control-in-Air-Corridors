import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import time
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O
class SmallSetTransformer(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.enc = nn.Sequential(
            SAB(dim_in=1, dim_out=64, num_heads=4),
            SAB(dim_in=64, dim_out=128, num_heads=4),
            SAB(dim_in=128, dim_out=64, num_heads=4),
        )
        self.dec = nn.Sequential(
            PMA(dim=64, num_heads=4, num_seeds=1),
            nn.Linear(in_features=64, out_features=64),
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
        )


    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        x=self.fc(x)
        return x.squeeze(axis=1)

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)
    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


def gen_data(batch_size, max_length=20, test=False):
    length = np.random.randint(5, max_length - 5)
    # length = 7
    x = np.random.randint(1, 100, (batch_size, length + 2))
    xa=np.zeros((batch_size, max_length-length - 2))
    mx=np.sort(x,axis=1)

    y_squared = np.power(mx[:,5], 1/2)

    x, y = np.expand_dims(np.hstack([x,xa]), axis=2), np.expand_dims(y_squared, axis=1)
    return x, y

#
# a=[np.array([[[1],[2]]]),
#    np.array([[[2],[1]]]),
#    np.array([[[0],[1],[2]]]),
#    np.array([[[0],[2],[1]]]),
#    np.array([[[1],[0],[2]]]),
#    np.array([[[1],[2],[0]]]),
#    np.array([[[0],[0],[1],[2]]]),]
# model=SmallSetTransformer()
#
# for v in a:
#     print(model(torch.from_numpy(v).float()))




models = [
    ("Set Transformer", SmallSetTransformer()),
]

def train(model):
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # criterion = nn.L1Loss().cuda()
    # criterion = nn.MSELoss.cuda()
    losses = []
    a=time.time()
    for i in range(2000):
        # x, y = gen_data(batch_size=1, max_length=10)
        # x, y = torch.from_numpy(x).float().cuda(), torch.from_numpy(y).float().cuda()
        x, y = gen_data(batch_size=2 ** 12, max_length=20)
        x = torch.from_numpy(x).float().cuda()

        y = torch.from_numpy(y).float().cuda()
        # model(x_fixed,x_var)
        loss = F.mse_loss(model( x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 5== 0:
            print(loss)
            losses.append(loss.item())
    print(time.time()-a)
    return losses

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(rc={"figure.figsize": (8, 4)}, style="whitegrid")
for _name, _model in models:
    _losses = train(_model)
    plt.plot(_losses, label=_name)
plt.legend()
plt.xlabel("Steps")
plt.ylabel("Mean Absolute Error")
plt.yscale("log")
plt.show()