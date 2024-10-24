import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

def positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return pe

class Tokenizer(nn.Module):
    def __init__(self, input_dim_pad=32, hidden=64, output_dim=128):
        super(Tokenizer, self).__init__()
        self.input_dim_pad = input_dim_pad
        self.fc64 = nn.Linear(32, hidden)

        # TEST: fix padsize to 32
        #self.fc64 = nn.Linear(64, hidden)
        #self.fc128 = nn.Linear(128, hidden)

        # TEST: removed layers, and changed fc64/bn1 to output 128 instead of "hidden"
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc11 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, output_dim)
        self.act = nn.ReLU()
        #self.res_fc = nn.Linear(hidden, output_dim) if hidden != output_dim else None
        self.output_dim = output_dim

    def forward(self, input_tensor, position_index=-1, max_len=3):
        # Padding or truncating the input tensor
        
        #assert input_tensor.size(-1) <= self.input_dim_pad * 2

        # Assert required padding size. However, currently it will always go to the max pad size of 4*initial.
        # Did you mean to check if the input_tensor is larger than the value??
        # for i in range(3):
        #     multplier = 2 ** i
        #     if input_tensor.size(-1) >= self.input_dim_pad * multplier:
        #         padSize = self.input_dim_pad * multplier

        padSize = 32
            
        padded_input = F.pad(input_tensor, (0, padSize - input_tensor.size(-1)))

        # This is needed for 3D inputs (training?)
        input_dims = len(input_tensor.shape)
        if input_dims == 3:
            padded_input = padded_input.view(-1, padSize)

         # Used to check with multiplier for deciding which FC layer to use
        # Changed to check with padsize, i think that's the initial intent
        # To be tested

        # if padSize == 32:
        #     x = self.bn1(self.fc32(padded_input))
        # elif padSize == 64:
        #     x = self.bn1(self.fc64(padded_input))
        # elif padSize == 128:
        #     x = self.bn1(self.fc128(padded_input))

        # No, removing the layers didn't improve training a lot. Stayed stuck at 0.2 difficulty
        x = self.bn1(self.fc64(padded_input))
        x = F.relu(self.fc11(x))
        x = self.bn2(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        if position_index >= 0:
            pos_encoding = positional_encoding(max_len, self.output_dim)
            pos_encoding = pos_encoding[:, position_index, :].to(input_tensor.device)
            x = x + pos_encoding

        if input_dims == 3:
            x = x.view(input_tensor.shape[0], -1, self.output_dim)
        elif input_dims == 2:
            x = x.unsqueeze(1)
        return x


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

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / np.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class FcModule(nn.Module):
    def __init__(self, net_width=256):
        super().__init__()
        self.net_width = net_width
        self.fc1_1 = nn.Linear(net_width, int(net_width / 2))
        self.bn1 = nn.BatchNorm1d(int(net_width / 2))
        self.fc1_2 = nn.Linear(int(net_width / 2), net_width)

    def forward(self, x, times=1):
        # If the length of the input is 2*netwidth, aka concatenated inputs in smallsettransformer
        # Pre-pass through layer 0 to make it 1xnetwidth
        # This means we can pass a residual input instead of cat
        input1 = x
        x = self.fc1_1(x)
        x = self.bn1(x)
        x = F.relu(self.fc1_2(x))
        x = x + input1
        ##############  add-on 11/27
        # if times > 1:
        #     input2 = x
        #     x = self.fc2_1(x)
        #     x = self.bn2(x)
        #     x = F.relu(self.fc2_2(x)) + input2
        ###################3
        return x
