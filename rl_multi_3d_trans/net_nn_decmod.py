# from modules import SAB, PMA
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from air_corridor.tools.util import nan_recoding
from rl_multi_3d_trans.modules import FcModule, Tokenizer, MAB


## trans for neighbors, then no-input-query
## combined with self

class SmallSetTransformer(nn.Module):
    def __init__(self, neighbor_dimension=7, net_width=256, with_position=False, token_query=False, num_enc=4,
                 logger=None):
        super().__init__()

        # Unused stuff
        #self.S = nn.Parameter(torch.Tensor(1, 1, net_width))
        #nn.init.xavier_uniform_(self.S)
        #encoder_layer = nn.TransformerEncoderLayer(d_model=net_width, nhead=4, dim_feedforward=512, batch_first=True)
        #self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_enc)
        
        self.decoder_mab = MAB(net_width, net_width, net_width, num_heads=4, ln=True)
        
        # # pytorch official decoder, having bugs
        # self.decoder_layer = nn.TransformerDecoderLayer(d_model=net_width, nhead=8, activation='gelu',
        #                                                 dim_feedforward=512,
        #                                                 batch_first=True)
        
        # Unused layers
        #self.tk = Tokenizer(output_dim=net_width)
        #self.fc = nn.Linear(net_width, net_width)
        
        self.with_position = with_position
        
        # Unused layers
        #self.token_query = token_query
        #self.fc1 = nn.Linear(2 * net_width, net_width)
        #self.fc2 = nn.Linear(net_width, net_width)
        #self.fc3 = nn.Linear(net_width, net_width)
        
        self.logger = logger
        self.fc_module = FcModule(net_width)

        ########################################
        # self.bn1 = nn.BatchNorm1d(net_width)
        # self.bn2 = nn.BatchNorm1d(net_width)
        # self.fc3 = nn.Linear(net_width, net_width)
        # ########################################

    def forward(self, x, state):
        #x1 = x #self.encoder(x)
        nan_recoding(self.logger, x, 'encoding')

        #query = self.S.repeat(x.size(0), 1, 1)

        #debug try to feed state as query. It works!
        # LIST OF MODS TO ORIGNAL CODE 
        # no encoder, only the MAB decoder
        # The MAB decoder takes the self-state as query, and all observations as key/value
        # The MAB already has a residual connection so no concatenation or addition

        # the MAB decoder splits the vectors along axis 2 (for separating heads), requiring adding a dimension to state
        # which is now [D_net * 1 * 1]
        # Not sure if this splitting is required
        # Is it because how batches are handled during training?
        state2 = state.view(state.size(0),1,state.size(1))

        # Add the self-state output (state) to the "database" to use as K,V 
        # Use self-state output (state) as Q
        x7 = self.decoder_mab(state2, torch.cat([x,state2],dim=1))
        x7 = x7.view(x7.size(0), -1)
        x7 = x7 + state
        
        # Originally, fc_module can dynamically skip a FC layer if the size is not 2*netwidth.
        # instead I've specifically removed the unused layer for less weights
        x8 = self.fc_module(x7)
        return x8


class FixedBranch(nn.Module):
    def __init__(self, input_dimension=11, net_width=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.network(x)
        return x


class BetaActorMulti(nn.Module):
    def __init__(self, s1_dim, s2_dim, action_dim, net_width, shared_layers=None, beta_base=1.0):
        super(BetaActorMulti, self).__init__()
        self.fc1 = nn.Linear(net_width, net_width)
        self.fc2_a = nn.Linear(net_width, int(net_width / 2))
        self.bn1 = nn.BatchNorm1d(int(net_width / 2)) # Seems unused?
        self.fc2_b = nn.Linear(int(net_width / 2), net_width)
        self.alpha_head = nn.Linear(net_width, action_dim)
        self.beta_head = nn.Linear(net_width, action_dim)
        if shared_layers is None:
            self.intput_merge = MergedModel(s1_dim, s2_dim, net_width)
        else:
            self.intput_merge = shared_layers
        self.beta_base = beta_base

    def forward(self, s1, s2):
        merged_input = self.intput_merge(s1, s2)
        x = F.relu(self.fc1(merged_input))
        x_a = F.relu(self.fc2_a(x))
        x_b = F.relu(self.fc2_b(x_a)) + x
        alpha = F.softplus(self.alpha_head(x_b)) + self.beta_base
        beta = F.softplus(self.beta_head(x_b)) + self.beta_base
        return alpha, beta

    def get_dist(self, s1, s2):
        alpha, beta = self.forward(s1, s2)
        dist = Beta(alpha, beta)
        return dist, alpha, beta

    def dist_mode(self, s1, s2):
        alpha, beta = self.forward(s1, s2)
        mode = (alpha-1+1e-5) / (alpha + beta-2+2e-5)
        return mode

class CriticMulti(nn.Module):
    def __init__(self, s1_dim, s2_dim, net_width, shared_layers=None):
        super(CriticMulti, self).__init__()
        self.C4 = nn.Linear(net_width, 1)
        if shared_layers is None:
            self.intput_merge = MergedModel(s1_dim, s2_dim, net_width)
        else:
            self.intput_merge = shared_layers

    def forward(self, s1, s2):
        # ReLU won't work dumbass, you'll only get positive values
        merged_input = self.intput_merge(s1, s2)
        v = self.C4(merged_input)
        return v


class ResBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += identity
        out = F.relu(out)
        return out


class MergedModel(nn.Module):
    def __init__(self, s1_dim, s2_dim, net_width, with_position, token_query, num_enc, logger=None,num_dec = 0):
        super(MergedModel, self).__init__()
        # self.fixed_branch = FixedBranch(s1_dim, net_width)
        self.trans = SmallSetTransformer(net_width, net_width, with_position, token_query, num_enc, logger)
        self.net_width = net_width
        self.tk1 = Tokenizer(output_dim=net_width)
        self.tk2 = Tokenizer(output_dim=net_width)

        self.with_position = with_position
        self.logger = logger

    def forward(self, s1, s2=None):
        s3 = s2
        s1_p = self.tk1(s1)
        s1_p = s1_p.view(s1_p.size(0),s1_p.size(2))
        s3_p = self.tk2(s3)
        x = self.trans(s3_p, state=s1_p)
        nan_recoding(self.logger, x, 'trans_output')
        return x
