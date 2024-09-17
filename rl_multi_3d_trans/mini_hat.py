import argparse
from datetime import datetime
import glob
import logging
import os
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime
from functools import reduce

from tkinter import filedialog

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


import air_corridor.d3.scenario.D3shapeMove as d3
from air_corridor.tools.log_config import setup_logging
from air_corridor.tools.util import save_init_params
from rl_multi_3d_trans.ppo import PPO

kwargs = {
        "state_dim": 26,
        "s2_dim": 22,
        "action_dim": 3,
        "env_with_Dead": True,
        "gamma": 0.99,
        "lambd": 0.95,  # For GAE
        "clip_rate": 0.2,  # 0.2
        "K_epochs": 10,
        "net_width": 256,
        "a_lr": 0.00015,
        "c_lr": 1.5e-05,
        "dist": "Beta",
        "l2_reg": 0.001,  # L2 regulization for Critic
        "a_optim_batch_size": 1536,
        "c_optim_batch_size": 1536,
        "entropy_coef": 0.001,
    "entropy_coef_decay": 0.99,
    "activation": "tanh",
    "share_layer_flag": True,
    "anneal_lr": True,
    "totoal_steps": 10000000.0,
    "with_position": False,
    "token_query": True,
    "num_enc": 1,
    "dir": "minihat",
    "writer": None,
    "logger": None,
    "net_model": "fc",
    "beta_base": 1.0
    }

print(torch.cuda.is_available())

actorpath = filedialog.askopenfilename(title='Select actor parameter file (.pth)',filetypes=[("Model parameters","*.pth")])
criticpath = filedialog.askopenfilename(title='Select critic parameter file (.pth)',filetypes=[("Model parameters","*.pth")])

actorpath = str(actorpath)
criticpath = str(criticpath)

print(actorpath)
print(criticpath)

model = PPO(**kwargs)
model.critic.load_state_dict(torch.load(criticpath))
model.actor.load_state_dict(torch.load(actorpath))


def gen_path(start,end):
# Wayfind between two points using tori and cylinders in a sequence 
    path = []


    return path