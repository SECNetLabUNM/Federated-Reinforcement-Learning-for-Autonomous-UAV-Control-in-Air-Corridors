from collections import OrderedDict

import torch

global_step = 21.25e6
global_step_str = str(global_step / 1000000) + 'm'



num_models = 3
for role in ['actor', 'critic']:
    avg_state_dict = {}  # Initialize avg_state_dict for each role

    for i in range(num_models):
        saved_weights = torch.load(f"ppo_{role}_{global_step_str}_c{i}.pth")
        if not avg_state_dict:  # If avg_state_dict is empty, initialize it
            avg_state_dict = {key: value.clone() for key, value in saved_weights.items()}
        else:
            for key, value in saved_weights.items():
                avg_state_dict[key] += value


    for key, value in avg_state_dict.items():
        avg_state_dict[key] = avg_state_dict[key] /num_models
    torch.save(avg_state_dict, f"ppo_{role}_{global_step}.pth")
