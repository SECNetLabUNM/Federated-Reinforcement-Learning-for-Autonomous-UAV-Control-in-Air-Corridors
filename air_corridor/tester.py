from rl_multi_3d_trans import main
import json
from air_corridor.tools.util import load_init_params

with open('main_params.json','r') as config_file:
    dakt = json.load(config_file)

opt= load_init_params(name='main_params', dir = "./")
kwargs = load_init_params(name='net_params', dir = "./")