# from pettingzoo.mpe import simple_adversary_v3
import collections
import json
import openpyxl
import numpy as np
from pynput import keyboard

import air_corridor.d3.scenario.D3shapeMove as d3
from air_corridor.tools.visualization import Visualization as vl
from rl_multi_3d_trans.ppo import PPO

key_to_action = {
    'w': np.array([0.5, 0.5, 1]),
    'q': np.array([0.75, 0.5, 1]),
    'e': np.array([0.25, 0.5, 1]),
    'd': np.array([0.0, 0.5, 1]),
    'c': np.array([-0.25, 0.5, 1]),
    'x': np.array([-0.5, 0.5, 1]),
    'z': np.array([-0.75, 0.5, 1]),
    'a': np.array([-1., 0.5, 1]),
    's': np.array([-0.75, 0.5, 1]),
}

# convert [-1,1] to [0,1]
for key, value in key_to_action.items():
    value[0] = value[0] / 2 + 0.5

action_updated = False


def on_press(key):
    global action_updated
    try:
        key_char = str(key.char)
        if key_char in key_to_action:
            for agent in env.agents:
                current_actions[agent] = key_to_action[key_char]
            action_updated = True
    except AttributeError:
        pass  # Ignore special keys


# Start the keyboard listener in a separate thread
listener = keyboard.Listener(on_press=on_press)
listener.start()
env = d3.parallel_env(render_mode="")

loadModel = True
if loadModel:
    # loadFolder = '/mnt/storage/result/d2move_20231219152459_3d_a3/horizon8_batch16_enc6_dt0.333_spaceTrue_level1'
    # modelINdex = '1.5m'
    # net_model='trans'

    loadFolder = '/mnt/storage/result/d2move_20240105163701_3d_fc3/shareTrue_modfc3_horizon8_batch16_enc1_dt1_spaceTrue_level1_capacity1'
    modelINdex = '9.5m'
    net_model = 'fc3'

    with open(f"{loadFolder}/net_params.json", 'r') as json_file:
        kwargs = json.load(json_file)

    kwargs['net_model'] = net_model
    model = PPO(**kwargs)
    model.load(folder=loadFolder,
               global_step=modelINdex)

max_round = 100
ani = vl(max_round, to_base=True)
lst_status=[]


for i in range(max_round):
    print(f"{i}/{max_round}")
    s, infos = env.reset(num_agents=1, level=1, ratio=1,collision_free=True)
    current_actions = {}
    step = 0
    agents = env.agents
    ani.put_data(agents={agent: agent.position for agent in env.agents}, corridors=env.corridors, round=i)
    while env.agents:

        if loadModel:
            s1 = {agent: s[agent]['self'] for agent in env.agents}
            s2 = {agent: s[agent]['other'] for agent in env.agents}
            s1_lst = [state for agent, state in s1.items()]
            s2_lst = [state for agent, state in s2.items()]
            a_lst, logprob_a_lst = model.evaluate(s1_lst, s2_lst)
            actions = {agent: a for agent, a in zip(env.agents, a_lst)}
        else:
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        action_updated = False
        s, rewards, terminations, truncations, infos = env.step(actions)
        ani.put_data(round=i, agents={agent: agent.position for agent in env.agents})
        #print(rewards)

        env.agents = [agent for agent in env.agents if not agent.terminated]

        lst_status+=[agent.status for agent in agents if agent.terminated]
        step += 1
        # print(step)

state_count=collections.Counter(lst_status)
print(state_count)
ani.show_animation()

env.close()
