# from pettingzoo.mpe import simple_adversary_v3
import collections
import json

import air_corridor.d3.scenario.D3shapeMove as d3
import numpy as np
from air_corridor.tools.util import load_init_params
from air_corridor.tools.visualization1 import Visualization as vl
from pynput import keyboard
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
max_round = 100
# loadFolder = '/mnt/storage/result/d2move_20231219152459_3d_a3/horizon8_batch16_enc6_dt0.333_spaceTrue_level1'
# modelINdex = '1.5m'
# net_model='trans'
# init = True
# loadFolder = '/home/kun/PycharmProjects/air-corridor/rl_multi_3d_trans/d2move_1704839543_0/0'
# modelINdex = '1.75m'
# net_model = 'fc2'
# init = False
# loadFolder = '/mnt/storage/result/d2move_20240109213135_b_two_pieces/shareTrue_modfc2_horizon8_batch16_enc3_dt1_spaceTrue_level3_capacity5_beta_base1.0_beta_adaptor_coefficient1.1'
# modelINdex = '2.0m'
# net_model = 'fc2'

# init = False
# loadFolder = '/mnt/storage/result/d2move_20240112111034_b_one_pieces/future1_shareTrue_modtran2_1_3_horizon8_batch16_enc4_dt1_spaceTrue_level2_capacity5_beta_base1.0_beta_adaptor_coefficient1.1'
# modelINdex = '2.0m'
# net_model = 'tran2_1_3'

# init = False
# loadFolder = '/mnt/storage/result/d2move_20240113173827_b_one_pieces/acc0.5_future1_shareTrue_modtran2_1_3_horizon8_batch16_enc3_dt0.5_spaceTrue_level2_capacity5_beta_base1.0_beta_adaptor_coefficient1.1'
# modelINdex = '3.0m'
# net_model = 'tran2_1_3'

loadModel = True
model_set = list(range(3))
agents_set = list(range(2, 9))
level_set = [10, 11, 12, 13, 14]
won_record = np.zeros([len(model_set), len(agents_set), len(level_set)])
for i0, _ in enumerate(model_set):

    #########################################################
    if i0 == 0:
        # one-piece state; one-piece training

        loadFolder = '/mnt/storage/result/d2move_20240119162023_b_one_pieces_good_performance_for_one_piece/ratio0.8_future1_shareTrue_modfc2_horizon8_batch16_enc3_dt1_spaceTrue_level2_capacity5_beta_base1.0_beta_adaptor_coefficient1.1'
        modelINdex = '9.5m'
        net_model = 'fc2'
        # Counter({'won': 217, 'breached_wall': 148, 'breached': 93, 'collided': 32, 'breached_rad_wall': 10})
    elif i0 == 1:
        # one-piece state; two-piece training

        loadFolder = '/mnt/storage/result/d2move_20240119162547_b_one_pieces/acc0.3_future1_shareTrue_modtran2_1_3_horizon8_batch16_enc2_dt1_spaceTrue_level3_capacity5_beta_base1.0_beta_adaptor_coefficient1.1'
        modelINdex = '7.0m'
        net_model = 'tran2_1_3'
        # 5:Counter({'won': 319, 'breached_wall': 113, 'breached': 51, 'breached_rad_wall': 13, 'collided': 4})
    elif i0 == 2:
        # two-piece state; two-piece training

        loadFolder = '/mnt/storage/result/d2move_20240119164655_b_two_pieces/future2_shareTrue_modfc2_horizon8_batch16_enc4_dt1_spaceTrue_level3_capacity5_beta_base1.0_beta_adaptor_coefficient1.1'
        modelINdex = '6.5m'
        net_model = 'fc2'
        # 5: Counter({'won': 256, 'breached_wall': 162, 'breached': 46, 'collided': 24, 'breached_rad_wall': 12})
        # 4: Counter({'won': 234, 'breached_wall': 107, 'breached': 42, 'collided': 12, 'breached_rad_wall': 5})
        # 3: Counter({'won': 201, 'breached_wall': 59, 'breached': 29, 'collided': 6, 'breached_rad_wall': 5})
        # 2: Counter({'won': 125, 'breached_wall': 42, 'breached': 26, 'breached_rad_wall': 7})
    #########################################################

    kwargs = load_init_params(name='net_params', dir=loadFolder)
    opt = load_init_params(name='main_params', dir=loadFolder)

    with open(f"{loadFolder}/net_params.json", 'r') as json_file:
        kwargs1 = json.load(json_file)
        # opt=json.load

    kwargs['net_model'] = net_model
    model = PPO(**kwargs)

    model.load(folder=loadFolder, global_step=modelINdex)
    for i1, num_agents in enumerate(agents_set):
        for i2, level in enumerate(level_set):

            ani = vl(max_round, to_base=False)
            status = {}
            for z in range(max_round):
                # print(f"{i}/{max_round}")
                '''
                training scenarios can be different from test scenarios, so num_corridor_in_state and corridor_index_awareness
                need to match the setting for UAV during training.
                '''
                s, infos = env.reset(num_agents=num_agents,
                                     level=level,
                                     dt=opt['dt'],
                                     num_corridor_in_state=opt['num_corridor_in_state'],
                                     corridor_index_awareness=opt['corridor_index_awareness'],
                                     test=True)
                current_actions = {}
                step = 0
                agents = env.agents
                ani.put_data(agents={agent: agent.position for agent in env.agents}, corridors=env.corridors, round=z)
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
                    ani.put_data(round=z, agents={agent: agent.position for agent in env.agents})
                    # print(rewards)

                    for agent in env.agents:
                        if agent.status != 'Normal' and agent not in status:
                            status[agent] = agent.status
                    env.agents = [agent for agent in env.agents if not agent.terminated]
                    step += 1
                    # print(step)

            state_count = collections.Counter(status.values())
            env.close()
            unified_state = {key: value / num_agents / max_round for key, value in state_count.items()}
            print(i0, i1, i2)
            print(state_count)
            print(unified_state)
            won_record[i0, i1, i2] = unified_state['won']
# ani.show_animation()
# Specify the file name where you want to save the array
file_name = 'my_array.npy'

# Save the array to the file
np.save(file_name, won_record)
