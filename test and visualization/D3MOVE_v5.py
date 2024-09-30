# from pettingzoo.mpe import simple_adversary_v3
import collections
import json

import air_corridor.d3.scenario.D3shapeMove as d3
import numpy as np
from air_corridor.tools.util import load_init_params
from air_corridor.tools.visualization import Visualization as vl
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
max_round = 10

loadModel = True
model_set = list(range(1))
agents_set = list(range(5, 6))
# level_set = [10, 11, 12, 13, 14]
level_set = [14]
won_record = np.zeros([len(model_set), len(agents_set), len(level_set)])
for i0, _ in enumerate(model_set):

    result = '/mnt/storage/result/'

    # ### fc10 for circle release, trained with 6 agents, level 20
    # loadFolder = '/mnt/storage/result/d2move_20240325092534_new_net/width_128epoch4_index_True_state2_cbfFalse_acc0.3_future2_shareTrue_netfc10_horizon8_batch16_enc2_dec2_spaceTrue_level20_capacity6_beta_base1.0_beta_adaptor_coefficient1.1'
    # modelINdex = '19.5m'
    # net_model = 'fc10'
    # trained_level = 20

    ### fc10 for grid release, trained with 6 agents, level 20
    # loadFolder = '/mnt/storage/result/d2move_20240406220318_new_net/width_128epoch4_index_True_state2_corindexTrue_acc0.3_future2_shareTrue_netfc10_horizon8_batch16_enc2_dec2_spaceTrue_level20_capacity6_beta_base1.0_beta_adaptor_coefficient1.1'
    # modelINdex = '19.5m'
    # net_model = 'fc10'
    # trained_level = 20



    ### fc10e grid release, trained with 6 agents, level 19
    loadFolder = f"{result}d2move_20240417112207_new_net/dec_3_width_128epoch4_corindexTrue_netdec_horizon8_level19_capacity6_beta_adaptor1.1"
    modelIndex = '19.0m'
    net_model = 'dec'
    trained_level = 19

    kwargs = load_init_params(name='net_params', dir=loadFolder)
    opt = load_init_params(name='main_params', dir=loadFolder)

    with open(f"{loadFolder}/net_params.json", 'r') as json_file:
        kwargs1 = json.load(json_file)
        # opt=json.load

    kwargs['net_model'] = net_model
    model = PPO(**kwargs)

    model.load(folder=loadFolder, global_step=modelIndex)
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
                                     beta_adaptor_coefficient=opt['beta_adaptor_coefficient'],
                                     test=True)
                current_actions = {}
                step = 0
                agents = env.agents
                ani.put_data(agents={agent: agent.position for agent in env.agents},
                             corridors=env.corridors,
                             round=z)
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
            try:
                won_record[i0, i1, i2] = unified_state['won']
            except:
                pass
ani.show_animation()
# Specify the file name where you want to save the array
file_name = 'my_array2.npy'

# Save the array to the file
np.save(file_name, won_record)
