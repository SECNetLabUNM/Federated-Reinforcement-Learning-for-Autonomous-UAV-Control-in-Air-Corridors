# from pettingzoo.mpe import simple_adversary_v3
import collections
import json
import multiprocessing
import pickle
import time
from itertools import product

import numpy as np
import os
import air_corridor.d3.scenario.D3shapeMove as d3
from air_corridor.tools.util import load_init_params
from rl_federated.ppo_fed import PPO

env = d3.parallel_env(render_mode="")
max_round = 100


def process_combination(combination):
    model_key, num_agents, level, visibility = combination
    # print(num_agents)
    #########################################################
    # print(os.getcwd() + "\n")
    result = '/mnt/storage/result/'
    if model_key == '1-1':
        ### fc10 for circle release, trained with 6 agents, level 20
        loadFolder = f"{result}d2move_20240325092534_new_net/width_128epoch4_index_True_state2_cbfFalse_acc0.3_future2_shareTrue_netfc10_horizon8_batch16_enc2_dec2_spaceTrue_level20_capacity6_beta_base1.0_beta_adaptor_coefficient1.1"
        modelIndex = '20.0m'
        net_model = 'fc10'
        trained_level = 20

    kwargs = load_init_params(name='net_params', dir=loadFolder)
    opt = load_init_params(name='main_params', dir=loadFolder)
    kwargs['net_model'] = net_model
    kwargs['num_agents'] = num_agents
    model = PPO(**kwargs)
    try:
        model.load(folder=loadFolder, global_step=modelIndex)
    except:
        return
    # ani = vl(max_round, to_base=False)
    status = {}
    for z in range(max_round):
        # print(f"{i}/{max_round}")
        '''
        training scenarios can be different from test scenarios, so num_corridor_in_state and corridor_index_awareness
        need to match the setting for UAV during training.
        '''
        s, infos = env.reset(num_agents=num_agents,
                             num_obstacles=4,
                             num_ncfo=3,
                             level=level,
                             dt=opt['dt'],
                             beta_adaptor_coefficient=opt['beta_adaptor_coefficient'],
                             test=True,
                             visibility=visibility)
        step = 0
        # ani.put_data(agents={agent: agent.position for agent in env.agents}, corridors=env.corridors, round=z)
        while env.agents:
            s1 = {agent: s[agent]['self'] for agent in env.agents}
            s2 = {agent: s[agent]['other'] for agent in env.agents}
            a_lst = []
            for agent, state in s1.items():
                s1_lst = [s1[agent]]
                s2_lst = [s2[agent]]
                index = agent.name
                a_ele, _ = model.evaluate(s1_lst, s2_lst, index)
                a_lst.append(a_ele[0])

            actions = {agent: a for agent, a in zip(env.agents, a_lst)}
            s, rewards, terminations, truncations, infos = env.step(actions)
            # ani.put_data(round=z, agents={agent: agent.position for agent in env.agents})
            for agent in env.agents:
                if agent.status != 'Normal' and agent not in status:
                    status[agent] = agent.status
            env.agents = [agent for agent in env.agents if not agent.terminated]
            step += 1
            # print(step)

    state_count = collections.Counter(status.values())
    total_agents = num_agents * max_round
    won_rate = round(state_count['won'] / max(total_agents, 1), 2)
    print(f"{combination}  {trained_level}  {state_count['won']}/{total_agents}  {won_rate}")
    return model_key, num_agents, level, state_count['won'], won_rate, state_count


# ani.show_animation()
# Specify the file name where you want to save the array

def main():
    multiprocessing.set_start_method('spawn', force=True)

    # Your existing setup code
    # model_set = reversed(['1-1', '1-2'])
    # model_set = reversed(['2-1', '2-2', '2-3', '2-4'])
    # model_set = reversed(['3-1', '3-2', '3-3', '3-4', '3-5', '3-6'])
    model_set = reversed(['1-1'])

    level_set = reversed([20])
    agents_dic = {6: 0, 8: 0}
    # agents_dic = {j: i for i, j in enumerate(range(2, 10))}
    # level_set = [14,15]
    # agents_dic = {j: i for i, j in enumerate(range(9, 10))}
    model_dic = {j: i for i, j in enumerate(model_set)}
    level_dic = {j: i for i, j in enumerate(level_set)}
    visibility_dic = {j: i for i, j in enumerate([1 / 3, 1 / 2])}
    combinations = list(product(model_dic.keys(), agents_dic.keys(), level_dic.keys(), visibility_dic.keys()))

    # Limit the number of processes to 4
    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(process_combination, combinations)
        # print(results)

    file_path_json = './plot/test_data.json'
    os.makedirs(os.path.dirname(file_path_json), exist_ok=True)
    with open(file_path_json, 'w') as file:
        json.dump(results, file, indent=4)

    won_matrix = np.zeros([len(model_dic.keys()), len(agents_dic.keys()), len(level_dic.keys())])
    won_rate_matrix = np.zeros([len(model_dic.keys()), len(agents_dic.keys()), len(level_dic.keys())])
    for result in results:
        if result:
            model_key, num_agents, level_key, won_value, won_rate, status_count = result
            won_matrix[model_dic[model_key], agents_dic[num_agents], level_dic[level_key]] = won_value
            won_rate_matrix[model_dic[model_key], agents_dic[num_agents], level_dic[level_key]] = won_rate

    arrays = {'model': model_dic,
              'agent': agents_dic,
              'level': level_dic,
              'won_times': won_matrix,
              'won_rate': won_rate_matrix,
              'status_count': status_count}

    file_path_array = './plot/array.pkl'
    os.makedirs(os.path.dirname(file_path_array), exist_ok=True)
    with open(file_path_array, 'wb') as f:
        pickle.dump(arrays, f)

    # file_name = 'test_data.npy'
    # np.save(file_name, won_record)


if __name__ == "__main__":
    begin = time.time()
    main()
    print(time.time() - begin)
