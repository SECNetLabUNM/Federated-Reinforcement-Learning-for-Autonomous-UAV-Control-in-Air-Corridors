# from pettingzoo.mpe import simple_adversary_v3
# file setup for federated test

import collections
import json
import multiprocessing
import time
from itertools import product

import air_corridor.d3.scenario.D3shapeMove as d3
from air_corridor.tools.util import load_init_params
from rl_multi_3d_trans.ppo import PPO

env = d3.parallel_env(render_mode="")
max_round = 1000


def process_combination(combination):
    model_key, num_agents, level, turbulence_variance, visibility = combination
    # print(num_agents)
    #########################################################
    # print(os.getcwd() + "\n")
    result = '/mnt/storage/result/'
    if model_key == '1-1':
        ### fc10 for circle release, trained with 6 agents, level 20
        loadFolder = f"{result}d2move_20240325092534_new_net/width_128epoch4_index_True_state2_cbfFalse_acc0.3_future2_shareTrue_netfc10_horizon8_batch16_enc2_dec2_spaceTrue_level20_capacity6_beta_base1.0_beta_adaptor_coefficient1.1"
        modelIndex = '19.5m'
        net_model = 'fc10'
        trained_level = 20
    if model_key == '1-2':
        ### fc10 for grid release, trained with 6 agents, level 20
        loadFolder = f"{result}d2move_20240416110709_new_net/width_128epoch4_index_True_state2_corindexTrue_acc0.3_future2_shareTrue_netfc10_horizon8_batch16_enc2_dec2_spaceTrue_level20_capacity12_beta_base1.0_beta_adaptor_coefficient1.1"
        modelIndex = '21.5m'
        net_model = 'fc10'
        trained_level = 20
    # if model_key == '2-1':
    #     ### fc10e grid release, trained with 6 agents, level 19
    #     loadFolder = f"{result}d2move_20240417112201_new_net/width_128epoch4_corindexTrue_netfc10_3e_horizon8_level19_capacity6_beta_adaptor1.1"
    #     modelIndex = '16.0m'
    #     net_model = 'fc10_3e'
    #     trained_level = 19
    # if model_key == '2-2':
    #     ### fc10e grid release, trained with 6 agents, level 19
    #     loadFolder = f"{result}d2move_20240417112201_new_net/width_128epoch4_corindexFalse_netfc10_3e_horizon8_level19_capacity6_beta_adaptor1.1"
    #     modelIndex = '16.25m'
    #     net_model = 'fc10_3e'
    #     trained_level = 19
    # if model_key == '2-3':
    #     ### fc10e grid release, trained with 6 agents, level 19
    #     loadFolder = f"{result}d2move_20240417112207_new_net/dec_3_width_128epoch4_corindexTrue_netdec_horizon8_level19_capacity6_beta_adaptor1.1"
    #     modelIndex = '19.0m'
    #     net_model = 'dec'
    #     trained_level = 19
    # if model_key == '2-4':
    #     ### fc10e grid release, trained with 6 agents, level 19
    #     loadFolder = f"{result}d2move_20240417112207_new_net/dec_3_width_128epoch4_corindexFalse_netdec_horizon8_level19_capacity6_beta_adaptor1.1"
    #     modelIndex = '19.5m'
    #     net_model = 'dec'
    #     trained_level = 19
    if model_key == '3-1':
        ### dec grid release, trained with 4-12 agents, level 19
        loadFolder = f"{result}d2move_20240422072449_new_net/width_128epoch4_corindexFalse_netdec_horizon8_level19_capacity4_beta_adaptor1.1"
        modelIndex = '8.75m'
        net_model = 'dec'
        trained_level = 19
    if model_key == '3-2':
        ### dec grid release, trained with 4-12 agents, level 19
        loadFolder = f"{result}d2move_20240422072449_new_net/width_128epoch4_corindexTrue_netdec_horizon8_level19_capacity4_beta_adaptor1.1"
        modelIndex = '15.25m'
        net_model = 'dec'
        trained_level = 19
    if model_key == '3-3':
        ### dec grid release, trained with 4-12 agents, level 19
        loadFolder = f"{result}d2move_20240422072449_new_net/width_128epoch4_corindexFalse_netfc10_3e_horizon8_level19_capacity4_beta_adaptor1.1"
        modelIndex = '14.75m'
        net_model = 'fc10_3e'
        trained_level = 19
    if model_key == '3-4':
        ### dec grid release, trained with 4-12 agents, level 19
        loadFolder = f"{result}d2move_20240422072449_new_net/width_128epoch4_corindexTrue_netfc10_3e_horizon8_level19_capacity4_beta_adaptor1.1"
        modelIndex = '14.25m'
        net_model = 'fc10_3e'
        trained_level = 19
    if model_key == '3-5':
        ### dec grid release, trained with 4-12 agents, level 19
        loadFolder = f"{result}d2move_20240422072449_new_net/width_128epoch4_corindexFalse_netfc12_horizon8_level19_capacity4_beta_adaptor1.1"
        modelIndex = '17.25m'
        net_model = 'fc12'
        trained_level = 19
    if model_key == '3-6':
        ### dec grid release, trained with 4-12 agents, level 19
        loadFolder = f"{result}d2move_20240422072449_new_net/width_128epoch4_corindexTrue_netfc12_horizon8_level19_capacity4_beta_adaptor1.1"
        modelIndex = '17.0m'
        net_model = 'fc12'
        trained_level = 19
    if model_key == '4-1':
        ### dec grid release, trained with 4-12 agents, level 19
        loadFolder = f"{result}d2move_20240325092534_new_net/width_128epoch4_index_True_state2_cbfFalse_acc0.3_future2_shareTrue_netfc10_horizon8_batch16_enc2_dec2_spaceTrue_level20_capacity6_beta_base1.0_beta_adaptor_coefficient1.1"
        modelIndex = '20.0m'
        net_model = 'fc10'
        trained_level = 20
    if model_key == '4-2':
        ### dec grid release, trained with 4-12 agents, level 19
        loadFolder = f"{result}d2move_20240325092534_new_net/width_128epoch4_index_True_state2_cbfFalse_acc0.3_future2_shareTrue_netfc10_horizon8_batch16_enc2_dec2_spaceTrue_level20_capacity6_beta_base1.0_beta_adaptor_coefficient1.1"
        modelIndex = '20.0m'
        net_model = 'fc10'
        trained_level = 20
    if model_key == '4-5':
        ### dec grid release, trained with 4-12 agents, level 19
        loadFolder = f"{result}d2move_1715744847_new_net/9_8_16_1_1.5_21_fc10_1e-6_1e-5_0.1_False_2_all_8"
        modelIndex = '20.5m'
        net_model = 'fc10'
        trained_level = 20
    if model_key == '4-6':
        ### dec grid release, trained with 4-12 agents, level 19
        loadFolder = f"{result}d2move_1715189556_new_net/9_8_16_4.0_21_fc10_1e-6_1e-5_0.1_False_2_all_8"
        modelIndex = '21.75m'
        net_model = 'fc10'
        trained_level = 20
    if model_key == '4-7':
        ### dec grid release, trained with 9 agents, turbulence 0.075
        loadFolder = f"{result}d2move_1715189557_new_net/9_8_16_1.5_21_fc10_1e-6_1e-5_0.075_False_2_all_8"
        modelIndex = '23.75m'
        net_model = 'fc10'
        trained_level = 20
    if model_key == '4-8':
        ### dec grid release, trained with 9 agents, turbulence 0.075
        loadFolder = f"{result}d2move_1715189556_new_net/9_8_16_4.0_21_fc10_1e-6_1e-5_0.075_False_2_all_8"
        modelIndex = '23.75m'
        net_model = 'fc10'
        trained_level = 20

    kwargs = load_init_params(name='net_params', dir=loadFolder)
    opt = load_init_params(name='main_params', dir=loadFolder)
    kwargs['net_model'] = net_model
    model = PPO(**kwargs)
    try:
        model.load(folder=loadFolder, global_step=modelIndex)
    except:
        return
    # ani = vl(max_round, to_base=False)
    status = {}
    accumulated_won_speed = 0
    accumulated_won_time=0
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
                             turbulence_variance=turbulence_variance,
                             visibility=visibility)
        step = 0
        # ani.put_data(agents={agent: agent.position for agent in env.agents}, corridors=env.corridors, round=z)
        while env.agents:
            s1 = {agent: s[agent]['self'] for agent in env.agents}
            s2 = {agent: s[agent]['other'] for agent in env.agents}
            s1_lst = [state for agent, state in s1.items()]
            s2_lst = [state for agent, state in s2.items()]
            a_lst, logprob_a_lst = model.evaluate(s1_lst, s2_lst)
            # a_lst, logprob_a_lst, alpha, beta = model.select_action(s1_lst, s2_lst)
            actions = {agent: a for agent, a in zip(env.agents, a_lst)}
            s, rewards, terminations, truncations, infos = env.step(actions)
            # ani.put_data(round=z, agents={agent: agent.position for agent in env.agents})
            for agent in env.agents:
                if agent.status != 'Normal' and agent not in status:
                    status[agent] = agent.status
                    if agent.trajectory_ave_speed > 0:
                        accumulated_won_speed += agent.trajectory_ave_speed
                        accumulated_won_time+=agent.travel_time

            env.agents = [agent for agent in env.agents if not agent.terminated]
            step += 1
            # print(step)

    state_count = collections.Counter(status.values())
    ave_won_speed = accumulated_won_speed / max(1, state_count['won'])
    ave_won_time = accumulated_won_time / max(1, state_count['won'])
    total_agents = num_agents * max_round
    won_rate = round(state_count['won'] / max(total_agents, 1), 2)
    print(
        f"{combination}  {trained_level}  {state_count['won']}/{total_agents}  {won_rate}, speed:{round(ave_won_speed, 3)}")
    # return model_key, num_agents, level, state_count['won'], won_rate, state_count, ave_won_speed
    return combination, state_count['won'], won_rate, state_count, ave_won_speed, ave_won_time


# ani.show_animation()
# Specify the file name where you want to save the array

def main():
    multiprocessing.set_start_method('spawn', force=True)

    # Your existing setup code

    # dic_id_name = {'1-1': '10-circle',
    #                '1-2': '10-grid',
    #                '2-1': '3e-T6',
    #                '2-2': '3e-F6',
    #                '2-3': 'HD-T6',
    #                '2-4': 'HD-F6',
    #                '3-1': 'HD-F',
    #                '3-2': 'HD-T',
    #                '3-3': 'HTransRL-F',
    #                '3-4': 'HTransRL-T',
    #                '3-5': 'DS-F',
    #                '3-6': 'DS-T',
    #                }

    # for HTransRL vanilla
    model_set = reversed(['3-1', '3-2', '3-3', '3-4', '3-5', '3-6'])

    # 4-(5-8) fed models
    # model_set = reversed(['4-1', '4-5', '4-6', '4-7', '4-8'])
    level_set = reversed([20, 21])
    # agents_dic = {6: 0, 9: 1, 12: 2, 18: 3, 24: 4, 36: 5}
    agents_dic = {6: 0, 9: 1, 12: 2, 18: 3, 24: 4, 36: 5}
    # agents_dic = {6: 0, 12: 2}
    model_dic = {j: i for i, j in enumerate(model_set)}
    level_dic = {j: i for i, j in enumerate(level_set)}
    turbulence_dic = {j: i for i, j in enumerate([0])}
    visibility_dic = {j: i for i, j in enumerate([4.0])}
    combinations = list(product(model_dic.keys(),
                                agents_dic.keys(),
                                level_dic.keys(),
                                turbulence_dic.keys(),
                                visibility_dic.keys()))

    # Limit the number of processes to 4
    with multiprocessing.Pool(processes=9) as pool:
        results = pool.map(process_combination, combinations)
        # print(results)

    with open('plot/test_data_time.json', 'w') as file:
        json.dump(results, file, indent=4)

    # with open('./plot/test_data_ori.json', 'r') as file:
    #     results = json.load(file)

    # won_matrix = np.zeros([len(model_dic.keys()), len(agents_dic.keys()), len(level_dic.keys())])
    # won_rate_matrix = np.zeros([len(model_dic.keys()), len(agents_dic.keys()), len(level_dic.keys())])
    # for result in results:
    #     if result:
    #         model_key, num_agents, level_key, won_value, won_rate, status_count, ave_won_speed = result
    #         won_matrix[model_dic[model_key], agents_dic[num_agents], level_dic[level_key]] = won_value
    #         won_rate_matrix[model_dic[model_key], agents_dic[num_agents], level_dic[level_key]] = won_rate
    #
    # arrays = {'model': model_dic,
    #           'agent': agents_dic,
    #           'level': level_dic,
    #           'won_times': won_matrix,
    #           'won_rate': won_rate_matrix,
    #           'status_count': status_count,
    #           'ave_won_speed': ave_won_speed}
    # with open('./plot/array.pkl', 'wb') as f:
    #     pickle.dump(arrays, f)


if __name__ == "__main__":
    begin = time.time()
    main()
    print(time.time() - begin)
