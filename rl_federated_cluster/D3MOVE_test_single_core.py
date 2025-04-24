# from pettingzoo.mpe import simple_adversary_v3
import collections
import json
import numpy as np

import air_corridor.d3.scenario.D3shapeMove as d3
from air_corridor.tools.util import load_init_params
from air_corridor.tools.visualization import Visualization as vl
from ppo_cluster import PPO
import pandas as pd
from gym_pybullet_drones.utils.utils import sync, str2bool
import time
env = d3.parallel_env(render_mode="")

import loratorch as lora

# trained model info
# which model to load
loadModel = True
strs = ['lora']#['upper','lower','sig02_fed03','sig005_fed03','sig01_fed05','sig01_fed03','sig01_fed01']
for model_str in strs:
    dframe = pd.DataFrame({'exp':[],'rate_won':[], 'ave_w_speed':[]})

    if loadModel:
        result = '/home/meng/Documents/Code/HTransRL/exp fl/'
        loadFolder = f"{result}{model_str}"
        modelIndex = '1.0m'
        net_model = 'fc10_3e'
        trained_level = 19

    kwargs = load_init_params(name='net_params', dir=loadFolder)
    opt = load_init_params(name='main_params', dir=loadFolder)
    with open(f"{loadFolder}/net_params.json", 'r') as json_file:
        kwargs1 = json.load(json_file)
        # opt=json.load
    kwargs['net_model'] = net_model
    model = PPO(**kwargs)
    model.load(folder=loadFolder, global_step=modelIndex,lora=True)

    opt = load_init_params(name='main_params', dir=loadFolder)
    max_round = 1000

    ani_bool = False
    if ani_bool:
        ani = vl(max_round + 1, to_base=False)

    status = {}
    # level 14: cttc;
    # level 20: cttcttc
    # level 21: cttcttcttc
    level_key = 21
    level_dic = {14: 'cttc',
                20: 'cttcttc',
                21: 'cttcttcttc'}
    for num_agents in [5,10,15,20, 25, 30, 35, 40]:

        #SET DT
        dt = 1
        START = time.time()
        status = {}
        print(f"simulation in {level_key-34} corridors with {num_agents} UAVs")
        ani = vl(max_round + 1, to_base=False)
        ave_speed = 0
        for i in range(max_round + 1):
            
            '''
            training scenarios can be different from test scenarios, so num_corridor_in_state and corridor_index_awareness
            need to match the setting for UAV during training.
            '''
            # level 14: cttc;
            # level 20: cttcttc
            # level 21: cttcttcttc
            options = {'difficulty': 1.0}
            s, infos = env.reset(num_agents=num_agents,
                                num_obstacles=4,
                                num_ncfo=3,
                                level=19,
                                dt=dt, #opt['dt'],
                                beta_adaptor_coefficient=opt['beta_adaptor_coefficient'],
                                test=True,
                                options=options,
                                turbulence_variance=0.1,
                                visibility=6,
                                velocity_max=1.5,
                                acceleration_max=0.3)
            current_actions = {}
            step = 0
            agents = env.agents
            if ani_bool:
                ani.put_data(agents={agent: agent.position for agent in env.agents},
                            ncfos={ncfo: ncfo.position for ncfo in env.ncfos}, corridors=env.corridors, round=i)
            
            terminated = False
            while not terminated:
                if loadModel:
                    s1 = {agent: s[agent]['self'] for agent in env.agents}
                    s2 = {agent: s[agent]['other'] for agent in env.agents}
                    s1_lst = [state for agent, state in s1.items()]
                    s2_lst = [state for agent, state in s2.items()]
                    a_lst, logprob_a_lst,_,_ = model.select_action(s1_lst, s2_lst, 0)
                    actions = {agent: a for agent, a in zip(env.agents, a_lst)}
                else:
                    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
                action_updated = False
                s, rewards, terminations, truncations, infos = env.step(actions)
                if ani_bool and step%(1/dt) ==0:
                    ani.put_data(round=i, agents={agent: agent.position for agent in env.agents},
                                ncfos={ncfo: ncfo.position for ncfo in env.ncfos})
                # print(rewards)
                done = {agent: terminations[agent] | truncations[agent] for agent in env.agents}
                
                step += 1

                terminated = True
                for agent in env.agents:
                    if not agent.terminated:
                        terminated = False
                # if env.sim_mode=='sitl_pygym':
                #     env.sitl_env.render()
                #     sync(step, START, env.sitl_env.CTRL_TIMESTEP)
                        
                # print(step)
            #if 1:  # i % 10 != 0:

            for agent in env.agents:
                    if done[agent]:
                        ave_speed += agent.trajectory_ave_speed if agent.trajectory_ave_speed > 0 else 0
                    if agent.status != 'Normal' and agent not in status:
                        status[agent] = agent.status


        state_count = collections.Counter(status.values())
        print(f"{i}/{max_round} - {state_count}")
        print(state_count['won'] / max(sum(state_count.values()), 1))
        if state_count['won'] > 0:
            print(ave_speed/state_count['won'])
            ave_won_speed = ave_speed / max(1, state_count['won'])
        else:
            ave_won_speed = 0
        # Write to CSV
        dframe.loc[-1] = [num_agents,state_count['won'] / max(sum(state_count.values()), 1),ave_won_speed]
        dframe.index = dframe.index + 1  # shifting index
        dframe = dframe.sort_index() 
        dframe.to_csv(f"{loadFolder}.csv",)
                #
        if ani_bool:
            ani.show_animation(gif=True, save_to=f"{level_key - 34}_{num_agents}")
    env.close()

