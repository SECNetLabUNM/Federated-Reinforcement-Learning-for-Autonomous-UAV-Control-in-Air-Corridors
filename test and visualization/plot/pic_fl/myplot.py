import json
from itertools import product

import numpy as np

with open('test_data_merged.json', 'r') as file:
    results = json.load(file)

model_set = ['4-1', '4-5', '4-6', '4-7', '4-8']

level_set = [20, 21]
agents_dic = {6: 0, 9: 1, 12: 2, 18: 3, 24: 4, 36: 5}
model_dic = {j: i for i, j in enumerate(model_set)}
level_dic = {j: i for i, j in enumerate(level_set)}
turbulence_dic = {j: i for i, j in enumerate([0.10, 0.2])}
visibility_dic = {j: i for i, j in enumerate([4.0, 1.5])}
combinations = list(product(model_dic.keys(),
                            agents_dic.keys(),
                            level_dic.keys(),
                            turbulence_dic.keys(),
                            visibility_dic.keys()))

won_matrix = np.zeros([len(model_dic.keys()),
                       len(agents_dic.keys()),
                       len(level_dic.keys()),
                       len(turbulence_dic.keys()),
                       len(visibility_dic.keys())])
won_rate_matrix = np.zeros([len(model_dic.keys()),
                            len(agents_dic.keys()),
                            len(level_dic.keys()),
                            len(turbulence_dic.keys()),
                            len(visibility_dic.keys())])
for result in results:
    if result:
        (model_key, num_agents, level_key, turbulence_key,
         visibility_key), won_value, won_rate, status_count, ave_won_speed = result
        won_matrix[model_dic[model_key], agents_dic[num_agents], level_dic[level_key], turbulence_dic[turbulence_key],
        visibility_dic[visibility_key]] = won_value
        won_rate_matrix[
            model_dic[model_key], agents_dic[num_agents], level_dic[level_key], turbulence_dic[turbulence_key],
            visibility_dic[visibility_key]] = won_rate

import matplotlib.pyplot as plt

# to keep plot legend sequence 1-4
model_lst = sorted([(key, value) for key, value in model_dic.items() if key not in ['1-1', '1-2']],
                   key=lambda x: x[1], reverse=True)

dic_id_name = {'4-1': 'Origin',
               '4-5': 'FT-1.5-0.1',
               '4-6': 'FT-4.0-0.1',
               '4-7': 'FT-1.5-0.075',
               '4-8': 'FT-4.0-0.075',
               }
dic_id_marker = {'4-1': 'o',  # circle
                 '4-5': 'v',  # triangle_down
                 '4-6': 's',  # square
                 '4-7': '^',  # triangle_up
                 '4-8': 'p',  # pentagon
                 }

dic_id_color = {'4-1': 'red',  # Assuming a default color, can be adjusted as needed
                '4-5': 'blue',  # Assuming a default color, can be adjusted as needed
                '4-6': 'green',  # Assuming a default color, can be adjusted as needed
                '4-7': 'cyan',  # Assuming a default color, can be adjusted as needed
                '4-8': 'magenta',  # Assuming a default color, can be adjusted as needed
                }

dic_level_name = {14: 'cttc',
                  19: 'training environment',
                  20: 'cttcttc',
                  21: 'cttcttcttc'
                  }

############################################################################################33

for visi_index, (visi_key, _) in enumerate(visibility_dic.items()):
    for level_key, _ in level_dic.items():
        fig, axs = plt.subplots(ncols=2, figsize=(8, 4.1), sharey=True, tight_layout=True)
        for tur_index, (tur_key, tur_v) in enumerate(turbulence_dic.items()):
            for model_id, i in model_lst:
                # Convert agents_dic.values() to a list and use proper indexing for won_rate
                agents = list(agents_dic.values())
                level_index = level_dic[level_key]

                won_rate_data = won_rate_matrix[i, agents, level_index, tur_index, visi_index]

                axs[tur_index].plot(agents_dic.keys(),
                                    won_rate_data,
                                    marker=dic_id_marker[model_id],
                                    label=dic_id_name[model_id],
                                    color=dic_id_color[model_id])  # Call plot on axs, not fig

            axs[tur_index].set(xlabel=f"Number of UAVs\n({chr(ord('a') + tur_index)}) turbulence \u03C3 = {tur_key}")
            axs[tur_index].grid()
        axs[0].set(ylabel='Arrival rate')

        plt.legend()
        fig.savefig(f"fl_test_visi_{visi_key}_level_{level_key}.jpg", dpi=600)
        fig.savefig(f"fl_test_visi_{visi_key}_level_{level_key}.pdf")
        plt.show()

