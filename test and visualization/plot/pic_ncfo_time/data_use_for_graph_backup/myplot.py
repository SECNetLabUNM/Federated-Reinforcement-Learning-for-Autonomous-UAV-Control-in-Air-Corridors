import json
import pickle

import matplotlib.gridspec as gridspec
import numpy as np

from air_corridor.tools.util import generate_hexagon_grid

#
with open('array.pkl', 'rb') as f:
    arrays = pickle.load(f)

with open('test_data.json', 'r') as f:
    data = json.load(f)

import matplotlib.pyplot as plt

model_dic = arrays['model']
# to keep plot legend sequence 1-4
model_lst = sorted([(key, value) for key, value in model_dic.items() if key not in ['1-1', '1-2']],
                   key=lambda x: x[1], reverse=True)
agents_dic = arrays['agent']
level_dic = arrays['level']
won_rate = arrays['won_rate']

dic_id_name = {'1-1': '10-circle',
               '1-2': '10-grid',
               '2-1': '3e-T6',
               '2-2': '3e-F6',
               '2-3': 'dec-T6',
               '2-4': 'dec-F6',
               '3-1': 'Dec-F',
               '3-2': 'Dec-T',
               '3-3': '3e-F',
               '3-4': '3e-T',
               '3-5': 'maxs-F',
               '3-6': 'maxs-T',
               }
dic_id_marker = {'1-1': 'o',  # circle
                 '1-2': 'v',  # triangle_down
                 '2-1': 's',  # square
                 '2-2': '^',  # triangle_up
                 '2-3': 'p',  # pentagon
                 '2-4': '*',  # star
                 '3-1': 'h',  # hexagon1
                 '3-2': '<',  # triangle_left
                 '3-3': '>',  # triangle_right
                 '3-4': 'D',  # diamond
                 '3-5': 'x',  # x (cross)
                 '3-6': '+',  # plus
                 }
dic_id_level = {'1-1': '10-circle',
                '1-2': 20,
                '2-1': 20,
                '2-2': 19,
                '2-3': 19,
                '2-4': 19,
                '3-1': 19,
                '3-2': 19,
                '3-3': 19,
                '3-4': 19,
                '3-5': 19,
                '3-6': 19,
                }
dic_id_color = {'1-1': 'yellow',  # Assuming a default color, can be adjusted as needed
                '1-2': 'gray',  # Assuming a default color, can be adjusted as needed
                '2-1': 'olive',  # Assuming a default color, can be adjusted as needed
                '2-2': 'cyan',  # Assuming a default color, can be adjusted as needed
                '2-3': 'magenta',  # Assuming a default color, can be adjusted as needed
                '2-4': 'black',  # Assuming a default color, can be adjusted as needed
                '3-1': 'tab:red',
                '3-2': 'tab:blue',
                '3-3': 'tab:green',
                '3-4': 'tab:orange',
                '3-5': 'tab:purple',
                '3-6': 'tab:pink',
                }
dic_level_segment = {14: 0,
                     19: 1,
                     20: 2,
                     21: 3
                     }
dic_level_name = {14: 'cttc',
                  19: 'training environment',
                  20: 'cttcttc',
                  21: 'cttcttcttc'
                  }

# Create subplots
circle_radius = 2
vertex_distance = 0.6
vertices = generate_hexagon_grid(circle_radius, vertex_distance)
vertex_x_vals, vertex_y_vals = zip(*vertices)

##############################################33##############################################33



##############################################33##############################################33
sizeH = 4
sizeV = 1.65
fig = plt.figure(figsize=(9, 3))
gs = gridspec.GridSpec(1, 3, width_ratios=[2, 3, 3])  # 1 row, 3 columns with the specified width ratios
axs = [fig.add_subplot(gs[0, i]) for i in range(3)]  # First column
axs[0].scatter(vertex_x_vals, vertex_y_vals, color='red', label='Rotated Vertices')
circle = plt.Circle((0, 0), circle_radius, color='blue', fill=False)
axs[0].add_patch(circle)
axs[0].set_xlim(-circle_radius, circle_radius)
axs[0].set_ylim(-circle_radius, circle_radius)
axs[0].set_aspect('equal', adjustable='box')
# axs[0].set_title("(a)", loc='center')
axs[0].set(xlabel=f"\n({chr(ord('a') + 0)}) ")
for j, level_key in enumerate([20, 21], start=1):
    for model_key, i in model_lst:
        # Convert agents_dic.values() to a list and use proper indexing for won_rate
        agents = list(agents_dic.values())
        level_index = level_dic[level_key]
        won_rate_data = won_rate[i, agents, level_index]
        axs[j].plot(agents_dic.keys(), won_rate_data, marker=dic_id_marker[model_key],
                    label=dic_id_name[model_key], color=dic_id_color[model_key])  # Call plot on axs, not fig
    axs[j].grid()
    if j == 1:
        axs[1].set(xlabel=f"Number of UAVs\n({chr(ord('a') + j)}) {dic_level_name[level_key]}", ylabel='Arrival rate')
    if j == 2:
        axs[2].set(xlabel=f"(Number of UAVs\n({chr(ord('a') + j)}) {dic_level_name[level_key]}")
# plt.title('Arrival rate on cylinder-torus-torus-cylinder')
plt.legend()
fig.tight_layout()
fig.savefig('test_3.jpg')
fig.savefig('test_3.pdf')
plt.show()


##############################################33##############################################33


def generate_data(models, agent_key, level_key, unified=True):
    fail_data = []

    for model_key in models:
        row_data = [0 for _ in range(4)]
        try:
            index = next(i for i, row in enumerate(data) if row[:3] == [model_key, agent_key, level_key])
        except StopIteration:
            continue

        for i, (key, value) in enumerate(data[index][5].items()):
            if key in {'breached_wall', 'breached_rad_t_wall', 'breached_rad'}:
                row_data[0] += value
            elif key == 'breached_c':
                row_data[1] += value
            elif key == 'collided_UAV':
                row_data[2] += value
            elif key == 'collided':
                row_data[3] += value
        if unified:
            fail_data.append(np.array(row_data) / sum(row_data))
        else:
            fail_data.append(np.array(row_data))
    return np.transpose(np.array(fail_data))


fail_list = {'breached_wall': 'breach torus',
             'breached_rad_t_wall': 'breach torus',
             'breached_rad': 'breach torus',
             'breached_c': 'breach cylinder',
             'collided_UAV': 'collision with NFCOs',
             'collided': 'collision among UAVs'}
# #
fail_reason_color = {'breach torus': 'red',
                     'breach cylinder': 'blue',
                     'collision among UAVs': 'green',
                     'collision with NFCOs': 'pink'}
model_fraction = {'3-5': 'DS', '3-1': 'HD', '3-4': 'HT'}
fig, axs = plt.subplots(4, 1, figsize=(10, 8))
num_bars = len(model_fraction.keys())
settings = [[20, 12], [21, 12], [20, 36], [21, 36]]
colors = ['b', 'g', 'r', 'orange']
for ax, (level_key, agent_key) in zip(axs, settings):
    fail_data = generate_data(model_fraction.keys(), agent_key, level_key)
    left = [0] * 3
    for i in range(4):
        ax.barh(range(3), fail_data[i], left=left, color=colors[i])
        left = left + fail_data[i]
    ax.set_yticks(range(3))
    ax.set_yticklabels([model_fraction[key] for key in model_fraction.keys()])
    ax.set_title(f"Agents: {agent_key}, Test: {dic_level_name[level_key]}")
# Tight layout to prevent overlap
plt.tight_layout()
plt.show()
##############################################33##############################################33
fig, axs = plt.subplots(4, 1, figsize=(10, 8))
num_bars = len(model_fraction.keys())
settings = [[20, 12], [21, 12], [20, 36], [21, 36]]
colors = ['b', 'g', 'r', 'orange']
for ax, (level_key, agent_key) in zip(axs, settings):
    fail_data = generate_data(model_fraction.keys(), agent_key, level_key, unified=False)
    left = [0] * 3
    for i in range(4):
        ax.barh(range(3), fail_data[i], left=left, color=colors[i])
        left = left + fail_data[i]
        # if i < 2:
        #     ax.set_xlim(0, 500)
        # else:
        #     ax.set_xlim(0, 2200)
    ax.set_yticks(range(3))
    ax.set_yticklabels([model_fraction[key] for key in model_fraction.keys()])
    ax.set_title(f"Agents: {agent_key}, Test: {dic_level_name[level_key]}")

plt.tight_layout()
plt.show()

################################

# speed for 19 (5 corridors) and 21 (10 corridors) scenarios

width=0.2
level_key=21
speed_21=[]
for model_key in model_fraction.keys():
    row=[]
    for agent_key in [6,9,12,18,24,36]:
        try:
            index = next(i for i, row in enumerate(data) if row[:3] == [model_key, agent_key, level_key])
        except StopIteration:
            continue
        row.append(data[index][-1])
    speed_21.append(row)


level_key=20
speed_20=[]
for model_key in model_fraction.keys():
    row=[]
    for agent_key in agents_dic.keys():
        try:
            index = next(i for i, row in enumerate(data) if row[:3] == [model_key, agent_key, level_key])
        except StopIteration:
            continue
        row.append(data[index][-1])
    speed_20.append(row)


fig, ax = plt.subplots()
rects = []
x=np.array(list(range(len(agents_dic.keys()))))
speed_data=np.transpose(np.array(speed_21))
for i in range(len(model_fraction.keys())):
    # Generate random data and create bars for each method
    rects.append(ax.bar(x + i*width, speed_data[:, i], width, label=list(model_fraction.values())[i]))

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_ylabel('Average speed for arrivals')
# ax.set_title('Scores by scenario and method')
ax.set_xticks(x + width*(len(model_fraction.keys())-1)/2)
ax.set_xticklabels(agents_dic.keys())
ax.set_ylabel('Number of agents')
ax.grid()
ax.legend()

# Attach a text label above each bar in rects, displaying its height.
for rect in rects:
    for bar in rect:
        height = bar.get_height()
        ax.annotate('%.2f' % height,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        ax.set_ylim(1.15, 1.3)

plt.show()
print(1)




fig, ax = plt.subplots()
rects = []
x=np.array(list(range(len(agents_dic.keys()))))
speed_data=np.transpose(np.array(speed_20))
for i in range(len(model_fraction.keys())):
    # Generate random data and create bars for each method
    rects.append(ax.bar(x + i*width, speed_data[:, i], width, label=list(model_fraction.values())[i]))

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_ylabel('Average speed for arrivals')
# ax.set_title('Scores by scenario and method')
ax.set_xticks(x + width*(len(model_fraction.keys())-1)/2)
ax.set_xticklabels(agents_dic.keys())
ax.set_ylabel('Number of agents')
ax.grid()
ax.legend()

# Attach a text label above each bar in rects, displaying its height.
for rect in rects:
    for bar in rect:
        height = bar.get_height()
        ax.annotate('%.2f' % height,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        ax.set_ylim(1.15, 1.3)

plt.show()
