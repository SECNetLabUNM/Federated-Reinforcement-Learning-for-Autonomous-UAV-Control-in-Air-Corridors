import json
from itertools import product

import matplotlib.gridspec as gridspec
import numpy as np

with open('test_data_all.json', 'r') as file:
    results_all = json.load(file)


with open('test_data_45.json', 'r') as file:
    results_45 = json.load(file)

with open('test_data_time.json', 'r') as file:
    results_time = json.load(file)

keys=[item[0] for item in results_45]

for j,result in enumerate(results_all):
    com, won_value, won_rate, status_count, ave_won_speed = result
    if com[0]=='4-5':
        if com in keys:
            i=keys.index(com)
            results_all[j] = results_45[i]


with open('test_data_merged.json', 'w') as file:
    json.dump(results_all, file, indent=4)
