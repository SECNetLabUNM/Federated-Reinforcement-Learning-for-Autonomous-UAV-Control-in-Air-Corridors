import matplotlib.pyplot as plt
import numpy as np

# Sample data
scenarios = ['Scenario 1', 'Scenario 2', 'Scenario 3']
methods = ['Method A', 'Method B', 'Method C', 'Method D']
data = np.random.rand(3, 4)  # Random data for 3 scenarios and 4 methods

num_scenarios = len(scenarios)
x = np.arange(num_scenarios)  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects = []

for i in range(len(methods)):
    # Generate random data and create bars for each method
    rects.append(ax.bar(x + i*width, data[:, i], width, label=methods[i]))

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by scenario and method')
ax.set_xticks(x + width*(len(methods)-1)/2)
ax.set_xticklabels(scenarios)
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

plt.show()
