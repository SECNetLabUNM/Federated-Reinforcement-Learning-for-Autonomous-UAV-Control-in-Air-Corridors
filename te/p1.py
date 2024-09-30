import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


# Function to generate random data
def generate_data(num_bars, num_categories):
    return np.random.rand(num_bars, num_categories)


# Parameters
num_categories = 10  # Number of different categories/colors in each bar
betas = [0.1, 1, 10]  # Different beta values
num_bars = 5  # Number of bars in each subplot
# Creating the figure
fig, axes = plt.subplots(len(betas), 1, figsize=(10, 8))
settings = [[20, 12], [20, 36], [21, 12], [21, 36]]
for ax, beta in zip(axes, betas):
for
    data = generate_data(num_bars, num_categories)
    cumulative_data = np.cumsum(data, axis=1)
    ax.set_title(f'β={beta}')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, num_bars)
    for i in range(num_bars):
        widths = data[i]
        starts = cumulative_data[i] - widths
        ax.barh(i, widths, left=starts, color=cm.viridis(np.linspace(0, 1, num_categories)))
    ax.set_yticks(range(num_bars))
    ax.set_yticklabels([f"Category {i + 1}" for i in range(num_bars)])
# Tight layout to prevent overlap
plt.tight_layout()
plt.show()
