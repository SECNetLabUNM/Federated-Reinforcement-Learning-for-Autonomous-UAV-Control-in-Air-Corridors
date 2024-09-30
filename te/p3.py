import matplotlib.pyplot as plt
import numpy as np

# Sample data
data = np.random.rand(4, 10)  # Random data for 4 datasets, each with 10 points

# Create a figure with a specific size
fig = plt.figure(figsize=(12, 8))

# Create subplots
# Left side: 4x1 subplots
left_axes = [fig.add_subplot(4, 2, i+1) for i in range(4)]
# Right side: 1 large plot
right_ax = fig.add_subplot(1, 2, 2)

# Plotting data on the left subplots
for i, ax in enumerate(left_axes):
    ax.plot(data[i], marker='o', linestyle='-', label=f'Dataset {i+1}')
    ax.set_title(f'Dataset {i+1}')
    ax.set_ylabel('Value')
    ax.legend()

# Adjust the subplot layout
plt.tight_layout()

# Plotting a summary on the right subplot
# For example, aggregate the data across datasets
mean_values = np.mean(data, axis=1)
right_ax.barh(range(4), mean_values, color='skyblue')
right_ax.set_xlabel('Average Value')
right_ax.set_yticks(range(4))
right_ax.set_yticklabels([f'Dataset {i+1}' for i in range(4)])
right_ax.set_title('Summary of Average Values Across Datasets')

# Show plot
plt.show()
