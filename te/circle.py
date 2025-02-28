import numpy as np
import matplotlib.pyplot as plt

def generate_evenly_distributed_points(num_points, radius):
    # Calculate angles at equal intervals
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    # Convert polar to Cartesian coordinates
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return x, y

def plot_points(num_points, radius, subplot_title,circle_radius=2):
    x, y = generate_evenly_distributed_points(num_points, radius)
    plt.scatter(x, y, label=f'{num_points} Points',color='red')
    plt.xlim(-radius-0.1, radius+0.1)  # Add some padding
    plt.ylim(-radius-0.1, radius+0.1)  # Add some padding
    circle = plt.Circle((0, 0), circle_radius, color='blue', fill=False)
    plt.gca().add_patch(circle)
    plt.xlim(-circle_radius-0.1, circle_radius+0.1)
    plt.ylim(-circle_radius-0.1, circle_radius+0.1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(subplot_title)
    plt.legend()

# Plot settings
radius = 1.4
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# Subplot 1: 5 points
plt.sca(ax[0])
plot_points(5, radius,  '5 Evenly Distributed Points')

# Subplot 2: 9 points
plt.sca(ax[1])
plot_points(9, radius, '9 Evenly Distributed Points')
plt.savefig('circle_release.jpg')
plt.savefig('circle_release.pdf')
plt.show()
