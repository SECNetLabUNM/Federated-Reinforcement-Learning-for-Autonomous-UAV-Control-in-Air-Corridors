import time

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
# Assuming the vec2vec_rotation function is defined as in your code
def vec2vec_rotation(v1, v2):
    """
    Create a rotation matrix that rotates vector v1 to vector v2
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v = np.cross(v1, v2)
    c = np.dot(v1, v2)
    s = np.linalg.norm(v)

    I = np.identity(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = I + vx + np.dot(vx, vx) * ((1 - c) / (s**2))
    return R

# Define a cylinder
def cylinder(r, h, theta_res=100, z_res=100):
    theta = np.linspace(0, 2 * np.pi, theta_res)
    z = np.linspace(0, h, z_res)
    theta, z = np.meshgrid(theta, z)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, z

# Create cylinder data
radius = 10
height = 10
Xc, Yc, Zc = cylinder(radius, height)

# Apply rotation
rotation_matrix = vec2vec_rotation(np.array([0, 0, 1]), np.array([1, 1, 1]))
x_rot, y_rot, z_rot = [], [], []
for a, b, c in zip(Xc, Yc, Zc):
    x_p, y_p, z_p = np.dot(rotation_matrix, np.array([a, b, c]))
    x_rot.append(x_p)
    y_rot.append(y_p)
    z_rot.append(z_p)

# Plot the rotated cylinder
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(np.array(x_rot), np.array(y_rot), np.array(z_rot), edgecolor='royalblue', lw=0.1, rstride=20, cstride=8, alpha=0.3)

# Set labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

def close_plot_after(delay=10):
    """
    Closes the plot window after a specified delay in seconds.
    """
    plt.pause(delay)  # Wait for the specified delay
    plt.close()       # Close the plot window

# Rest of your plotting and animation setup
# ...

# Show the plot
plt.show(block=False)  # Non-blocking show
import threading
# Start a timer to close the plot automatically
timer = threading.Thread(target=close_plot_after, args=(10,))
timer.start()
plt.close()
