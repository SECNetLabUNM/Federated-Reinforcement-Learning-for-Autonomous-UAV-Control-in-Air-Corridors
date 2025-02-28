import matplotlib.pyplot as plt
import numpy as np
import matplotlib

import plotly.tools as tls
import plotly.offline as py

matplotlib.use('TkAgg')
from air_corridor.tools.util import *
# Define a torus
def torus(R, r, R_res=100, r_res=100):
    u = np.linspace(0, 1.5 * np.pi, R_res)
    v = np.linspace(0, 2 * np.pi, r_res)
    u, v = np.meshgrid(u, v)
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return x, y, z

# Create torus data
R_torus = 10  # Major radius
r_torus = 3   # Minor radius
Xt, Yt, Zt = torus(R_torus, r_torus)
rotation_matrix = vec2vec_rotation(np.array([0, 0, 1]), np.array([1, 1, 1]))
# Apply rotation
x_rot_torus, y_rot_torus, z_rot_torus = [], [], []
for a, b, c in zip(Xt, Yt, Zt):
    x_p, y_p, z_p = np.dot(rotation_matrix, np.array([a, b, c]))
    x_rot_torus.append(x_p)
    y_rot_torus.append(y_p)
    z_rot_torus.append(z_p)

# Plot the rotated torus
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(np.array(x_rot_torus), np.array(y_rot_torus), np.array(z_rot_torus), edgecolor='royalblue', lw=0.1, rstride=20, cstride=8, alpha=0.1)

# Set labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plotly_fig = tls.mpl_to_plotly(fig)
py.plot(plotly_fig, filename='interactive_plot.html')
plt.show()
