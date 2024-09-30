import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import plotly.graph_objects as go

# Sample data for a 3D sine wave
x = np.linspace(0, 2 * np.pi, 100)
y = np.linspace(0, 2 * np.pi, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis')

# Update function for FuncAnimation
def update(frame):
    ax.clear()
    Z = np.sin(X + frame / 10) * np.cos(Y + frame / 10)
    ax.plot_surface(X, Y, Z, cmap='viridis')
    return ax,

ani = FuncAnimation(fig, update, frames=range(100), blit=False)

# We can't directly convert a 3D Matplotlib plot to Plotly as we did for 2D
# Instead, we create the 3D Plotly animation directly

# Convert to a Plotly animation
plotly_frames = [go.Frame(data=[go.Surface(z=np.sin(X + f / 10) * np.cos(Y + f / 10))]) for f in range(100)]
plotly_fig = go.Figure(data=[go.Surface(z=Z)], frames=plotly_frames)
plotly_fig.update_layout(scene=dict(zaxis=dict(range=[-1, 1]), xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))

# Export to HTML
plotly_fig.write_html('your_interactive_plot.html')
