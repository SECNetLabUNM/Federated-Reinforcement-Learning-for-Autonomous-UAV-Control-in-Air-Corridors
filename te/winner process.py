import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters for the Wiener process
T = 1.0  # Total time
N = 1000  # Number of steps
dt = T/N  # Time step

# Create an array of time points
t = np.linspace(0, T, N+1)

# Generate random Wiener increments (normal distribution, mean=0, std=sqrt(dt))
dWx = np.random.normal(0, np.sqrt(dt), N)
dWy = np.random.normal(0, np.sqrt(dt), N)
dWz = np.random.normal(0, np.sqrt(dt), N)

# Cumulative sum to simulate the Wiener process
Wx = np.cumsum(dWx)
Wy = np.cumsum(dWy)
Wz = np.cumsum(dWz)

# Adding the starting point (0,0,0) to the path
Wx = np.insert(Wx, 0, 0)
Wy = np.insert(Wy, 0, 0)
Wz = np.insert(Wz, 0, 0)

# Plotting the 3D Wiener process
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(Wx, Wy, Wz)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('3D Wiener Process')
plt.show()