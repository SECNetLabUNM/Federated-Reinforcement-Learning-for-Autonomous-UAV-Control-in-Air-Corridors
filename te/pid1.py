import numpy as np
import matplotlib.pyplot as plt

# Constants
V_max = 2.0  # Maximum velocity (m/s)
A_max = 0.3  # Maximum acceleration (m/s^2)
dt = 1.0  # Time step in seconds
epsilon = 0.1  # Threshold for reaching destination


# Trajectory definition (as functions)
def line_segment(p0, p1, t):
    """ Linear interpolation between p0 and p1 """
    return p0 + (p1 - p0) * t


def circular_arc(center, radius, theta0, theta1, t):
    """ Calculate a point on a circular arc """
    theta = np.linspace(theta0, theta1, num=100)  # More points for a smoother arc
    index = int(t * (len(theta) - 1))
    return center + radius * np.array([np.cos(theta[index]), np.sin(theta[index])])


# PID Controller
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.previous_error = np.array([0.0, 0.0])
        self.integral = np.array([0.0, 0.0])

    def update(self, current_position, target_position):
        error = target_position - current_position
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        return output


# Function to find the closest point on the trajectory
def closest_point_on_trajectory(current_pos):
    # Here we use a dummy trajectory: Straight line followed by a semicircle
    # More complex paths will need a more sophisticated projection method
    p0 = np.array([0.0, 0])
    p1 = np.array([10.0, 0])
    center = np.array([10, 5])
    radius = 5
    theta0 = -np.pi / 2
    theta1 = np.pi / 2

    # Check linear segment
    line_dir = p1 - p0
    line_len = np.linalg.norm(line_dir)
    line_dir /= line_len
    projection = np.dot((current_pos - p0), line_dir)
    projection = np.clip(projection, 0, line_len)
    closest_point = p0 + projection * line_dir

    # Check circular segment
    circle_pos = current_pos - center
    circle_pos /= np.linalg.norm(circle_pos)
    circle_point = center + radius * circle_pos
    if projection == line_len:  # Only consider the circle if at the end of the line
        closest_point = circle_point

    return closest_point


# Simulation
current_position = np.array([0.0, 0.0])  # Start at the beginning
pid = PIDController(Kp=0.2, Ki=0.05, Kd=0.1)
path_points = [current_position.copy()]
total_distance = 0

while np.linalg.norm(current_position - np.array([15, 5])) >= epsilon:
    target_position = closest_point_on_trajectory(current_position)
    velocity = pid.update(current_position, target_position) * dt
    velocity = np.clip(velocity, -V_max, V_max)  # Ensure max velocity is not exceeded
    current_position += velocity
    path_points.append(current_position.copy())
    total_distance += np.linalg.norm(velocity)

# Calculate average velocity
total_time = len(path_points) * dt
average_velocity = total_distance / total_time

# Plotting
path_points = np.array(path_points)
plt.figure(figsize=(12, 8))
plt.plot(path_points[:, 0], path_points[:, 1], 'r-', label='Actual Trajectory')
plt.title('Path Following with Dynamic Projection-Based PID Control')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.legend()
plt.grid(True)
plt.show()

# Output average velocity
print(f"Average velocity during the trip: {average_velocity:.2f} m/s")