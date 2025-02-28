import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

# Simulated first-order process
def process(y, t, Kp, Ki, Kd, setpoint, last_error, integral):
    error = setpoint - y
    derivative = error - last_error
    integral += error
    u = Kp*error + Ki*integral + Kd*derivative
    dydt = (-y + u) / tau
    return dydt

# Objective function to minimize (integrated squared error)
def objective(params, setpoint, initial_conditions, t):
    Kp, Ki, Kd = params
    last_error = 0
    integral = 0
    solution = odeint(process, initial_conditions, t, args=(Kp, Ki, Kd, setpoint, last_error, integral))
    error = setpoint - solution.ravel()
    return np.sum(error**2)

# PID parameters
initial_params = [0.1, 0.01, 0.01]  # Initial guess for [Kp, Ki, Kd]
tau = 10  # Time constant of the process
setpoint = 1  # Desired setpoint
t = np.linspace(0, 100, 500)  # Time vector

# Initial conditions
initial_conditions = [0]  # Starting at y(0) = 0

# Minimize the objective function
result = minimize(objective, initial_params, args=(setpoint, initial_conditions, t), method='L-BFGS-B', options={'disp': True})

print(f"Optimized PID Parameters: Kp = {result.x[0]}, Ki = {result.x[1]}, Kd = {result.x[2]}")
