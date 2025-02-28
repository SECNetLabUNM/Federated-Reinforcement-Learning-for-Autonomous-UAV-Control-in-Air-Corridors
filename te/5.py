import time

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp  # Proportional gain
        self.Ki = Ki  # Integral gain
        self.Kd = Kd  # Derivative gain
        self.setpoint = setpoint  # Desired setpoint

        self.prev_error = 0  # Previous error
        self.integral = 0     # Integral term (accumulated error)

    def update(self, feedback):
        # Calculate error
        error = self.setpoint - feedback

        # Proportional term
        P = self.Kp * error

        # Integral term
        self.integral += error
        I = self.Ki * self.integral

        # Derivative term
        D = self.Kd * (error - self.prev_error)

        # Calculate control signal
        control_signal = P + I + D

        # Update previous error for next iteration
        self.prev_error = error

        return control_signal

# Define PID parameters
Kp = 0.5
Ki = 0.1
Kd = 0.2
setpoint = 50  # Desired temperature in Celsius

# Create PID controller
pid_controller = PIDController(Kp, Ki, Kd, setpoint)

# Simulate temperature measurement function
def measure_temperature():
    # Simulate temperature measurement (replace with actual sensor reading)
    return 40 + 5 * (time.time() % 10)

# Simulate heater control function
def control_heater(power):
    # Simulate controlling the heater (replace with actual control mechanism)
    print(f"Adjusting heater power to {power}%")

# Main control loop
def main():
    while True:
        # Measure current temperature
        current_temperature = measure_temperature()

        # Update PID controller with current temperature
        control_signal = pid_controller.update(current_temperature)

        # Apply control signal to heater (clamp between 0 and 100)
        power = max(0, min(control_signal, 100))
        control_heater(power)

        # Print current temperature and control signal
        print(f"Current temperature: {current_temperature} C, Control signal: {control_signal}")

        # Wait for some time (simulate real-time operation)
        time.sleep(1)

if __name__ == "__main__":
    main()
