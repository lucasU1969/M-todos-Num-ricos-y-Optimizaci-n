import numpy as np
import matplotlib.pyplot as plt
import math as m

# Constantes
k = 1000
y0 = 10
r = 0.1
initial_condition = 100
t0 = 0
t_max = 100
h_values = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]  # Valores de h que deseas probar
# Funciones de crecimiento exponencial y logístico
def exponential_growth_equation(y0, t):
    return y0 * np.exp(r * t)

def logistic_growth_equation(y0, t):
    return (y0*k*np.exp(r*t))/((k-y0) + y0*np.exp(r*t))

# Método de Euler
def euler_ode(h, ode, initial_condition, time):
    y_values = [initial_condition]
    for i in range(1, len(time)):
        y_next = y_values[-1] + h * ode(time[i], y_values[-1])
        y_values.append(y_next)
    return np.array(y_values)

# Método de Runge-Kutta de segundo orden
def runge_kutta_2_ode(h, ode, initial_condition, time):
    y_values = [initial_condition]
    for i in range(1, len(time)):
        k1 = h * ode(time[i], y_values[-1])
        k2 = h * ode(time[i] + h, y_values[-1] + k1)
        y_next = y_values[-1] + 0.5 * (k1 + k2)
        y_values.append(y_next)
    return np.array(y_values)

# Método de Runge-Kutta de cuarto orden (RK4)
def runge_kutta_4_ode(h, ode, initial_condition, time):
    y_values = [initial_condition]
    for i in range(1, len(time)):
        k1 = h * ode(time[i], y_values[-1])
        k2 = h * ode(time[i] + h/2, y_values[-1] + k1/2)
        k3 = h * ode(time[i] + h/2, y_values[-1] + k2/2)
        k4 = h * ode(time[i] + h, y_values[-1] + k3)
        y_next = y_values[-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
        y_values.append(y_next)
    return np.array(y_values)

def exponential_ode(t, y):
    return r * y

def logistic_ode(t, y):
    return r * y * ((k - y)/k)

# def calculate_average_error(acumulate_error):
#     average_error = 0
#     for i in range(len(acumulate_error)):
#         average_error += acumulate_error[i]

#     return average_error/len(acumulate_error)

def calculate_average_error(acumulate_error, exact_solution):
    relative_errors = [abs(error / real_value) for error, real_value in zip(acumulate_error, exact_solution)]
    return sum(relative_errors) / len(relative_errors)

def calculate_cumulative_error(approximate_solution, exact_solution):
    cumulative_error = []
    for i in range(len(approximate_solution)):
        cumulative_error.append(abs(approximate_solution[i] - exact_solution[i]))

    return cumulative_error

def calculate_average_relative_error(acumulate_error, exact_solution):
    relative_errors = [abs(error / real_value) for error, real_value in zip(acumulate_error, exact_solution)]
    return sum(relative_errors) / len(relative_errors)

# Cálculo de errores para diferentes valores de h
euler_errors = []
rk2_errors = []
rk4_errors = []
min_rk4 = 0
min_rk2 = 0
min_euler = 0
h_euler = 0
h_rk2 = 0
h_rk4 = 0
for h in h_values:
    time = np.arange(t0, t_max, h)
    # Euler
    euler_solution = euler_ode(h, exponential_ode, initial_condition, time)
    euler_exact_solution = exponential_growth_equation(initial_condition, time)
    euler_cumulative_error = calculate_cumulative_error(euler_solution, euler_exact_solution)
    euler_error = calculate_average_error(euler_cumulative_error, euler_exact_solution)
    euler_errors.append(euler_error)
    # Runge-Kutta de segundo orden
    rk2_solution = runge_kutta_2_ode(h, exponential_ode, initial_condition, time)
    rk2_exact_solution = exponential_growth_equation(initial_condition, time)
    rk2_cumulative_error = calculate_cumulative_error(rk2_solution, rk2_exact_solution)
    rk2_error = calculate_average_error(rk2_cumulative_error, rk2_exact_solution)
    rk2_errors.append(rk2_error)
    # Runge-Kutta de cuarto orden
    rk4_solution = runge_kutta_4_ode(h, exponential_ode, initial_condition, time)
    rk4_exact_solution = exponential_growth_equation(initial_condition, time)
    rk4_cumulative_error = calculate_cumulative_error(rk4_solution, rk4_exact_solution)
    rk4_error = calculate_average_error(rk4_cumulative_error, rk4_exact_solution)
    rk4_errors.append(rk4_error)

    if(min_rk4 == 0 or min_rk4 > rk4_error):
        min_rk4 = rk4_error
        h_rk4 = h
    if(min_rk2 == 0 or min_rk2 > rk2_error):
        min_rk2 = rk2_error
        h_rk2 = h
    if(min_euler == 0 or min_euler > euler_error):
        min_euler = euler_error
        h_euler = h
print(h_rk4)
print(h_rk2)
print(h_euler)
# Graficar los errores en función de h
plt.figure(figsize=(10, 6))
plt.plot(h_values, euler_errors, label='Euler', marker='o')
plt.plot(h_values, rk2_errors, label='Runge-Kutta de segundo orden', marker='o')
plt.plot(h_values, rk4_errors, label='Runge-Kutta de cuarto orden', marker='o')
# plt.xscale('log')  # Escala logarítmica en el eje x
plt.yscale('log')  # Escala logarítmica en el eje y
plt.xlabel('Tamaño del Paso (h)')
plt.ylabel('Error Relativo')
plt.title('Error Relativo vs Tamaño del Paso (h)')
plt.legend()
plt.grid(True)
plt.show()
