import numpy as np
import matplotlib.pyplot as plt
import math as m

#constantes
k = 0.1
y0 = 10
r = 0.1
h = 0.5
initial_condition = 10
t0 = 0
t_max = 4


def euler_ode(h, ode, initial_condition, t0, t_max):
    y_values = [initial_condition]
    t = t0
    while t < t_max:
        y_next = y_values[-1] + h * ode(t, y_values[-1])
        y_values.append(y_next)
        t += h
    return np.array(y_values)

def runge_kutta_ode(h, ode, initial_condition, t0, t_max):
    y_values = [initial_condition]
    t = t0
    while t < t_max:
        k1 = h * ode(t, y_values[-1])
        k2 = h * ode(t + h/2, y_values[-1] + k1/2)
        k3 = h * ode(t + h/2, y_values[-1] + k2/2)
        k4 = h * ode(t + h, y_values[-1] + k3)
        y_next = y_values[-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
        y_values.append(y_next)
        t += h
    return np.array(y_values)

def exponential_growth_equation(y0, t):
    return y0 * np.exp(r * t)

def logistic_growth_equation(y0, t):
    return (y0*k*m.exp(k*t))/((k-y0) + y0*m.exp(k*t))

def exponential_ode(t, y):
    return r * y

def calculate_average_error(approximation, exact_solution):
    return np.mean(np.abs(approximation - exact_solution))

# Tiempo para graficar
t_values = np.arange(t0, t_max + h, h)

# Aproximación con el método de Euler
euler_solution = euler_ode(h, exponential_ode, initial_condition, t0, t_max)
euler_exact_solution = exponential_growth_equation(t_values, initial_condition, 0.1)
euler_error = calculate_average_error(euler_solution, euler_exact_solution)
euler_cumulative_error = np.cumsum(np.abs(euler_solution - euler_exact_solution))

# Aproximación con el método de Runge-Kutta de cuarto orden (RK4)
rk4_solution = runge_kutta_ode(h, exponential_ode, initial_condition, t0, t_max)
rk4_exact_solution = exponential_growth_equation(t_values, initial_condition, 0.1)
rk4_error = calculate_average_error(rk4_solution, rk4_exact_solution)
rk4_cumulative_error = np.cumsum(np.abs(rk4_solution - rk4_exact_solution))

# Graficar los errores acumulativos
plt.figure(figsize=(10, 6))
plt.plot(t_values, euler_cumulative_error, label=f'Error acumulado Euler ({euler_error:.4f})')
plt.plot(t_values, rk4_cumulative_error, label=f'Error acumulado RK4 ({rk4_error:.4f})')
plt.title('Comparación de errores acumulativos entre el método de Euler y RK4')
plt.xlabel('Tiempo')
plt.ylabel('Error Acumulado')
plt.legend()
plt.grid(True)
plt.show()