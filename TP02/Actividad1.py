import numpy as np
import matplotlib.pyplot as plt
import math as m

#constantes
k = 100
y0 = 10
r = 0.3
h = 0.1
initial_condition = 10
t0 = 0
t_max = 100

time = np.arange(t0, t_max, h)


def euler_ode(h, ode, initial_condition, t0, t_max):
    y_values = [initial_condition]
    t = t0
    while t < t_max - h:
        y_next = y_values[-1] + h * ode(t, y_values[-1])
        y_values.append(y_next)
        t += h
    return np.array(y_values)


def runge_kutta_2_ode(h, ode, initial_condition, t0, t_max):
    y_values = [initial_condition]
    # t_values = [t0]
    t = t0
    while t < t_max - h:
        k1 = h * ode(t, y_values[-1])
        k2 = h * ode(t + h, y_values[-1] + k1)
        y_next = y_values[-1] + 0.5 * (k1 + k2)
        y_values.append(y_next)
        t += h
        # t_values.append(t)
    return np.array(y_values)



def runge_kutta_4_ode(h, ode, initial_condition, t0, t_max):
    y_values = [initial_condition]
    t = t0
    while t < t_max - h:
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
    return (y0*k*np.exp(r*t))/((k-y0) + y0*np.exp(r*t))

def exponential_ode(t, y):
    return r * y

def logistic_ode(t, y):
    return r * y * ((k - y)/k)

def calculate_average_error(acumulate_error):
    average_error = 0
    for i in range(len(acumulate_error)):
        average_error += acumulate_error[i]

    return average_error/len(acumulate_error)

def calculate_cumulative_error(approximate_solution, exact_solution):
    cumulative_error = []
    for i in range(len(approximate_solution)):
        cumulative_error.append(abs(approximate_solution[i] - exact_solution[i]))

    print(cumulative_error)
    return cumulative_error


# Tiempo para graficar
t_values = np.arange(t0, t_max + h, h)
print(len(t_values))
# Aproximación con el método de Euler
euler_solution = euler_ode(h, exponential_ode, initial_condition, t0, t_max)
euler_exact_solution = exponential_growth_equation( initial_condition, t_values)
euler_cumulative_error = calculate_cumulative_error(euler_solution, euler_exact_solution)
euler_error = calculate_average_error(euler_cumulative_error)

#Approximación con el método de Runge-Kutta de segundo orden
rk2_solution = runge_kutta_2_ode(h, exponential_ode, initial_condition, t0, t_max)
rk2_exact_solution = exponential_growth_equation(initial_condition, t_values)
rk2_cumulative_error = calculate_cumulative_error(rk2_solution, rk2_exact_solution)
rk2_error = calculate_average_error(rk2_cumulative_error)

# Aproximación con el método de Runge-Kutta de cuarto orden (RK4)
rk4_solution = runge_kutta_4_ode(h, exponential_ode, initial_condition, t0, t_max)
rk4_exact_solution = exponential_growth_equation(initial_condition, t_values)
rk4_cumulative_error = calculate_cumulative_error(rk4_solution, rk4_exact_solution)
rk4_error = calculate_average_error(rk4_cumulative_error)

#Aproximación de la ecuación logística y el método de Euler
logistic_solution_euler = euler_ode(h, logistic_ode, initial_condition, t0, t_max)
logistic_exact_solution = logistic_growth_equation(initial_condition, t_values)
logistic_cumulative_error = calculate_cumulative_error(logistic_solution_euler, logistic_exact_solution)
logistic_error = calculate_average_error(logistic_cumulative_error)

#Aproximación de la ecuación logística y el método de Runge-Kutta de segundo orden
logistic_solution_rk2 = runge_kutta_2_ode(h, logistic_ode, initial_condition, t0, t_max)
logistic_exact_solution_rk2 = logistic_growth_equation(initial_condition, t_values)
logistic_cumulative_error_rk2 = calculate_cumulative_error(logistic_solution_rk2, logistic_exact_solution_rk2)
logistic_error_rk2 = calculate_average_error(logistic_cumulative_error_rk2)

#Aproximación de la ecuación logística y el método de Runge-Kutta de cuarto orden (RK4)
logistic_solution_rk4 = runge_kutta_4_ode(h, logistic_ode, initial_condition, t0, t_max)
print(len(logistic_solution_rk4))
logistic_exact_solution_rk4 = logistic_growth_equation(initial_condition, t_values)
print(len(logistic_exact_solution_rk4))
logistic_cumulative_error_rk4 = calculate_cumulative_error(logistic_solution_rk4, logistic_exact_solution_rk4)
logistic_error_rk4 = calculate_average_error(logistic_cumulative_error_rk4)


#Graficar las aproximaciones con los métodos de Euler y RK2 y RK4
# plt.figure(figsize=(10, 6))
# plt.plot(t_values, euler_solution, label='Aproximación Euler') 
# plt.plot(t_values, rk2_solution, label='Aproximación RK2')
# plt.plot(t_values, rk4_solution, label='Aproximación RK4')
# plt.title('Comparación entre las aproximaciones')
# plt.xlabel('Tiempo')
# plt.ylabel('Tamaño Poblacional')
# plt.legend()
# plt.grid(True)
# plt.show()
#Graficar las aproximaciones con los métodos de Euler y RK4 y la solución exacta exponencial
plt.figure(figsize=(10, 6))
plt.plot(t_values, euler_exact_solution, label='Solución exacta')
plt.plot(t_values, euler_solution, label='Aproximación Euler')
plt.plot(t_values, rk2_solution, label='Aproximación RK2')
plt.plot(t_values, rk4_solution, label='Aproximación RK4')
plt.title('Comparación entre las aproximaciones y la solución exacta')
plt.xlabel('Tiempo')
plt.ylabel('Tamaño Poblacional')
plt.legend()
plt.grid(True)
plt.show()

# Graficar los errores acumulativos de la aproximación de la ecuación exponencial
plt.figure(figsize=(10, 6))
plt.plot(t_values, euler_cumulative_error, label=f'Error acumulado Euler ({euler_error:.4f})')
plt.plot(t_values, rk2_cumulative_error, label=f'Error acumulado RK2 ({rk2_error:.4f})')
plt.plot(t_values, rk4_cumulative_error, label=f'Error acumulado RK4 ({rk4_error:.4f})')
plt.title('Comparación de errores acumulativos entre el método de Euler y RK4')
plt.xlabel('Tiempo')
plt.ylabel('Error Acumulado')
plt.legend()
plt.grid(True)
plt.show()

#Graficar las aproximaciones con los métodos de Euler y RK4 y la solución exacta logística
plt.figure(figsize=(10, 6))
plt.plot(t_values, logistic_exact_solution, label='Solución exacta')
plt.plot(t_values, logistic_solution_euler, label='Aproximación Euler')
plt.plot(t_values, logistic_solution_rk2, label='Aproximación RK2')
plt.plot(t_values, logistic_solution_rk4, label='Aproximación RK4')
plt.title('Comparación entre las aproximaciones y la solución exacta')
plt.xlabel('Tiempo')
plt.ylabel('Tamaño Poblacional')
plt.legend()
plt.grid(True)
plt.show()

# Graficar los errores acumulativos de la aproximación de la ecuación logística
plt.figure(figsize=(10, 6))
plt.plot(t_values, logistic_cumulative_error, label=f'Error acumulado Euler ({logistic_error:.4f})')
plt.plot(t_values, logistic_cumulative_error_rk2, label=f'Error acumulado RK2 ({logistic_error_rk2:.4f})')
plt.plot(t_values, logistic_cumulative_error_rk4, label=f'Error acumulado RK4 ({logistic_error_rk4:.4f})')
plt.title('Comparación de errores acumulativos entre el método de Euler, RK2 Y RK4')
plt.xlabel('Tiempo')
plt.ylabel('Error Acumulado')
plt.legend()
plt.grid(True)
plt.show()

# #quiero hacer un grafico del error promedio vs el valor de h para el metodo de runge kuta aproximando la ecuacion logistica
h_values = np.arange(0.1, 2, 0.01)
rk4_error_values = []
for h in h_values:
    t_values2 = np.arange(t0, t_max + h, h)
    print(len(t_values2))
    rk4_solution = runge_kutta_4_ode(h, logistic_ode, initial_condition, t0, t_max)
    print(len(rk4_solution))
    rk4_exact_solution = logistic_growth_equation(initial_condition, t_values2)
    print(len(rk4_exact_solution))
    rk4_cumulative_error = calculate_cumulative_error(rk4_solution, rk4_exact_solution)
    rk4_error = calculate_average_error(rk4_cumulative_error)
    rk4_error_values.append(rk4_error)

plt.figure(figsize=(10, 6))
plt.plot(h_values, rk4_error_values, label='Error promedio')
plt.title('Error promedio vs valor de h')
plt.xlabel('Valor de h')
plt.ylabel('Error promedio')
plt.legend()
plt.grid(True)
plt.show()


