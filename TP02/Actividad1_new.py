import numpy as np
import matplotlib.pyplot as plt
import math as m

#constantes
k = 1000
y0 = 10
r = 0.3
h = 0.5
initial_condition = 500
t0 = 0
t_max = 10
time = np.arange(t0, t_max, h)


def euler_ode(h, ode, initial_condition, time):
    y_values = [initial_condition]
    for i in range(1, len(time)):
        y_next = y_values[-1] + h * ode(time[i], y_values[-1])
        y_values.append(y_next)
    return np.array(y_values)

def runge_kutta_2_ode(h, ode, initial_condition, time):
    y_values = [initial_condition]
    for i in range(1, len(time)):
        k1 = h * ode(time[i], y_values[-1])
        k2 = h * ode(time[i] + h, y_values[-1] + k1)
        y_next = y_values[-1] + 0.5 * (k1 + k2)
        y_values.append(y_next)
    return np.array(y_values)

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

def exponential_growth_equation(y0, t):
    return y0 * np.exp(r * t)

def logistic_growth_equation(y0, t):
    return (y0*k*np.exp(r*t))/((k-y0) + y0*np.exp(r*t))

def exponential_ode(t, y):
    return r * y

def logistic_ode(t, y):
    return r * y * ((k - y)/k)

# def calculate_average_error(acumulate_error):
#     average_error = 0
#     for i in range(len(acumulate_error)):
#         average_error += acumulate_error[i]

#     return average_error/len(acumulate_error)

def calculate_average_error(acumulate_error):
    return sum(acumulate_error) / len(acumulate_error)

def calculate_cumulative_error(approximate_solution, exact_solution):
    cumulative_error = []
    for i in range(len(approximate_solution)):
        cumulative_error.append(abs(approximate_solution[i] - exact_solution[i]))

    return cumulative_error

def calculate_average_relative_error(acumulate_error, exact_solution):
    relative_errors = [abs(error / real_value) for error, real_value in zip(acumulate_error, exact_solution)]
    return sum(relative_errors) / len(relative_errors)

# Tiempo para graficar
# Aproximación con el método de Euler
euler_solution = euler_ode(h, exponential_ode, initial_condition, time)
euler_exact_solution = exponential_growth_equation( initial_condition, time)
euler_cumulative_error = calculate_cumulative_error(euler_solution, euler_exact_solution)
euler_error = calculate_average_error(euler_cumulative_error)

#Approximación con el método de Runge-Kutta de segundo orden
rk2_solution = runge_kutta_2_ode(h, exponential_ode, initial_condition, time)
rk2_exact_solution = exponential_growth_equation(initial_condition, time)
rk2_cumulative_error = calculate_cumulative_error(rk2_solution, rk2_exact_solution)
rk2_error = calculate_average_error(rk2_cumulative_error)

# Aproximación con el método de Runge-Kutta de cuarto orden (RK4)
rk4_solution = runge_kutta_4_ode(h, exponential_ode, initial_condition, time)
rk4_exact_solution = exponential_growth_equation(initial_condition, time)
rk4_cumulative_error = calculate_cumulative_error(rk4_solution, rk4_exact_solution)
rk4_error = calculate_average_error(rk4_cumulative_error)

#Aproximación de la ecuación logística y el método de Euler
logistic_solution_euler = euler_ode(h, logistic_ode, initial_condition, time)
logistic_exact_solution = logistic_growth_equation(initial_condition, time)
logistic_cumulative_error = calculate_cumulative_error(logistic_solution_euler, logistic_exact_solution)
logistic_error = calculate_average_error(logistic_cumulative_error)

#Aproximación de la ecuación logística y el método de Runge-Kutta de segundo orden
logistic_solution_rk2 = runge_kutta_2_ode(h, logistic_ode, initial_condition, time)
logistic_exact_solution_rk2 = logistic_growth_equation(initial_condition, time)
logistic_cumulative_error_rk2 = calculate_cumulative_error(logistic_solution_rk2, logistic_exact_solution_rk2)
logistic_error_rk2 = calculate_average_error(logistic_cumulative_error_rk2)

#Aproximación de la ecuación logística y el método de Runge-Kutta de cuarto orden (RK4)
logistic_solution_rk4 = runge_kutta_4_ode(h, logistic_ode, initial_condition, time)
logistic_exact_solution_rk4 = logistic_growth_equation(initial_condition, time)
logistic_cumulative_error_rk4 = calculate_cumulative_error(logistic_solution_rk4, logistic_exact_solution_rk4)
logistic_error_rk4 = calculate_average_error(logistic_cumulative_error_rk4)



# #Graficar las aproximaciones a la exponencial con los métodos de Euler y RK2 y RK4
# plt.figure(figsize=(10, 6))
# plt.plot(time, euler_solution, label='Aproximación Euler') 
# plt.plot(time, rk2_solution, label='Aproximación RK2')
# plt.plot(time, rk4_solution, label='Aproximación RK4')
# plt.title('Comparación entre las aproximaciones de la solución exponencial')
# plt.xlabel('Tiempo')
# plt.ylabel('Tamaño Poblacional')
# plt.legend()
# plt.grid(True)
# plt.show()
#Graficar las aproximaciones con la logistica con los metodos de Euler y RK2 y RK4
plt.figure(figsize=(10, 6))
plt.plot(time, logistic_solution_euler, label='Aproximación Euler')
plt.plot(time, logistic_solution_rk2, label='Aproximación RK2')
plt.plot(time, logistic_solution_rk4, label='Aproximación RK4')
plt.title('Comparación entre las aproximaciones de la solución logística')
plt.xlabel('Tiempo')
plt.ylabel('Tamaño Poblacional')
plt.legend()
plt.grid(True)
plt.show()

# #Graficar las aproximaciones con los métodos de Euler y RK4 y la solución exacta exponencial
# plt.figure(figsize=(10, 6))
# plt.plot(time, euler_exact_solution, label='Solución exacta')
# plt.plot(time, euler_solution, label='Aproximación Euler')
# plt.plot(time, rk2_solution, label='Aproximación RK2')
# plt.plot(time, rk4_solution, label='Aproximación RK4')
# plt.title('Comparación entre las aproximaciones y la solución exacta')
# plt.xlabel('Tiempo')
# plt.ylabel('Tamaño Poblacional')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Graficar los errores acumulativos de la aproximación de la ecuación exponencial
# plt.figure(figsize=(10, 6))
# plt.plot(time, euler_cumulative_error, label=f'Error promedio Euler ({euler_error:.4f})')
# plt.plot(time, rk2_cumulative_error, label=f'Error promedio RK2 ({rk2_error:.4f})')
# plt.plot(time, rk4_cumulative_error, label=f'Error proemdio RK4 ({rk4_error:.4f})')
# plt.title('Comparación de errores acumulativos entre el método de Euler y RK4')
# plt.xlabel('Tiempo')
# plt.ylabel('Error Acumulado')
# plt.legend()
# plt.grid(True)
# plt.show()

# #Graficar las aproximaciones con los métodos de Euler y RK4 y la solución exacta logística
# plt.figure(figsize=(10, 6))
# plt.plot(time, logistic_exact_solution, label='Solución exacta')
# plt.plot(time, logistic_solution_euler, label='Aproximación Euler')
# plt.plot(time, logistic_solution_rk2, label='Aproximación RK2')
# plt.plot(time, logistic_solution_rk4, label='Aproximación RK4')
# plt.title('Comparación entre las aproximaciones y la solución exacta')
# plt.xlabel('Tiempo')
# plt.ylabel('Tamaño Poblacional')
# plt.legend()
# plt.grid(True)
# plt.show()

# Graficar los errores acumulativos de la aproximación de la ecuación logística
plt.figure(figsize=(10, 6))
plt.plot(time, logistic_cumulative_error, label=f'Error promedio Euler ({logistic_error:.4f})')
plt.plot(time, logistic_cumulative_error_rk2, label=f'Error promedio RK2 ({logistic_error_rk2:.4f})')
plt.plot(time, logistic_cumulative_error_rk4, label=f'Error promedio RK4 ({logistic_error_rk4:.4f})')
plt.title('Comparación de errores acumulativos entre el método de Euler, RK2 Y RK4 para solución logística')
plt.xlabel('Tiempo')
plt.ylabel('Error Acumulado')
plt.legend()
plt.grid(True)
plt.show()

# Graficar los errores acumulativos de la aproximación de la ecuación exponencial
plt.figure(figsize=(10, 6))
plt.plot(time, euler_cumulative_error, label=f'Error promedio Euler ({euler_error:.4f})')
plt.plot(time, rk2_cumulative_error, label=f'Error promedio RK2 ({rk2_error:.4f})')
plt.plot(time, rk4_cumulative_error, label=f'Error promedio RK4 ({rk4_error:.4f})')
plt.title('Comparación de errores acumulativos entre el método de Euler, RK2 Y RK4 para solución exponencial')
plt.xlabel('Tiempo')
plt.ylabel('Error Acumulado')
plt.legend()
plt.grid(True)
plt.show()


# #quiero hacer un grafico del error promedio vs el valor de h para el metodo de runge kuta aproximando la ecuacion logistica
h_values = [0.001, 0.01, 0.1, 0.5]


#aca pongo el porque elegi estos valores
rk4_error_values = []
rk2_error_values = []
euler_error_values = []
for j in h_values:
    time = np.arange(t0, t_max, j)
    rk4_solution = runge_kutta_4_ode(j, logistic_ode, initial_condition, time)
    rk4_exact_solution = logistic_growth_equation(initial_condition, time)
    rk4_cumulative_error = calculate_cumulative_error(rk4_solution, rk4_exact_solution)
    rk4_error = calculate_average_relative_error(rk4_cumulative_error, rk4_exact_solution)
    rk4_error_values.append(rk4_error)

    rk2_solution = runge_kutta_2_ode(j, logistic_ode, initial_condition, time)
    rk2_exact_solution = logistic_growth_equation(initial_condition, time)
    rk2_cumulative_error = calculate_cumulative_error(rk2_solution, rk2_exact_solution)
    rk2_error = calculate_average_relative_error(rk2_cumulative_error, rk2_exact_solution)
    rk2_error_values.append(rk2_error)

    euler_solution = euler_ode(j, logistic_ode, initial_condition, time)
    euler_exact_solution = logistic_growth_equation(initial_condition, time)
    euler_cumulative_error = calculate_cumulative_error(euler_solution, euler_exact_solution)
    euler_error = calculate_average_relative_error(euler_cumulative_error, euler_exact_solution)
    euler_error_values.append(euler_error)

plt.figure(figsize=(10, 6))
plt.plot(h_values, rk4_error_values, label='Error promedio RK4')
plt.plot(h_values, rk2_error_values, label='Error promedio RK2')
plt.plot(h_values, euler_error_values, label='Error promedio Euler')
plt.title('Error promedio realtivo vs valor de h')
plt.xlabel('Valor de h')
plt.ylabel('Error promedio')
plt.legend()
plt.grid(True)
plt.show()





