import numpy as np
import matplotlib.pyplot as plt
import math as m

# Constantes
r = 0.1  # Tasa instantánea de crecimiento per cápita
K = 1000  # Capacidad de carga del ambiente
N0 = 10  # Población inicial
h = 0.1  # Tamaño del paso
t0 = 0  # Tiempo inicial
t_max = 10  # Tiempo final

def euler_ode(h, ode, initial_condition, t0, t_max):
    """
    Función que realiza la aproximación de una ODE utilizando el método de Euler.

    Args:
    - h: Tamaño de paso.
    - ode: Función que representa la ODE. Debe tener la forma ode(t, y), donde t es el tiempo y y es la variable dependiente.
    - initial_condition: Valor inicial de la variable dependiente.
    - t0: Tiempo inicial.
    - t_max: Tiempo máximo hasta el cual se realiza la aproximación.

    Returns:
    - Array con los valores aproximados de la variable dependiente a lo largo del tiempo.
    """
    y_values = [initial_condition]
    t = t0
    while t < t_max:
        y_next = y_values[-1] + h * ode(t, y_values[-1])
        y_values.append(y_next)
        t += h
    return np.array(y_values)

def runge_kutta_ode(h, ode, initial_condition, t0, t_max):
    """
    Función que realiza la aproximación de una ODE utilizando el método de Runge-Kutta de cuarto orden (RK4).

    Args:
    - h: Tamaño de paso.
    - ode: Función que representa la ODE. Debe tener la forma ode(t, y), donde t es el tiempo y y es la variable dependiente.
    - initial_condition: Valor inicial de la variable dependiente.
    - t0: Tiempo inicial.
    - t_max: Tiempo máximo hasta el cual se realiza la aproximación.

    Returns:
    - Array con los valores aproximados de la variable dependiente a lo largo del tiempo.
    """
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

def exponential_ode(t, N):
    return r * N

def logistic_ode(t, N):
    return (r * N * (K - N)) / K

def exponential_growth_solution(t):
    return N0 * m.exp(r * t)

def logistic_growth_solution(t):
    return (K * N0 * m.exp(r * t)) / (K + N0 * (m.exp(r * t) - 1))

def main():
    t_values = np.arange(t0, t_max + h, h)
    exponential_solution = np.array([exponential_growth_solution(t) for t in t_values])
    logistic_solution = np.array([logistic_growth_solution(t) for t in t_values])

    # Exponential growth
    exponential_euler = euler_ode(h, exponential_ode, N0, t0, t_max)
    exponential_runge_kutta = runge_kutta_ode(h, exponential_ode, N0, t0, t_max)

    # Logistic growth
    logistic_euler = euler_ode(h, logistic_ode, N0, t0, t_max)
    logistic_runge_kutta = runge_kutta_ode(h, logistic_ode, N0, t0, t_max)

    plt.plot(t_values, exponential_solution, label='Exponential growth solution')
    plt.plot(t_values, logistic_solution, label='Logistic growth solution')
    plt.plot(t_values, exponential_euler, label='Euler exponential growth')
    plt.plot(t_values, exponential_runge_kutta, label='Runge-Kutta exponential growth')
    plt.plot(t_values, logistic_euler, label='Euler logistic growth')
    plt.plot(t_values, logistic_runge_kutta, label='Runge-Kutta logistic growth')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('Population Growth Comparison')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
