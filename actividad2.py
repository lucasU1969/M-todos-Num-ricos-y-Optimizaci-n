import numpy as np
import matplotlib.pyplot as plt
import math as m


# Constantes
r1 = 0.1  # Tasa instantánea de crecimiento per cápita de la primera especie
r2 = 0.1  # Tasa instantánea de crecimiento per cápita de la segunda especie
K1 = 1000  # Capacidad de carga del ambiente de la primera especie
K2 = 1000  # Capacidad de carga del ambiente de la segunda especie
h = 0.1  # Tamaño del paso
t0 = 0  # Tiempo inicial
t_max = 10  # Tiempo final
alpha12 = 0.01  # coeficiente de competencia interespecífica de la primera especie sobre la segunda
alpha21 = 0.01  # coeficiente de competencia interespecífica de la segunda especie sobre la primera
N1_0 = 100  # Población inicial de la primera especie
N2_0 = 100  # Población inicial de la segunda especie

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


def lotka_volterra_1(N1, N2):
    return (r1*N1*(K1-N1-alpha12*N2))/K1

def lotka_volterra_2(N1, N2):
    return (r2*N2*(K2-N2-alpha21*N1))/K2

# Aproximación de la ODE



