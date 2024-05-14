import numpy as np
import matplotlib.pyplot as plt

def dN1dt(N1, N2, r1, K1, alpha12):
    return (r1 * N1 * (K1 - N1 - alpha12 * N2)) / K1

def dN2dt(N1, N2, r2, K2, alpha21):
    return (r2 * N2 * (K2 - N2 - alpha21 * N1)) / K2

def rk4_for_ode_system(h:float, odes:list, initial_conditions:list, t0:float, tmax:float):
    """
    Parameters:
    h: float
        Step size
    odes: list of functions
        List of functions that represent the system of ODEs
    initial_conditions: list of floats
        Initial conditions of the system
    t0: float
        Initial time
    tmax: float
        Final time
    
    Returns:
    approximations: np.array
        Array with the approximations of the system of ODEs

    """
    approximations = np.array(np.array([initial_conditions]))
    t = t0 + h
    while t < tmax:
        k1 = h*np.array([odes[0](approximations[-1][0], approximations[-1][1]), odes[1](approximations[-1][0], approximations[-1][1])])
        k2 = h*np.array([odes[0](approximations[-1][0] + k1[0]/2, approximations[-1][1] + k1[1]/2), odes[1](approximations[-1][0] + k1[0]/2, approximations[-1][1] + k1[1]/2)])
        k3 = h*np.array([odes[0](approximations[-1][0] + k2[0]/2, approximations[-1][1] + k2[1]/2), odes[1](approximations[-1][0] + k2[0]/2, approximations[-1][1] + k2[1]/2)])
        k4 = h*np.array([odes[0](approximations[-1][0] + k3[0], approximations[-1][1] + k3[1]), odes[1](approximations[-1][0] + k3[0], approximations[-1][1] + k3[1])])
        next_approximation = approximations[-1] + (k1 + 2*k2 + 2*k3 + k4)/6
        approximations = np.append(approximations, [next_approximation], axis=0)
        t += h
    return approximations

# Parámetros
r1_n = .5
r1_0 = 0
r1_p = -.5
r2 = 0.1
K1 = 1200
K2 = 800
alpha12 = 2.7
alpha21 = 1.5

# Condiciones iniciales
N1_0 = 100
N2_0 = 100
initial_conditions = [N1_0, N2_0]

# Tiempo
t0 = 0
tmax = 30
h = 0.005
t = np.linspace(t0, tmax, int(tmax/h) +1)

# Resolviendo el sistema de ecuaciones diferenciales
odes_n = [lambda N1, N2: dN1dt(N1, N2, r1_n, K1, alpha12), lambda N1, N2: dN2dt(N1, N2, r2, K2, alpha21)]
odes_0 = [lambda N1, N2: dN1dt(N1, N2, r1_0, K1, alpha12), lambda N1, N2: dN2dt(N1, N2, r2, K2, alpha21)]
odes_p = [lambda N1, N2: dN1dt(N1, N2, r1_p, K1, alpha12), lambda N1, N2: dN2dt(N1, N2, r2, K2, alpha21)]

approximations_n = rk4_for_ode_system(h, odes_n, initial_conditions, t0, tmax)
approximations_0 = rk4_for_ode_system(h, odes_0, initial_conditions, t0, tmax)
approximations_p = rk4_for_ode_system(h, odes_p, initial_conditions, t0, tmax)
print(len(approximations_0))

# Graficando
plt.figure()
plt.plot(t, approximations_n[:, 0],color='blue', label='N1 para r1 = .5')
plt.plot(t, approximations_n[:, 1],color='lightblue', label='N2 parar1 = .5')
plt.plot(t, approximations_0[:, 0], color='green', label='N1 parar1 = 0')
plt.plot(t, approximations_0[:, 1],color='lightgreen', label='N2 parar1 = 0')
plt.plot(t, approximations_p[:, 0],color='red', label='N1 parar1 = -.5')
plt.plot(t, approximations_p[:, 1],color='pink', label='N2 parar1 = -.5')
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.title('Variación del parámetro r1 dentro del caso 3')
plt.legend()
plt.show()


odes_01 = [lambda N1, N2: dN1dt(N1, N2, 0.1, K1, alpha12), lambda N1, N2: dN2dt(N1, N2, r2, K2, alpha21)]
# gráfico variando N1_0
N1_0 = 0
N2_0 = 100
initial_conditions = [N1_0, N2_0]
approximations_1 = rk4_for_ode_system(h, odes_01, initial_conditions, t0, tmax)
N1_0 = 600
N2_0 = 100
initial_conditions = [N1_0, N2_0]
approximations_2 = rk4_for_ode_system(h, odes_01, initial_conditions, t0, tmax)
N1_0 = 1800
N2_0 = 100
initial_conditions = [N1_0, N2_0]
approximations_3 = rk4_for_ode_system(h, odes_01, initial_conditions, t0, tmax)
plt.figure()
plt.plot(t, approximations_1[:, 0],color='blue', label='N1 para N1_0 = 0')
plt.plot(t, approximations_1[:, 1],color='lightblue', label='N2 para N1_0 = 0')
plt.plot(t, approximations_2[:, 0], color='green', label='N1 para N1_0 = 20')
plt.plot(t, approximations_2[:, 1],color='lightgreen', label='N2 para N1_0 = 20')
plt.plot(t, approximations_3[:, 0],color='red', label='N1 para N1_0 = 180')
plt.plot(t, approximations_3[:, 1],color='pink', label='N2 para N1_0 = 180')
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.title('Variación de N1_0 dentro del caso 3')
plt.yscale('symlog')
plt.legend()


plt.show()

