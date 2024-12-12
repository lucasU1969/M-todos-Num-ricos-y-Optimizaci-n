import numpy as np
import matplotlib.pyplot as plt


# Función que define el sistema de ecuaciones diferenciales
def model(X:list[float, float], r, N, P, alpha, beta, q):
    N, P = X
    dNdt = r*N - alpha*N*P
    dPdt = beta*N*P - q*P
    return [dNdt, dPdt]

def dNdt(N, P, r1, alpha):
    return r1*N - alpha*N*P

def dPdt(N, P, beta, q):
    return beta*N*P - q*P

def dNdt_extendido(N, P, r1, alpha, K):
    return r1*N - alpha*N*P - (r1*N**2)/K

def dPdt_extendido(N, P, beta, q):
    return beta*N*P - q*P

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


# Parámetros y condiciones iniciales
r1 = 0.5
q = 0.3
alpha = 0.1
beta = 0.3
K = 100  # Capacidad de carga para la competencia intraespecífica de las presas

N0 = q/beta + 1  # Condiciones iniciales para N (presas
P0 = r1/alpha + 1
t0 = 0
tmax = 100
h = 0.1

odes = [lambda N, P : dNdt(N, P, r1, alpha), lambda N, P : dPdt(N, P, beta, q)]
initial_conditions = [N0, P0]
approximations = rk4_for_ode_system(h, odes, initial_conditions, t0, tmax)
plt.plot(approximations[:, 0], approximations[:, 1])
plt.xlabel('N')
plt.ylabel('P')
plt.title('N vs P')
plt.show()

odes_extendido = [lambda N, P : dNdt_extendido(N, P, r1, alpha, K), lambda N, P : dPdt_extendido(N, P, beta, q)]
initial_conditions = [N0, P0]
approximations = rk4_for_ode_system(h, odes_extendido, initial_conditions, t0, tmax)
plt.plot(approximations[:, 0], approximations[:, 1])
plt.xlabel('N')
plt.ylabel('P')
plt.title('N vs P')
plt.show()
