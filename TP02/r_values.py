# Consideremos ahora que hay dos especies que no compiten entre sí por los recursos disponible en el
# sistema sino que una es la presa (N) y la otra es la predadora (P). Para las presas el número de individuos es
# N y su tasa de crecimiento r. Para el caso de los predadores el número de individuos es P, la eficiencia de
# captura α, la eficiencia para convertir presas en nuevos predadores β y la tasa de mortalidad per cápita q. De
# esta forma se definen las ecuaciones de Predador-Presa de Lotka-Volterra

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


#Graficar N y P para distintos valores de r en el tiempo
r_values = np.linspace(0.1, 1, 10)
N0 = 10
P0 = 10
alpha = 0.1
beta = 0.1
q = 0.1
t0 = 0
tmax = 5
h = 0.01

colores = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple']
iter = 0
for r in r_values:
    odes = [lambda N, P : dNdt(N, P, r, alpha), lambda N, P : dPdt(N, P, beta, q)]
    initial_conditions = [N0, P0]
    approximations = rk4_for_ode_system(h, odes, initial_conditions, t0, tmax)
    plt.plot(np.linspace(t0, tmax, int(tmax/h) + 1), approximations[:, 0], color = colores[iter], label=f'r = {r}')
    plt.plot(np.linspace(t0, tmax, int(tmax/h) + 1), approximations[:, 1], color = colores[iter])
    iter += 1
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.title('Evolución de las poblaciones en el tiempo para distintos valores de r')
plt.legend()   
plt.show()
