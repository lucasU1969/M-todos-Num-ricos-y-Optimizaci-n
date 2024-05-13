import numpy as np
import matplotlib.pyplot as plt
# from scipy.integrate import odeint


# Función que define el sistema de ecuaciones diferenciales
def model(X:list[float, float], t, r1, r2, K1, K2, alpha12, alpha21):
    N1, N2 = X
    dN1dt = (r1 * N1 * (K1 - N1 - (alpha12 * N2)) )/ K1
    dN2dt = (r2 * N2 * (K2 - N2 - (alpha21 * N1)) )/ K2
    return [dN1dt, dN2dt]

def dN1dt(N1, N2, r1, K1, alpha12):
    return (r1 * N1 * (K1 - N1 - alpha12 * N2)) / K1

def dN2dt(N1, N2, r2, K2, alpha21):
    return (r2 * N2 * (K2 - N2 - alpha21 * N1)) / K2

def runge_kutta_ode_sys(h:float, odes:list, initial_conditions:list[float, float], t0:float, t_max:float):
    approximations = [initial_conditions] ## [[N1_0, N2_0]]
    t = t0
    while t < t_max:
        k1 = [h * ode(t, approximations[-1][0], approximations[-1][1]) for ode in odes]
        k2 = [h * ode(t + h/2, approximations[-1][0] + k1[0]/2, approximations[-1][1] + k1[1]/2) for ode in odes]
        k3 = [h * ode(t + h/2, approximations[-1][0] + k2[0]/2, approximations[-1][1] + k2[1]/2) for ode in odes]
        k4 = [h * ode(t + h, approximations[-1][0] + k3[0], approximations[-1][1] + k3[1]) for ode in odes]
        next_approximation = [approximations[-1][i] + (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6 for i in range(2)]
        approximations.append(next_approximation)
        t += h
    return np.array(approximations)

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

def N1_isocline(N1, K1, alpha12):
    return (-N1 + K1) / alpha12

def N2_isocline(N1, K2, alpha21):
    return K2 - (alpha21 * N1)

def plot_isoclines(K1, K2, alpha12, alpha21):
    N1 = np.linspace(0, K1, 100) # tomo algunos valores de N1
    plt.plot(N1, N1_isocline(N1, K1, alpha12), label='dN1/dt = 0')
    plt.plot(N1, N2_isocline(N1, K2, alpha21), label='dN2/dt = 0')
    plt.xlabel('N1')
    plt.ylabel('N2')

# Ejemplo de uso
r1 = 1.2
r2 = 3
K1 = 100
K2 = 80
alpha21 = 0.1
alpha12 = 0.3

# Ejemplo de uso de runge kutta
N10 = 60
N20 = 70
t0 = 0
tmax = 50
h = 1

odes = [lambda n1, n2 : dN1dt(n1, n2, r1, K1, alpha12), lambda n1, n2 : dN2dt(n1, n2, r2, K2, alpha21)]
initial_conditions = [N10, N20]
approximations = rk4_for_ode_system(h, odes, initial_conditions, t0, tmax)
plt.plot(np.linspace(t0, tmax, int(tmax/h)), approximations[:, 0], label='N1')
plt.plot(np.linspace(t0, tmax, int(tmax/h)), approximations[:, 1], label='N2')
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.title('Evolución de las poblaciones en el tiempo')
plt.legend()
plt.show()


# Grafico de las aproximaciones
N1 = approximations[:, 0]
N2 = approximations[:, 1]
xs = np.linspace(0, tmax, len(N1))
fig, ax = plt.subplots(1, 1)
plt.plot(xs, N1, label='N1')
plt.plot(xs, N2, label='N2')
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.legend()
plt.show()

# # Graficos de los diagramas de fase variando el K1
K1s = np.linspace(0, 100    , 5)
fig, ax = plt.subplots(1, 1)
# for K1 in K1s:
#     approximations = runge_kutta_ode_sys(h, odes, initial_conditions, t0, tmax)
#     N1 = approximations[:, 0]
#     N2 = approximations[:, 1]
#     plt.plot(N1, N2, label=f'K1 = {K1}')
# plt.xlabel('N1')
# plt.ylabel('N2')
# plt.legend()
# plt.show()

# graficos de los diagramas de fase variando el K2
K2s = np.linspace(0, 100, 5)
fig, ax = plt.subplots(1, 1)
# for K2 in K2s:
#     approximations = runge_kutta_ode_sys(h, odes, initial_conditions, t0, tmax)
#     N1 = approximations[:, 0]
#     N2 = approximations[:, 1]
#     plt.plot(N1, N2, label=f'K2 = {K2}')
# plt.xlabel('N1')
# plt.ylabel('N2')
# plt.legend()
# plt.show()

# # graficos de los diagramas de fase variando el alpha12
alpha12s = np.linspace(-1, 1, 5)
fig, ax = plt.subplots(1, 1)
# for alpha12 in alpha12s:
#     approximations = runge_kutta_ode_sys(h, odes, initial_conditions, t0, tmax)
#     N1 = approximations[:, 0]
#     N2 = approximations[:, 1]
#     plt.plot(N1, N2, label=f'alpha12 = {alpha12}')
# plt.xlabel('N1')
# plt.ylabel('N2')
# plt.legend()
# plt.show()

# # graficos de los diagramas de fase variando el alpha21
alpha21s = np.linspace(-1, 1, 5)
fig, ax = plt.subplots(1, 1)
# for alpha21 in alpha21s:
#     approximations = runge_kutta_ode_sys(h, odes, initial_conditions, t0, tmax)
#     N1 = approximations[:, 0]
#     N2 = approximations[:, 1]
#     plt.plot(N1, N2, label=f'alpha21 = {alpha21}')
# plt.xlabel('N1')
# plt.ylabel('N2')
# plt.legend()
# plt.show()

# # Gráfico de los diagramas de fase variando el N10 
N10s = np.linspace(0, 100, 5)
fig, ax = plt.subplots(1, 1)
# for N10 in N10s:
#     approximations = runge_kutta_ode_sys(h, odes, [N10, N20], t0, tmax)
#     N1 = approximations[:, 0]
#     N2 = approximations[:, 1]
#     plt.plot(N1, N2, label=f'N10 = {N10}')
# plt.title('Diagrama de fase variando N10')
# plt.xlabel('N1')
# plt.ylabel('N2')
# plt.legend()
# plt.show()

# # Gráfico de los diagramas de fase variando el N20
r1 = 0.1
r2 = 0.3
K1 = 100
K2 = 80
alpha21 = 0.1
alpha12 = 1
N20s = np.linspace(0, 100, 5)
# fig, ax = plt.subplots(1, 1)
# for N20 in N20s:
#     approximations = runge_kutta_ode_sys(h, odes, [N10, N20], t0, tmax)
#     N1 = approximations[:, 0]
#     N2 = approximations[:, 1]
#     plt.plot(N1, N2, label=f'N20 = {N20}')
# plt.title('Diagrama de fase variando N20')
# plt.xlabel('N1')
# plt.ylabel('N2')
# plt.legend()
# plt.show()

# Gráfico de los diagramas de fase variando el N10 y N20
# Definir los parámetros del modelo
r1 = 0.1
r2 = 0.3
K1 = 100
K2 = 80
alpha21 = 0.1
alpha12 = 1
t0, tmax = 0.0, 300.0
h = 5

# Definir las funciones ODEs
odes = [lambda t, N1, N2: dN1dt(N1, N2, r1, K1, alpha12), lambda t, N1, N2: dN2dt(N1, N2, r2, K2, alpha21)]

# Crear una malla de condiciones iniciales
N10_grid, N20_grid = np.mgrid[0:100:10j, 0:100:10j]

fig, ax = plt.subplots()

for N10, N20 in zip(N10_grid.flatten(), N20_grid.flatten()):
    initial_conditions = [N10, N20]
    approximations = runge_kutta_ode_sys(h, odes, initial_conditions, t0, tmax)
    N1 = approximations[:, 0]
    N2 = approximations[:, 1]

    # Graficar la trayectoria
    ax.plot(N1, N2, 'k-', lw=0.5, alpha=0.5)

    # Agregar flechas para indicar la dirección del tiempo
    for i in range(len(N1) - 1):
        dx = N1[i + 1] - N1[i]
        dy = N2[i + 1] - N2[i]
        if i % 20 == 0:  # Add arrows at every 5th point
            ax.annotate('', xy=(N1[i], N2[i]), xytext=(N1[i + 1], N2[i + 1]),
                    arrowprops=dict(arrowstyle='<-', color='r', lw=0.5))
# sobre el mismo gráfico grafico las isoclinas
plot_isoclines(K1, K2, alpha12, alpha21)
# punto de equilibrio
plt.scatter( (K1 - alpha12*K2)/(1 - alpha12*alpha21), K2 - alpha21*(K1 - alpha12*K2)/(1 - alpha12*alpha21), color='red', label='Punto de equilibrio')
ax.set_title('Diagrama de fase variando N10 y N20')
ax.set_xlabel('N1')
ax.set_ylabel('N2')
ax.set_xlim(0, K1)
ax.set_ylim(0, K2)
plt.show()

# Grafico de las isoclinas
fig, ax = plt.subplots(1, 1)
plot_isoclines(K1, K2, alpha12, alpha21)
plt.scatter( (K1 - alpha12*K2)/(1 - alpha12*alpha21), K2 - alpha21*(K1 - alpha12*K2)/(1 - alpha12*alpha21), color='red', label='Punto de equilibrio')
plt.title('Isoclinas')
plt.legend()
plt.show()

#cuatro tipos de gráficos de isoclinas variando los parámetros
fig, ax = plt.subplots(1, 4)
ax[0].plot(np.linspace(0, 100, 100), N1_isocline(np.linspace(0, 100, 100), 100, 0.1))
ax[0].plot(np.linspace(0, 100, 100), N2_isocline(np.linspace(0, 100, 100), 80, 0.1))
ax[0].set_title('K1 = 100, K2 = 80, alpha12 = 0.1, alpha21 = 0.1')
ax[1].plot(np.linspace(0, 100, 100), N1_isocline(np.linspace(0, 100, 100), 100, 0.1))
ax[1].plot(np.linspace(0, 100, 100), N2_isocline(np.linspace(0, 100, 100), 80, 1))
ax[1].set_title('K1 = 100, K2 = 80, alpha12 = 0.1, alpha21 = 1')
ax[2].plot(np.linspace(0, 100, 100), N1_isocline(np.linspace(0, 100, 100), 100, 1))
ax[2].plot(np.linspace(0, 100, 100), N2_isocline(np.linspace(0, 100, 100), 80, 0.1))
ax[2].set_title('K1 = 100, K2 = 80, alpha12 = 1, alpha21 = 0.1')
ax[3].plot(np.linspace(0, 100, 100), N1_isocline(np.linspace(0, 100, 100), 100, 1))
ax[3].plot(np.linspace(0, 100, 100), N2_isocline(np.linspace(0, 100, 100), 80, 1))
ax[3].set_title('K1 = 100, K2 = 80, alpha12 = 1, alpha21 = 1')
plt.show()

#isoclinas variando K1
fig, ax = plt.subplots(1, 1)
for K1 in K1s:
    plt.plot(np.linspace(0, K1, 100), N1_isocline(np.linspace(0, K1, 100), K1, alpha12), label=f'K1 = {K1}')
plt.xlabel('N1')
plt.ylabel('N2')
plt.title('Isoclinas de N1 variando K1')
plt.legend()
plt.show()

#isoclinas variando alpha12
fig, ax = plt.subplots(1, 1)
for alpha12 in alpha12s:
    plt.plot(np.linspace(0, K1, 100), N1_isocline(np.linspace(0, K1, 100), K1, alpha12), label=f'alpha12 = {alpha12}')
plt.xlabel('N1')
plt.ylabel('N2')
plt.title('Isoclinas de N1 variando alpha12')
plt.legend()
plt.show()

# variar ninguno de los parámetros K2 y N2 afecta a la isoclina de N1

# Ahora voy a ver como cambia el diagrama de fase dependiendo de la variación de los parámetros

# Diagrama de fase
fig, ax = plt.subplots(1, 1)
plt.plot(N1, N2)
plt.xlabel('N1')
plt.ylabel('N2')
plt.show()

# campo vectorial de como se mueven las soluciones sobre las isoclinas
# tomo una grilla de condiciones iniciales y grafico el resultado del vector que me da la aproximación por runge kutta en ese punto. 
r1 = 0.1
r2 = 0.3
K1 = 100
K2 = 80
alpha21 = 0.1
alpha12 = 1
fig, ax = plt.subplots(1, 1)
N1s = np.linspace(0, 100, 15)
N2s = np.linspace(0, 100, 15)
N1s, N2s = np.meshgrid(N1s, N2s)
for i in range(15):
    for j in range(15):
        approximations = runge_kutta_ode_sys(h, odes, [N1s[i, j], N2s[i, j]], t0, tmax)
        plt.quiver(N1s[i, j], N2s[i, j], approximations[-1][0] - N1s[i, j], approximations[-1][1] - N2s[i, j])
#sobre este gráfico tengo que graficar las isoclinas
plot_isoclines(K1, K2, alpha12, alpha21)
# punto de equilibrio
plt.scatter( (K1 - alpha12*K2)/(1 - alpha12*alpha21), K2 - alpha21*(K1 - alpha12*K2)/(1 - alpha12*alpha21), color='red', label='Punto de equilibrio')
plt.xlabel('N1')
plt.ylabel('N2')
plt.title('Campo vectorial')
plt.legend()
plt.show()

# hasta acá no estaba comentado ----------------------------------------------

# Trayectorias en función del tiempo para distintas condiciones iniciales N1
# fig, ax = plt.subplots(1, 1)
# N1s = np.linspace(0, 100, 5)
# t = np.linspace(t0, tmax, int(tmax/h))
# for N1 in N1s:
#     approximations = runge_kutta_ode_sys(h, odes, [N1, N20], t0, tmax)
#     plt.plot(t, approximations[:, 1], label=f'N1 = {N1}')
# plt.xlabel('Tiempo')
# plt.ylabel('N2')
# plt.legend()
# plt.show()

# como cambian las distintas condiciones iniciales de N1 a la trayectoria en función del tiempo de N1
# fig, ax = plt.subplots(1, 1)
# N1s = np.linspace(0, 100, 5)
# t = np.linspace(t0, tmax, int(tmax/h))
# for N1 in N1s:
#     approximations = runge_kutta_ode_sys(h, odes, [N1, N20], t0, tmax)
#     plt.plot(t, approximations[:, 0], label=f'N1 = {N1}')
# plt.xlabel('Tiempo')
# plt.ylabel('N1')
# plt.legend()
# plt.show()


#como cambian las distintas condiciones iniciales de N1 a la trayecotoria en función del tiempo de N2
# fig, ax = plt.subplots(1, 1)
# N1s = np.linspace(0, 100, 5)
# t = np.linspace(t0, tmax, int(tmax/h))
# for N1 in N1s:
#     approximations = runge_kutta_ode_sys(h, odes, [N1, N20], t0, tmax)
#     plt.plot(t, approximations[:, 1], label=f'N1 = {N1}')
# plt.xlabel('Tiempo')
# plt.ylabel('N2')
# plt.legend()
# plt.show()

# ------------------------------------------------------------------------------------------------------- este es un gráfico aparte

# # isclinas variando los parametros
# fig, ax = plt.subplots(1, 1)
# for K1 in K1s:
#     for K2 in K2s:
#         for alpha12 in alpha12s:
#             for alpha21 in alpha21s:
#                 plot_isoclines(K1, K2, alpha12, alpha21)
# plt.xlabel('N1')
# plt.ylabel('N2')
# plt.show()

