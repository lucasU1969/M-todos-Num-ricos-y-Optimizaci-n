import matplotlib.pyplot as plt
import numpy as np

# Define las funciones para las isoclinas
def dNdt(N, P, r1, alpha):
    return r1*N - alpha*N*P

def dPdt(N, P, beta, q):
    return beta*N*P - q*P

def dNdt_extendido(N, P, r1, alpha, K):
    return r1*N - alpha*N*P - (r1*N**2)/K

def dPdt_extendido(N, P, beta, q):
    return beta*N*P - q*P
# Caso 1: P = r/alpha y N = q1/beta1
def isocline_case1(N, r, alpha, beta1, q1):
    return (r / alpha) * np.ones_like(N), (q1 / beta1) * np.ones_like(N)

def isocline_extendido(r, N, K, alpha, q1, beta1):
    return (r / alpha) * (1 - N/K) * np.ones_like(N), (q1 / beta1) * np.ones_like(N)

def velocidad_isocline(N, P, r, alpha, beta1, q1):
    U = r*N - alpha*N*P
    V = beta1*N*P - q1*P
    return U, V

def velocidad_isocline_extendido(N, P, r, alpha, beta1, q1, K):
    U = r*N - alpha*N*P - (r*N**2)/K
    V = beta1*N*P - q1*P
    return U, V
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
# Definir parámetros genéricos
r = 0.1  # Tasa de crecimiento de las presas
alpha = 0.02  # Tasa de depredación de las presas por los predadores
beta1 = 0.4  # Tasa de crecimiento de los predadores
q1 = 0.8  # Tasa de mortalidad de los predadores
K1 = 10  # Capacidad de carga del ecosistema
# Generar valores de N
N = np.linspace(0, 20, 200)
P = np.linspace(0, 20, 200)

beta2 = 0.5
q2 = 8
K2 = 5

initial_conditions1 = [10, 5]
initial_conditions2 = [18, 15]
initial_conditions3 = [2, 10]
initial_conditions4 = [8, 8]
initial_conditions5 = [5, 5]
initial_conditions6 = [1, 9]
initial_conditions7 = [5, 2]
initial_conditions8 = [7, 1]

h = 0.1
t0 = 0
tmax = 100

# Calcular las isoclinas para cada caso
case1_P, case1_N = isocline_case1(N, r, alpha, beta1, q1)
case1_P_ext, case1_N_ext = isocline_extendido(r, N, K1, alpha, q1, beta1)
case2_P_ext, case2_N_ext = isocline_extendido(r, N, K2, alpha, q2, beta2)

# Graficar los cuatro casos en gráficos separados
plt.figure(figsize=(14, 10))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# Caso 1: P = r/alpha y N = q1/beta1

X, Y = np.meshgrid(np.linspace(0, 10, 10), np.linspace(0, 10, 10))
U , V = velocidad_isocline(X, Y, r, alpha, beta1, q1)
plt.streamplot(X, Y, U, V, color='black')

for i in range(0, 50):
    for j in range(0, 50):
        U_norm = (r*i - alpha*i*j) / np.sqrt((r*i - alpha*i*j)**2 + (beta1*i*j - q1*j)**2)
        V_norm = (beta1*i*j - q1*j) / np.sqrt((r*i - alpha*i*j)**2 + (beta1*i*j - q1*j)**2)
        plt.quiver([i], [j], U_norm, V_norm, color='pink')

odes = [lambda N, P : dNdt(N, P, r, alpha), lambda N, P : dPdt(N, P, beta1, q1)]
trayectorias1_kmenor = rk4_for_ode_system(h, odes, initial_conditions7, t0, tmax)
trayectorias2_kmenor = rk4_for_ode_system(h, odes, initial_conditions8, t0, tmax)
trayectorias3_kmenor = rk4_for_ode_system(h, odes, initial_conditions4, t0, tmax)
plt.plot(trayectorias1_kmenor[:, 0], trayectorias1_kmenor[:, 1], linewidth = 3)
plt.plot(trayectorias2_kmenor[:, 0], trayectorias2_kmenor[:, 1], linewidth = 3)
plt.plot(trayectorias3_kmenor[:, 0], trayectorias3_kmenor[:, 1], linewidth = 3)
plt.scatter(initial_conditions7[0], initial_conditions7[1], color='black', label='Condiciones iniciales', zorder = 2)
plt.scatter(initial_conditions8[0], initial_conditions8[1], color='black', zorder = 2)
plt.scatter(initial_conditions4[0], initial_conditions4[1], color='black', zorder = 2)

plt.xlim(0, 10)
plt.ylim(0, 10)
plt.plot(N, case1_P, color = 'red', label=r'Isocline de $P = \frac{r}{\alpha}$')
plt.plot(case1_N, N, color = 'blue', label=r'Isocline de $N = \frac{q1}{\beta1}$')
plt.scatter(case1_N, case1_P, color='green', label='Punto de intersección', zorder = 3)
plt.scatter(K1, 0, label = 'K1 = 10', color = 'purple', zorder = 3)
plt.xlabel('N (Presas)')
plt.ylabel('P (Predadores)')
plt.title('Caso 1')
plt.legend()
plt.xlim(0, None)
plt.ylim(0, None)
plt.text(-1, 5, r'$P = \frac{r}{\alpha}$', fontsize=12, ha='center')
plt.text(10, -1.5, r'$N = \frac{q1}{\beta1}$' , fontsize=12, ha='center')
plt.xticks([])
plt.yticks([])
plt.grid(True)
plt.show()

#Grafico caso 1 extendido
plt.figure(figsize=(14, 10))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# Caso 1: P = r/alpha y N = q1/beta1
X, Y = np.meshgrid(np.linspace(0, 10, 20), np.linspace(0, 10, 20))
U , V = velocidad_isocline_extendido(X, Y, r, alpha, beta1, q1, K1)

# Trazar las líneas de flujo
plt.streamplot(X, Y, U, V, color='black')
for i in range(0, 20):
    for j in range(0, 20):
        U_norm = (r*i - alpha*i*j - (r*i**2)/K1) / np.sqrt((r*i - alpha*i*j - (r*i**2)/K1)**2 + (beta1*i*j - q1*j)**2)
        V_norm = (beta1*i*j - q1*j) / np.sqrt((r*i - alpha*i*j)**2 + (beta1*i*j - q1*j)**2)
        plt.quiver([i], [j], U_norm, V_norm, color='pink', scale=35)

odes_extendido = [lambda N, P : dNdt_extendido(N, P, r, alpha, K1), lambda N, P : dPdt_extendido(N, P, beta1, q1)]
trayectorias1_kmenor = rk4_for_ode_system(h, odes_extendido, initial_conditions4, t0, 1000)
trayectorias2_kmenor = rk4_for_ode_system(h, odes_extendido, initial_conditions5, t0, 1000)
trayectorias3_kmenor = rk4_for_ode_system(h, odes_extendido, initial_conditions6, t0, 1000)
plt.plot(trayectorias1_kmenor[:, 0], trayectorias1_kmenor[:, 1], linewidth = 3)
plt.plot(trayectorias2_kmenor[:, 0], trayectorias2_kmenor[:, 1], linewidth = 3)
plt.plot(trayectorias3_kmenor[:, 0], trayectorias3_kmenor[:, 1], linewidth = 3)
plt.scatter(initial_conditions4[0], initial_conditions4[1], color='black', label='Condiciones iniciales', zorder = 2)
plt.scatter(initial_conditions5[0], initial_conditions5[1], color='black', zorder = 2)
plt.scatter(initial_conditions6[0], initial_conditions6[1], color='black', zorder = 2)

plt.xlim(0, 10)
plt.ylim(0, 10)
plt.scatter((q1 / beta1) * np.ones_like(N), (r / alpha) * (1 - (q1/beta1)/K1) * np.ones_like(N), color='green', label='Punto de intersección', zorder = 3)
plt.plot(N, case1_P_ext, color = 'red', label=r'Isocline de $P = \frac{r}{\alpha}(1 - \frac{N}{K})$')
plt.plot(case1_N_ext, N, color = 'blue', label=r'Isocline de $N = \frac{q1}{\beta1}$')
plt.scatter(K1, 0, label = 'K1 = 10', color = 'purple', zorder = 3)
plt.xlabel('N (Presas)')
plt.ylabel('P (Predadores)')
plt.title('Caso 1 Extendido')
plt.legend()
plt.xlim(0, None)
plt.ylim(0, None)
plt.text(-1, 5, r'$P = \frac{r}{\alpha}(1 - \frac{N}{K})$', fontsize=12, ha='center')
plt.text(10, -1.5, r'$N = \frac{q1}{\beta1}$' , fontsize=12, ha='center')
plt.xticks([])
plt.yticks([])
plt.grid(True)
plt.show()

#Grafico caso 2 extendido
plt.figure(figsize=(14, 10))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# Caso 1: P = r/alpha y N = q1/beta1
X, Y = np.meshgrid(np.linspace(0, 20, 10), np.linspace(0, 20, 10))
U , V = velocidad_isocline_extendido(X, Y, r, alpha, beta2, q2, K2)

# Trazar las líneas de flujo
plt.streamplot(X, Y, U, V, color='black')
for i in range(0, 20):
    for j in range(0, 20):
        U_norm = (r*i - alpha*i*j - (r*i**2)/K2) / np.sqrt((r*i - alpha*i*j - (r*i**2)/K2)**2 + (beta2*i*j - q2*j)**2)
        V_norm = (beta2*i*j - q2*j) / np.sqrt((r*i - alpha*i*j)**2 + (beta2*i*j - q2*j)**2)
        plt.quiver([i], [j], U_norm, V_norm, color='pink')


odes_extendido = [lambda N, P : dNdt_extendido(N, P, r, alpha, K2), lambda N, P : dPdt_extendido(N, P, beta2, q2)]
trayectorias1 = rk4_for_ode_system(h, odes_extendido, initial_conditions1, t0, tmax)
trayectorias2 = rk4_for_ode_system(h, odes_extendido, initial_conditions2, t0, tmax)
trayectorias3 = rk4_for_ode_system(h, odes_extendido, initial_conditions3, t0, tmax)
plt.plot(trayectorias1[:, 0], trayectorias1[:, 1], linewidth = 3)
plt.plot(trayectorias2[:, 0], trayectorias2[:, 1], linewidth = 3)
plt.plot(trayectorias3[:, 0], trayectorias3[:, 1], linewidth = 3)

plt.xlim(0, 20)
plt.ylim(0, 20)

plt.plot(N, case2_P_ext, color = 'red', label=r'Isocline de $P = \frac{r}{\alpha}(1 - \frac{N}{K})$')
plt.plot(case2_N_ext, N, color = 'blue', label=r'Isocline de $N = \frac{q2}{\beta2}$')
plt.scatter(initial_conditions1[0], initial_conditions1[1], color='black', label='Condiciones iniciales', zorder = 2)
plt.scatter(initial_conditions2[0], initial_conditions2[1], color='black', zorder = 2)
plt.scatter(initial_conditions3[0], initial_conditions3[1], color='black', zorder = 2)
plt.scatter(K2, 0, label = 'K2 = 5', color = 'purple', zorder = 3)
plt.scatter((q2 / beta2) * np.ones_like(N), (r / alpha) * (1 - (q2/beta2)/K2) * np.ones_like(N), color='green', label='Punto de intersección')
plt.xlabel('N (Presas)')
plt.ylabel('P (Predadores)')
plt.title('Caso 2 Extendido')
plt.legend()
plt.xlim(0, None)
plt.ylim(0, None)
plt.text(-1, 5, r'$P = \frac{r}{\alpha}(1 - \frac{N}{K})$', fontsize=12, ha='center')
plt.text(10, -1.5, r'$N = \frac{q2}{\beta2}$' , fontsize=12, ha='center')
plt.xticks([])
plt.yticks([])
plt.grid(True)


# Mostrar los gráficos
plt.tight_layout()

plt.show()

# Definir parámetros y condiciones iniciales

# Definir parámetros y condiciones iniciales