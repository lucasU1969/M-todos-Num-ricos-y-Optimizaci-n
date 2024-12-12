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

# Parámetros
r = 1
alpha = 1
beta1 = 1
q1 = 1
K1 = 10
K2 = 5
beta2 = 0.5
q2 = 4

# Condiciones iniciales
N0 = np.linspace(0, 10, 10)
P0 = np.linspace(0, 10, 10)

h = 0.05
t0 = 0
tmax = 5

# Calcular las isoclinas para cada caso
case1_P, case1_N = isocline_case1(N0, r, alpha, beta1, q1)
case1_P_ext, case1_N_ext = isocline_extendido(r, N0, K1, alpha, q1, beta1)
case2_P_ext, case2_N_ext = isocline_extendido(r, N0, K2, alpha, q2, beta2)


# #Grafico caso 1 extendido
# plt.figure(figsize=(14, 10))
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.xlim(0, 10)
# plt.ylim(0, 10)
# plt.scatter((q1 / beta1) * np.ones_like(N0), (r / alpha) * (1 - (q1/beta1)/K1) * np.ones_like(N0), color='green', label='Punto de intersección', zorder = 3)
# plt.plot(N0, case1_P_ext, color = 'red', label=r'Isocline de $P = \frac{r}{\alpha}(1 - \frac{N}{K})$')
# plt.plot(case1_N_ext, N0, color = 'blue', label=r'Isocline de $N = \frac{q1}{\beta1}$')
# plt.xlabel('N (Presas)')
# plt.ylabel('P (Predadores)')
# plt.title('Caso 1 Extendido')
# plt.legend()
# plt.xlim(0, None)
# plt.ylim(0, None)
# plt.text(-1, 5, r'$P = \frac{r}{\alpha}(1 - \frac{N}{K})$', fontsize=12, ha='center')
# plt.text(5, -1, r'$N = \frac{q1}{\beta1}$' , fontsize=12, ha='center')
# plt.xticks([])
# plt.yticks([])
# plt.grid(True)
# plt.show()

# #Grafico caso 2 extendido
# plt.figure(figsize=(14, 10))
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.xlim(0, 10)
# plt.ylim(0, 10)
# plt.scatter((q2 / beta2) * np.ones_like(N0), (r / alpha) * (1 - (q2/beta2)/K2) * np.ones_like(N0), color='green', label='Punto de intersección', zorder = 3)
# plt.plot(N0, case2_P_ext, color = 'red', label=r'Isocline de $P = \frac{r}{\alpha}(1 - \frac{N}{K})$')
# plt.plot(case2_N_ext, N0, color = 'blue', label=r'Isocline de $N = \frac{q2}{\beta2}$')
# plt.xlabel('N (Presas)')
# plt.ylabel('P (Predadores)')
# plt.title('Caso 2 Extendido')
# plt.legend()
# plt.xlim(0, None)
# plt.ylim(0, None)
# plt.text(-1, 5, r'$P = \frac{r}{\alpha}(1 - \frac{N}{K})$', fontsize=12, ha='center')
# plt.text(5, -1, r'$N = \frac{q2}{\beta2}$' , fontsize=12, ha='center')
# plt.xticks([])
# plt.yticks([])
# plt.grid(True)
# plt.show()

# #Graficar caso 1
# plt.figure(figsize=(14, 10))
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.xlim(0, 10)
# plt.ylim(0, 10)
# plt.scatter((q1 / beta1) * np.ones_like(N0), (r / alpha) * np.ones_like(N0), color='green', label='Punto de intersección', zorder = 3)
# plt.plot(N0, case1_P, color = 'red', label=r'Isocline de $P = \frac{r}{\alpha}$')
# plt.plot(case1_N, N0, color = 'blue', label=r'Isocline de $N = \frac{q1}{\beta1}$')
# plt.xlabel('N (Presas)')
# plt.ylabel('P (Predadores)')
# plt.title('Caso 1')
# plt.legend()
# plt.xlim(0, None)
# plt.ylim(0, None)
# plt.text(-1, 5, r'$P = \frac{r}{\alpha}$', fontsize=12, ha='center')
# plt.text(5, -1, r'$N = \frac{q1}{\beta1}$' , fontsize=12, ha='center')
# plt.xticks([])
# plt.yticks([])
# plt.grid(True)
# plt.show()


# Define las funciones para las isoclinas
# (Aquí se mantienen igual)

# Parámetros
# (Aquí se mantienen igual)

# Condiciones iniciales
# (Aquí se mantienen igual)

# Calcular las isoclinas para cada caso
# (Aquí se mantienen igual)

# Crear subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Grafico caso 1 extendido
axs[0].scatter((q1 / beta1) * np.ones_like(N0), (r / alpha) * (1 - (q1/beta1)/K1) * np.ones_like(N0), color='green', label='Punto de intersección', zorder=3)
axs[0].plot(N0, case1_P_ext, color='red', label=r'Isocline de $P = \frac{r}{\alpha}(1 - \frac{N}{K})$')
axs[0].plot(case1_N_ext, N0, color='blue', label=r'Isocline de $N = \frac{q1}{\beta1}$')
axs[0].set_xlabel('N (Presas)')
axs[0].set_ylabel('P (Predadores)')
axs[0].set_title('Caso 1 Extendido')
axs[0].legend()
axs[0].set_xlim(0, None)
axs[0].set_ylim(0, None)
# axs[0].text(-1, 5, r'$P = \frac{r}{\alpha}(1 - \frac{N}{K})$', fontsize=12, ha='center')
# axs[0].text(5, -1, r'$N = \frac{q1}{\beta1}$', fontsize=12, ha='center')
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[0].grid(True)

# Grafico caso 2 extendido
axs[1].scatter((q2 / beta2) * np.ones_like(N0), (r / alpha) * (1 - (q2/beta2)/K2) * np.ones_like(N0), color='green', label='Punto de intersección', zorder=3)
axs[1].plot(N0, case2_P_ext, color='red', label=r'Isocline de $P = \frac{r}{\alpha}(1 - \frac{N}{K})$')
axs[1].plot(case2_N_ext, N0, color='blue', label=r'Isocline de $N = \frac{q2}{\beta2}$')
axs[1].set_xlabel('N (Presas)')
axs[1].set_ylabel('P (Predadores)')
axs[1].set_title('Caso 2 Extendido')
axs[1].legend()
axs[1].set_xlim(0, None)
axs[1].set_ylim(0, None)
# axs[1].text(-1, 5, r'$P = \frac{r}{\alpha}(1 - \frac{N}{K})$', fontsize=12, ha='center')
# axs[1].text(5, -1, r'$N = \frac{q2}{\beta2}$', fontsize=12, ha='center')
axs[1].set_xticks([])
axs[1].set_yticks([])
axs[1].grid(True)

# Graficar caso 1
axs[2].scatter((q1 / beta1) * np.ones_like(N0), (r / alpha) * np.ones_like(N0), color='green', label='Punto de intersección', zorder=3)
axs[2].plot(N0, case1_P, color='red', label=r'Isocline de $P = \frac{r}{\alpha}$')
axs[2].plot(case1_N, N0, color='blue', label=r'Isocline de $N = \frac{q1}{\beta1}$')
axs[2].set_xlabel('N (Presas)')
axs[2].set_ylabel('P (Predadores)')
axs[2].set_title('Caso 1')
axs[2].legend()
axs[2].set_xlim(0, None)
axs[2].set_ylim(0, None)
# axs[2].text(-1, 5, r'$P = \frac{r}{\alpha}$', fontsize=12, ha='center')
# axs[2].text(5, -1, r'$N = \frac{q1}{\beta1}$', fontsize=12, ha='center')
axs[2].set_xticks([])
axs[2].set_yticks([])
axs[2].grid(True)

# Ajustar el espaciado entre subplots
plt.tight_layout()

# Mostrar los subplots
plt.show()
