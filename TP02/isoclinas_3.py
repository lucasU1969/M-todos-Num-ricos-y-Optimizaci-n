import matplotlib.pyplot as plt
import numpy as np

# Define las funciones para las isoclinas

# Caso 1: P = r/alpha y N = q1/beta1
def isocline_case1(N, r, alpha, beta1, q1):
    return (r / alpha) * np.ones_like(N), (q1 / beta1) * np.ones_like(N)

def isocline_extendido(r, N, K, alpha, q1, beta1):
    return (r / alpha) * (1 - N/K) * np.ones_like(N), (q1 / beta1) * np.ones_like(N)

# # Caso 2: P = 0 y N = 0
# def isocline_case2(N):
#     return np.zeros_like(N), np.zeros_like(N)

# # Caso 3: P = r/alpha y N = 0
# def isocline_case3(N, r, alpha):
#     return (r / alpha) * np.ones_like(N), np.zeros_like(N)

# # Caso 4: P = 0 y N = q1/beta1
# def isocline_case4(N, beta1, q1):
#     return np.zeros_like(N), (q1 / beta1) * np.ones_like(N)

# Definir parámetros genéricos
r = 0.1  # Tasa de crecimiento de las presas
alpha = 0.02  # Tasa de depredación de las presas por los predadores
beta1 = 0.4  # Tasa de crecimiento de los predadores
q1 = 0.8  # Tasa de mortalidad de los predadores
K = 10  # Capacidad de carga del ecosistema
# Generar valores de N
N = np.linspace(0, 20, 200)

beta2 = 0.5
q2 = 20
K2 = 5

# Calcular las isoclinas para cada caso
case1_P, case1_N = isocline_case1(N, r, alpha, beta1, q1)
case1_P_ext, case1_N_ext = isocline_extendido(r, N, K, alpha, q1, beta1)
case2_P_ext, case2_N_ext = isocline_extendido(r, N, K2, alpha, q2, beta2)
# case2_P, case2_N = isocline_case2(N)
# case3_P, case3_N = isocline_case3(N, r, alpha)
# case4_P, case4_N = isocline_case4(N, beta1, q1)

# Graficar los cuatro casos en gráficos separados
plt.figure(figsize=(14, 10))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# Caso 1: P = r/alpha y N = q1/beta1
plt.scatter(case1_N, case1_P, color='green', label='Punto de intersección')
plt.plot(N, case1_P, color = 'red', label=r'Isocline de $P = \frac{r}{\alpha}$')
plt.plot(case1_N, N, color = 'blue', label=r'Isocline de $N = \frac{q1}{\beta1}$')
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

plt.figure(figsize=(14, 10))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# Caso 1: P = r/alpha y N = q1/beta1
plt.scatter((q1 / beta1) * np.ones_like(N), (r / alpha) * (1 - (q1/beta1)/K) * np.ones_like(N), color='green', label='Punto de intersección')
plt.plot(N, case1_P_ext, color = 'red', label=r'Isocline de $P = \frac{r}{\alpha}(1 - \frac{N}{K})$')
plt.plot(case1_N_ext, N, color = 'blue', label=r'Isocline de $N = \frac{q1}{\beta1}$')
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

plt.figure(figsize=(14, 10))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# Caso 1: P = r/alpha y N = q1/beta1
plt.scatter((q2 / beta2) * np.ones_like(N), (r / alpha) * (1 - (q2/beta2)/K2) * np.ones_like(N), color='green', label='Punto de intersección')
plt.plot(N, case2_P_ext, color = 'red', label=r'Isocline de $P = \frac{r}{\alpha}(1 - \frac{N}{K})$')
plt.plot(case2_N_ext, N, color = 'blue', label=r'Isocline de $N = \frac{q2}{\beta2}$')
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
