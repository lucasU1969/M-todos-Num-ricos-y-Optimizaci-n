#ahora voy a hacer un grafico que para un N0 fijo, grafique el tamaño poblacional para distintos valores de r en el tiempo
#pero utilizando mi solución logística

import numpy as np
import matplotlib.pyplot as plt

# Datos de ejemplo
tiempo = np.linspace(0, 10, 1000)  # Tiempo
# Graficar
plt.xlabel('Tiempo')
plt.ylabel('Tamaño poblacional')
plt.title('Tamaño poblacional en el tiempo')

# Valores de r a variar
r_values = [0.1, 0.2, 0.3, 0.4, 0.5]
N0 = np.linspace(100, 1000, 10)
capacidad_maxima = np.linspace(1000, 10000, 10)
K = 10000
N = 100

def solucion_logistica(t,N0,r,K):
    return K/(1 + (K/N0 - 1)*np.exp(-r*t))

def tamaño_poblacional_con_r_k_fijos(tiempo, N0, K):
    for N in N0:
        poblacion = [solucion_logistica(t, N, 0.3, K) for t in tiempo]
        plt.plot(tiempo, poblacion, label=f'N0 = {N}')

def tamaño_poblacional_con_N0_k_fijos(tiempo, r_values, K):
    for r in r_values:
        poblacion = [solucion_logistica(t, 500, r, K) for t in tiempo]
        plt.plot(tiempo, poblacion, label=f'r = {r}')

def tamaño_poblacional_con_N0_r_fijos(tiempo, capacidad_maxima):
    for K in capacidad_maxima:
        poblacion = [solucion_logistica(t, 500, 0.3, K) for t in tiempo]
        plt.plot(tiempo, poblacion, label=f'K = {K}')

def tamaño_poblacion_con_k_fijo(tiempo, N0, r):
    for N in N0:
        for r in r_values:
            poblacion = [solucion_logistica(t, N, r, 10000) for t in tiempo]
            plt.plot(tiempo, poblacion, label=f'r = {r}, N0 = {N}')

def tamaño_poblacion_con_r_fijo(tiempo, N0, capacidad_maxima):
    for N in N0:
        for K in capacidad_maxima:
            poblacion = [solucion_logistica(t, N, 0.3, K) for t in tiempo]
            plt.plot(tiempo, poblacion, label=f'N0 = {N}, K = {K}')

tamaño_poblacional_con_r_k_fijos(tiempo, N0, K)
plt.show()
tamaño_poblacional_con_N0_r_fijos(tiempo, capacidad_maxima)
plt.show()
tamaño_poblacional_con_N0_k_fijos(tiempo, r_values, capacidad_maxima)
plt.show()
tamaño_poblacion_con_k_fijo(tiempo, N0, r_values)
plt.show()
tamaño_poblacion_con_r_fijo(tiempo, N0, capacidad_maxima)
plt.show()


# for N in N0:
#     for r in r_values:
#         # Calcular el tamaño poblacional para cada tiempo
#         poblacion = [solucion_logistica(t, N, r, K) for t in tiempo]
#         # Graficar
#         plt.plot(tiempo, poblacion, label=f'r = {r}, N0 = {N}')

# plt.show()

# for N in N0:
#     for r in r_values:
#         for K in Capacidad_maxima:
#             # Calcular el tamaño poblacional para cada tiempo
#             poblacion = [solucion_logistica(t, N, r, K) for t in tiempo]
#             # Graficar
#             plt.plot(tiempo, poblacion, label=f'r = {r}, N0 = {N}, K = {K}')

# plt.show()