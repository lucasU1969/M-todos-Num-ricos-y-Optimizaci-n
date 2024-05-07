import numpy as np
import matplotlib.pyplot as plt

tiempo = np.linspace(0, 10, 1000)  # Tiempo
N0 = np.linspace(100, 1000, 10)
r_values = [0.1, 0.2, 0.3, 0.4, 0.5]

def variacion_tamaño_poblacional(t, N, r):
    return r*N*np.exp(r*t)

def tamaño_poblacional(tiempo, N0, r_values):
    for r in r_values:
        for N in N0:
            poblacion = [variacion_tamaño_poblacional(t, N, r) for t in tiempo]
            plt.plot(tiempo, poblacion, label=f'r = {r}, N0 = {N}')
    plt.xlabel('Tiempo')
    plt.ylabel('Tamaño Poblacional')
    plt.legend()
    plt.show()


tamaño_poblacional(tiempo, N0, r_values)


