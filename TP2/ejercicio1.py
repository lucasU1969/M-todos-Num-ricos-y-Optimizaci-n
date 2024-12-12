import numpy as np

import matplotlib.pyplot as plt

# Datos de ejemplo
tiempo = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Tiempo
# Graficar
plt.xlabel('Tiempo')
plt.ylabel('Tamaño poblacional')
plt.title('Tamaño poblacional en el tiempo')
# Valores de N0 a variar
N0_values = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# Graficar para cada valor de N0
for N0 in N0_values:
    # Calcular el tamaño poblacional para cada tiempo

    poblacion = [N0 * np.exp(0.2 * t) for t in tiempo]

    # Graficar
    plt.plot(tiempo, poblacion, label=f'N0 = {N0}')

# Configurar la leyenda
plt.legend()

# Mostrar el gráfico
plt.show()

# Valores de r a variar
r_values = [0.1, 0.2, 0.3, 0.4, 0.5]

# Graficar para cada valor de r y N0
for r in r_values:
    # Valores de N0 a variar
    N0_values = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    # Graficar para cada valor de N0
    for N0 in N0_values:
        # Calcular el tamaño poblacional para cada tiempo
        poblacion = [N0 * np.exp(r * t) for t in tiempo]

        # Graficar
        plt.plot(tiempo, poblacion, label=f'r = {r}, N0 = {N0}')

# Configurar la leyenda
plt.legend()

# Mostrar el gráfico
plt.show()