import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def fb(x1: float, x2: float) -> float:
    return (0.75*np.e)**((-(10*x1 - 2)**2)/4 - ((9*x2 - 2)**2)/4) \
        + (0.65*np.e)**(- ((9*x1 + 1)**2)/9 - ((10*x2 + 1)**2)/2) \
        + (0.55*np.e)**(- ((9*x1 - 6)**2)/4 - ((9*x2 - 3)**2)/4) \
        - (0.01*np.e)**(- ((9*x1 - 7)**2)/4 - ((9*x2 - 3)**2)/4)

# Generar puntos para la cuadrícula 3D
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)
Z = fb(X, Y)

# Crear la figura y el eje 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Graficar la superficie
ax.plot_surface(X, Y, Z, cmap='viridis')

# Mostrar el gráfico
plt.show()
