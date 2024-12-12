import numpy as np
import matplotlib.pyplot as plt

# Define la función de flujo (adaptada a tu ecuación diferencial)
def flow(x, y, k, r):
    dy = (y*k*np.exp(r*x))/((k-y) + y*np.exp(r*x))
    return dy

# Genera una cuadrícula de puntos en el espacio
x = np.linspace(0, 10, 50)
y = np.linspace(0, 100, 50)
X, Y = np.meshgrid(x, y)

# Parámetros de la ecuación diferencial
k = 10
r = 0.3

# Calcula las tasas de cambio (vectores de flujo) en cada punto de la cuadrícula
DY = flow(X, Y, k, r)

# Grafica el campo vectorial
plt.figure(figsize=(8, 6))
plt.quiver(Y, DY, np.zeros_like(Y), DY, color='blue', scale=20)
plt.xlabel('t')
plt.ylabel('y')
plt.title('Campo vectorial de la ecuación diferencial')
plt.grid(True)
plt.show()
