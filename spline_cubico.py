import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

# Definir la función a interpolar
def fa(x:float) -> float:
    return (0.3**abs(x))*np.sin(4*x) - np.tanh(2*x) + 2

coords_x = np.linspace(-4, 4, 100)
coords_y = fa(coords_x)

coords_x2 = coords_x[::10]
coords_y2 = coords_y[::10]
# Crear objeto de interpolación cúbica
interp_cubica = scipy.interpolate.CubicSpline(coords_x2, coords_y2)

# Definir puntos para la interpolación
x_interp = np.linspace(-4, 4, 100)
y_interp = interp_cubica(x_interp)

# Visualización de los datos originales y la interpolación cúbica
plt.figure(figsize=(8, 6))
plt.plot(coords_x, coords_y, '-', label='Datos originales')
plt.plot(x_interp, y_interp, '-', label='Interpolación cúbica')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Interpolación Cúbica')
plt.legend()
plt.grid(True)
plt.show()
