import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from ej1a import *

# Definir la función a interpolar
def fa(x:float) -> float:
    return (0.3**abs(x))*np.sin(4*x) - np.tanh(2*x) + 2

coords_x = np.linspace(-4, 4, 800)
coords_y = fa(coords_x)

coords_x2 = np.linspace(-4, 4, 8)
coords_y2 = fa(coords_x2)
# Crear objeto de interpolación cúbica
interp_cubica = scipy.interpolate.CubicSpline(coords_x2, coords_y2, bc_type='clamped')

puntos = [int(k) for k in range(2,21)]

# plt.title("Splines vs Lagrange")
# plt.plot([int(k) for k in range(2,21)], ev_error_splines_equiespaciados(coords_x, coords_y), label="Error promedio con splines")
# plt.plot([int(k) for k in range(2,21)], errores_promedio(coords_x, coords_y), label="Error promedio por lagrange")
# plt.ylabel("Error promedio")
# plt.xlabel("Cantidad de puntos del dataset")
# plt.legend()
# plt.yscale('log')
# plt.xscale('linear')
# plt.xticks(np.arange(2, 21, step=1))
# plt.show()  


# plt.title("Error absoluto sobre el intervalo")
# plt.plot(coords_x, errores_sobre_dom(coords_x, coords_y, coords_x, interp_cubica(coords_x)), label='Error absoluto sobre el intervalo')
# plt.scatter(coords_x2, [0]*8, color='k', label='Puntos equiespaciados')
# plt.xlabel('x')
# plt.ylabel('Error')
# plt.legend()
# plt.show()  

# Definir puntos para la interpolación
x_interp = np.linspace(-4, 4, 100)
y_interp = interp_cubica(x_interp)




# Visualización de los datos originales y la interpolación cúbica
plt.plot(coords_x, coords_y, '-', label='Función original', color='k')
plt.plot(x_interp, y_interp, '-', label='Interpolación cúbica (7)')
plt.scatter(coords_x2, coords_y2, label="Puntos censados")
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Interpolación por splines cúbicos')
plt.legend()
plt.show()
