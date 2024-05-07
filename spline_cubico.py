import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from ej1a import *

def fa(x:float) -> float:
    return (0.3**abs(x))*np.sin(4*x) - np.tanh(2*x) + 2

<<<<<<< HEAD
def fb(x: float) -> float:
    return 1

def error_absoluto(f1, f2, intervalo:list) -> list:
    return f1(intervalo) - f2(intervalo)

def error_absoluto_máximo( f1, f2, intervalo:list) -> float:
    return max(error_absoluto(f1,f2, intervalo))


# esto da la función fa
inicio = -4  # Valor inicial del intervalo
fin = 4    # Valor final del intervalo
numero_elementos = 1000  # cantidad de divisiones del intervalo
coords_x = np.linspace(inicio, fin, numero_elementos)
coords_y = fa(coords_x)

error_c_puntos = []
for i in range(2,21):
    c_x3 = np.linspace( inicio, fin, i)
    c_y3 = fa(c_x3)
    inter_cubica = scipy.interpolate.CubicSpline( c_x3, c_y3)
    print(i, error_absoluto_máximo(fa, inter_cubica, coords_x))
    error_c_puntos.append(error_absoluto_máximo(fa, inter_cubica, coords_x))
    if i==10:
        plt.figure(figsize=(8, 6))
        plt.plot(coords_x, coords_y)
        plt.plot(coords_x, inter_cubica(coords_x))
        plt.show()
=======
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
>>>>>>> origin/Ej01_a


<<<<<<< HEAD
plt.plot(range(2,21), error_c_puntos)
plt.show()
=======



# Visualización de los datos originales y la interpolación cúbica
plt.plot(coords_x, coords_y, '-', label='Función original', color='k')
plt.plot(x_interp, y_interp, '-', label='Interpolación cúbica (7)')
plt.scatter(coords_x2, coords_y2, label="Puntos censados")
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Interpolación por splines cúbicos')
plt.legend()
plt.show()
>>>>>>> origin/Ej01_a
