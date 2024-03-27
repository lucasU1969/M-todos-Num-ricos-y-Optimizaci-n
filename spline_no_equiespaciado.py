import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import ej1a

# quiero graficar la evolución del error de los splines equiespaciados y no equiespaciados.

#definimos la funcion
def fa(x:float) -> float:
    return (0.3**(np.abs(x)))*np.sin(4*x) - np.tanh(2*x) + 2



coords_x = np.linspace(-4, 4, 100)
coords_y = fa(coords_x)

plt.title("Interpolación por splines cúbicos")
plt.plot(coords_x, coords_y, label="Función original", color='k')


errores_equi = ej1a.errores_relativos_equiespaciados(coords_x, coords_y)


x_censadas = np.linspace(-2.478136535, 2.478136535, 8)
x_censadas = x_censadas + (1/10)*np.power(x_censadas, 3)
y_censadas = fa(x_censadas)

plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x_censadas, y_censadas, label="Puntos no equidistantes", color='b')
splines_cúbicos = spi.CubicSpline(x_censadas, y_censadas, bc_type='clamped')
plt.plot(coords_x, splines_cúbicos(coords_x), label="Splines Cúbicos")

plt.legend()
plt.show()
