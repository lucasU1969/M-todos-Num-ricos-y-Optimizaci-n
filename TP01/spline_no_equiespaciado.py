import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import funciones as f


coords_x = np.linspace(-4, 4, 100)
coords_y = f.fa(coords_x)
x_censadas = np.linspace(-2.478136535, 2.478136535, 8)
x_censadas = x_censadas + (1/10)*np.power(x_censadas, 3)
y_censadas = f.fa(x_censadas)
splines_cúbicos = spi.CubicSpline(x_censadas, y_censadas, bc_type='clamped')


plt.title("Interpolación por splines cúbicos")
plt.plot(coords_x, coords_y, label="Función original", color='k')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x_censadas, y_censadas, label="Puntos no equidistantes", color='b')
plt.plot(coords_x, splines_cúbicos(coords_x), label="Splines Cúbicos")

plt.legend()
plt.show()
