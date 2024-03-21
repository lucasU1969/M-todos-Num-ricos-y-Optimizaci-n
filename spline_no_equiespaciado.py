#el crierio seria tomar los puntos criticos de la funcon y luego hacer un spline con ellos
#para esto se usara la libreria scipy
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import sympy as sp

#definimos la funcion
def fa(x:float) -> float:
    return (0.3**abs(x))*np.sin(4*x) - np.tanh(2*x) + 2



def puntos_criticos(f, x): 
    
    return puntos_criticos

coords_x = np.linspace(-4, 4, 100)
coords_y = fa(coords_x)


#encontramos los puntos criticos
puntos_criticos = puntos_criticos(fa, coords_x)
coords_x2 = puntos_criticos
coords_y2 = fa(coords_x2)

#creamos el objeto de interpolacion
interp_cubica = spi.CubicSpline(coords_x2, coords_y2)

#definimos los puntos para la interpolacion
x_interp = np.linspace(-4, 4, 100)
y_interp = interp_cubica(x_interp)

#visualizamos los datos originales y la interpolacion cubica
plt.figure(figsize=(8, 6))
plt.plot(coords_x, coords_y, '-', label='Datos originales')
plt.plot(x_interp, y_interp, '-', label='Interpolación cúbica')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Interpolación Cúbica')
plt.legend()
plt.grid(True)
plt.show()
