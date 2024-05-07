import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from ej1a import *

def main():
    coords_x = np.linspace(-4, 4, 800)
    coords_y = f.fa(coords_x)

    xcensadas = np.linspace(-4, 4, 8)
    ycensadas = f.fa(xcensadas)

    splines_cúbicos = scipy.interpolate.CubicSpline(xcensadas, ycensadas, bc_type='clamped')

    plt.title("Splines vs Lagrange")
    plt.plot([int(k) for k in range(2,21)], f.ev_error_splines_equiespaciados(coords_x, coords_y), label="Error promedio con splines")
    plt.plot([int(k) for k in range(2,21)], f.ev_error_lagrange_equiespaciado(coords_x, coords_y), label="Error promedio por lagrange")
    plt.ylabel("Error promedio")
    plt.xlabel("Cantidad de puntos del dataset")
    plt.legend()
    plt.yscale('log')
    plt.xscale('linear')
    plt.xticks(np.arange(2, 21, step=1))
    plt.show()  


    plt.title("Error absoluto sobre el intervalo")
    plt.plot(coords_x, f.errores_sobre_intervalo(coords_x, coords_y, coords_x, splines_cúbicos(coords_x)), label='Error absoluto sobre el intervalo')
    plt.scatter(xcensadas, [0]*8, color='k', label='Puntos equiespaciados')
    plt.xlabel('x')
    plt.ylabel('Error')
    plt.legend()
    plt.show()  

    plt.plot(coords_x, coords_y, '-', label='Función original', color='k')
    plt.plot(coords_x, splines_cúbicos(coords_x), '-', label='Interpolación cúbica (7)')
    plt.scatter(xcensadas, ycensadas, label="Puntos censados")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Interpolación por splines cúbicos')
    plt.legend()
    plt.show()

    x_censadas = np.linspace(-2.478136535, 2.478136535, 8)
    x_censadas = x_censadas + (1/10)*np.power(x_censadas, 3)
    y_censadas = f.fa(x_censadas)
    splines_cúbicos = sci.CubicSpline(x_censadas, y_censadas, bc_type='clamped')

    plt.title("Interpolación por splines cúbicos")
    plt.plot(coords_x, coords_y, label="Función original", color='k')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x_censadas, y_censadas, label="Puntos no equidistantes", color='b')
    plt.plot(coords_x, splines_cúbicos(coords_x), label="Splines Cúbicos")

    plt.legend()
    plt.show()

    plt.title("splines equiespaciados vs no equiespaciados")
    plt.plot([int(k) for k in range(2,21)], f.ev_error_splines_equiespaciados(coords_x, coords_y), label="Error promedio con splines equiespaciados")
    plt.plot([int(k) for k in range(2,21)], f.ev_error_splines_no_equiespaciados(coords_x, coords_y), label="Error promedio con splines no equiespaciados")
    plt.ylabel("Error promedio")
    plt.xlabel("Cantidad de puntos del dataset")
    plt.legend()
    plt.yscale('log')
    plt.xscale('linear')
    plt.xticks(np.arange(2, 21, step=1))
    plt.show()
    


if __name__ == "__main__":
    main()