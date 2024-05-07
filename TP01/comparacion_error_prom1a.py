import funciones as f
import numpy as np
import scipy.interpolate as sci
import matplotlib.pyplot as plt


def main():

    coords_x = np.linspace(-4, 4, 100)
    coords_y = f.fa(coords_x)
    puntos = [int(k) for k in range(2,21)]


    plt.title("Puntos equiespaciados vs no equiespaciados para Lagrange de grado 7")
    plt.plot([int(k) for k in range(2,21)], f.ev_error_lagrange_no_equiespaciado, label="Error promedio con puntos no equiespaciados")
    plt.plot([int(k) for k in range(2,21)], f.ev_error_lagrange_equiespaciado(coords_x, coords_y), label="Error promedio por puntos equiespaciados")
    plt.ylabel("Error promedio")
    plt.xlabel("Cantidad de puntos que toma el polinomio")
    plt.legend()
    plt.yscale('log')
    plt.xscale('linear')
    plt.xticks(np.arange(2, 21, step=1))
    
    plt.show()

    x = np.linspace(np.pi, np.pi*2, 8)
    x = np.cos(x) * 4
    y = f.fa(x)

    lagrangiano =sci.lagrange(x, y)
    
    plt.title("Error sobre el intervalo")
    plt.plot(coords_x, f.errores_sobre_intervalo(coords_x, coords_y, coords_x, lagrangiano(coords_x)))
    plt.scatter(x, [0]*8)
    plt.xlabel('x')
    plt.ylabel('Error absoluto')
    plt.show()



if __name__ == "__main__":
    main()
