import numpy as np
import scipy.interpolate as sci
import matplotlib.pyplot as plt
import funciones as f


def main():

    coords_x = np.linspace(-4, 4, 800)
    coords_y = f.fa(coords_x)


    x = np.linspace(np.pi, np.pi*2, 8)
    x = np.cos(x) * 4
    y = f.fa(x)

    lagrangiano =sci.lagrange(x, y)
    puntos =np.array([int(k) for k in range(2,21)])

    
    plt.plot(puntos, f.ev_error_lagrange_no_equiespaciado(coords_x, coords_y), label="Error promedio con puntos no equiespaciados")
    plt.title("Error promedio en función de la cantidad de puntos")
    plt.xlabel("Cantidad de puntos")
    plt.ylabel("Error promedio")
    plt.show()

    
    plt.plot(coords_x, f.errores_sobre_intervalo(coords_x, coords_y, coords_x, lagrangiano(coords_x)), label='Error absoluto sobre el intervalo')
    plt.scatter(x, [0]*8, label='Puntos no equiespaciados')
    plt.xlabel('x')
    plt.ylabel('Error absoluto')
    plt.legend()
    plt.show()

    plt.title("Interpolación por polinomio de Lagrange no equiespaciado")
    plt.plot(coords_x, coords_y, color='k', label='Función original')
    plt.plot(coords_x, lagrangiano(coords_x), label='Polinomio de Lagrange de grado 7')
    plt.scatter(x, y, label='Puntos no equiespaciados')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    plt.title("Puntos equiespaciados vs no equiespaciados para Lagrange de grado 7")
    plt.plot(puntos, f.ev_error_lagrange_no_equiespaciado(coords_x, coords_y), label="Error promedio con puntos no equiespaciados")
    plt.plot(puntos, f.ev_error_lagrange_equiespaciado(coords_x, coords_y), label="Error promedio por puntos equiespaciados")
    plt.ylabel("Error promedio")
    plt.xlabel("Cantidad de puntos que toma el polinomio")
    plt.legend()
    plt.yscale('log')
    plt.xscale('linear')
    plt.xticks(np.arange(2, 21, step=1))
    plt.show()




if __name__ == "__main__":
    main()
