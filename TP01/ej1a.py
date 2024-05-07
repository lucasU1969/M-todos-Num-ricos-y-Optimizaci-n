import numpy as np
import scipy.interpolate as sci
import matplotlib.pyplot as plt
import funciones as f

def main():

    coords_x = np.linspace(-4, 4, 800)
    coords_y = f.fa(coords_x)

    dataset_x = np.linspace(-4, 4, 8)
    dataset_y = f.fa(dataset_x)

    dataset1_x = np.linspace(-4, 4, 12)
    dataset1_y = f.fa(dataset1_x)

    dataset2_x = np.linspace(-4, 4, 16)
    dataset2_y = f.fa(dataset2_x)

    lagrangiano = sci.lagrange(dataset_x, dataset_y)
    lagrangiano1 = sci.lagrange(dataset1_x, dataset1_y)
    lagrangiano2 = sci.lagrange(dataset2_x, dataset2_y)

    plt.title("Ejemplo del efecto de Runge")
    plt.plot(coords_x, f.errores_sobre_intervalo(coords_x, coords_y, coords_x, lagrangiano(coords_x)), label='Error del polinomio de grado 7')
    plt.plot(coords_x, f.errores_sobre_intervalo(coords_x, coords_y, coords_x, lagrangiano1(coords_x)), label='Error del polinomio de grado 11')
    plt.plot(coords_x, f.errores_sobre_intervalo(coords_x, coords_y, coords_x, lagrangiano2(coords_x)), label='Error del polinomio de grado 15')
    plt.xlabel('x')
    plt.ylabel('Error absoluto')
    plt.yscale('symlog')

    plt.legend()
    plt.show()

    error_prom = f.error_promedio(coords_x, coords_y, coords_x, lagrangiano(coords_x))
    plt.title('Error absoluto sobre el intervalo')
    plt.xlabel('x')
    plt.ylabel('Error')
    plt.plot(coords_x, f.errores_sobre_intervalo(coords_x, coords_y, coords_x, lagrangiano(coords_x)), label='Error absoluto sobre el intervalo')
    plt.scatter(dataset_x, [0]*8, label='Puntos equiespaciados', color='k')
    plt.legend()
    plt.show()

    plt.title("Interpolación por polinomio de Lagrange")
    plt.plot(coords_x, coords_y, color='k', label='Función original')
    plt.plot(coords_x, lagrangiano(coords_x), label='Polinomio de Lagrange de grado 7')
    plt.scatter(dataset_x, dataset_y, label='Puntos equiespaciados')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    plt.title("Error promedio en función de la cantidad de puntos")
    plt.plot(range(2, 21), f.ev_error_lagrange_equiespaciado(coords_x, coords_y), label="Error promedio por puntos equiespaciados")
    plt.xlabel("Cantidad de puntos")
    plt.ylabel("Error promedio")
    plt.show()


    # comparación entre los errores en el intervalo de lagrange y splines con 8 puntos
    splines = sci.CubicSpline(dataset_x, dataset_y, bc_type='clamped')
    plt.title("Comparación de errores en el intervalo")
    plt.plot(coords_x, f.errores_sobre_intervalo(coords_x, coords_y, coords_x, lagrangiano(coords_x)), label='Error absoluto de Lagrange')
    plt.plot(coords_x, f.errores_sobre_intervalo(coords_x, coords_y, coords_x, splines(coords_x)), label='Error absoluto de Splines')
    plt.scatter(dataset_x, [0]*8, label='Puntos equiespaciados', color='k')
    plt.xlabel('x')
    plt.ylabel('Error absoluto')
    plt.legend()
    plt.show()
    

    # no equiespaciados
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

    plt.title("Error sobre el intervalo")
    plt.plot(coords_x, f.errores_sobre_intervalo(coords_x, coords_y, coords_x, lagrangiano(coords_x)))
    plt.scatter(x, [0]*8)
    plt.xlabel('x')
    plt.ylabel('Error absoluto')
    plt.show()

if __name__ == "__main__":
    main()
