import numpy as np
import scipy.interpolate as sci
import matplotlib.pyplot as plt
from ej1a import fa, errores_sobre_dom, error_promedio, error_máximo

def main():

    coords_x = np.linspace(-4, 4, 100)
    coords_y = fa(coords_x)

    errores_prom = []
    for i in range(2, 21):
        x3 = np.linspace(np.pi, np.pi*2, i)
        x3 = np.cos(x3) * 4
        y3 = fa(x3)
        pol_lagrange = sci.lagrange(x3, y3)
        
        errores_prom.append(error_promedio(coords_x, coords_y, coords_x, pol_lagrange(coords_x)))


    plt.plot(range(2, 21), errores_prom)
    plt.show()


    # para el polinomio de grado 8 dibujo el error sobre el dominio y el polinomio
    x = np.linspace(np.pi, np.pi*2, 8)
    x = np.cos(x) * 4
    y = fa(x)

    lagrangiano =sci.lagrange(x, y)
    
    print("error absoluto para", 8,"puntos:", error_máximo(coords_x, coords_y, coords_x, lagrangiano(coords_x)))

    print("error promedio para", 8, "puntos:", error_promedio(coords_x, coords_y, coords_x, lagrangiano(coords_x)))
    
    plt.plot(coords_x, errores_sobre_dom(coords_x, coords_y, coords_x, lagrangiano(coords_x)))
    plt.show()

    plt.plot(coords_x, coords_y)
    plt.plot(coords_x, lagrangiano(coords_x))
    plt.show()



if __name__ == "__main__":
    main()
