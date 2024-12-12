from ej1a import *



def main():

    coords_x = np.linspace(-4, 4, 100)
    coords_y = fa(coords_x)
    puntos = [int(k) for k in range(2,21)]

    plt.title("Puntos equiespaciados vs no equiespaciados para Lagrange de grado 7")
    plt.plot(puntos, errores_promedio_no_equiespaciados(coords_x, coords_y), label="Error promedio con puntos no equiespaciados")
    plt.plot(puntos, errores_promedio(coords_x, coords_y), label="Error promedio por puntos equiespaciados")
    plt.ylabel("Error promedio")
    plt.xlabel("Cantidad de puntos que toma el polinomio")
    plt.legend()
    plt.yscale('log')
    plt.xscale('linear')
    plt.xticks(np.arange(2, 21, step=1))
    
    plt.show()




    # para el polinomio de grado 8 dibujo el error sobre el dominio y el polinomio
    x = np.linspace(np.pi, np.pi*2, 8)
    x = np.cos(x) * 4
    y = fa(x)

    lagrangiano =sci.lagrange(x, y)
    
    plt.plot(coords_x, errores_sobre_dom(coords_x, coords_y, coords_x, lagrangiano(coords_x)))
    plt.show()



if __name__ == "__main__":
    main()
