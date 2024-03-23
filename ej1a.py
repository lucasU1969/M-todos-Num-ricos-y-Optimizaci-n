import numpy as np
import scipy.interpolate as sci
import matplotlib.pyplot as plt

def fa(x:float) -> float:
    return (0.3**abs(x))*np.sin(4*x) - np.tanh(2*x) + 2

def norma(x0:float, y0:float, x1:float, y1:float) -> float:
    return np.sqrt((x0 - x1)**2 + (y0 -y1)**2)

def error_máximo(x0:list[float], y0:list[float], x1:list[float], y1:list[float]) -> float:
    # se supone que todas las listas tienen la misma longitud
    max = 0
    for i in range(len(x0)):
        if i==0 or max < norma(x0[i], y0[i], x1[i], y1[i]):
            max = norma(x0[i], y0[i], x1[i], y1[i])
    return max

def error_absoluto(x0:float, x1:float) -> float:
    return np.abs(x0 - x1)

def error_promedio(x0:list[float], y0:list[float], x1:list[float], y1:list[float]) -> float:
    # se supone que todas las listas tienen la misma longitud
    sum = 0
    for i in range(len(x0)):
        sum += norma(x0[i], y0[i], x1[i], y1[i])
    return sum/len(x0)

def errores_sobre_dom(x0:list[float], y0:list[float], x1:list[float], y1:list[float]) -> list[float]:
    errores = []
    for i in range(len(x0)):
        errores.append( np.sqrt((x0[i] - x1[i])**2 + (y0[i] -y1[i])**2))
    return errores


def main():

    coords_x = np.linspace(-4, 4, 100)
    coords_y = fa(coords_x)

    # calculo el mejor polinomio de lagrange interpolante
    errores_prom = []
    cant_puntos = 0
    cant_puntos_p = 0
    error_máx = 0
    error_promed = 0
    for i in range(2,21):
        data_x = np.linspace(-4, 4, i)
        data_y = fa(data_x)
        lagrangiano = sci.lagrange(data_x, data_y)
        errores_prom.append( error_promedio(coords_x, coords_y, coords_x, lagrangiano(coords_x)) )
        if i == 2:
            cant_puntos = i
            error_máx = error_máximo(coords_x, coords_y, coords_x, lagrangiano(coords_x))
            error_promed = error_promedio(coords_x, coords_y, coords_x, lagrangiano(coords_x))
        if error_promed > error_promedio(coords_x, coords_y, coords_x, lagrangiano(coords_x)):
            error_promed = error_promedio(coords_x, coords_y, coords_x, lagrangiano(coords_x))
            cant_puntos_p = i
        if error_máx > error_máximo(coords_x, coords_y, coords_x, lagrangiano(coords_x)):
            error_máx = error_máximo(coords_x, coords_y, coords_x, lagrangiano(coords_x))
            cant_puntos = i

    plt.plot(range(2,21), errores_prom)
    plt.show()

    plt.plot()

    # obs para reducir el mínimo el error máximo conviene 7 puntos pero para reducir el error promedio conviene usar 8 puntos

    # tomo los puntos para hacer el polinomio de lagrange
    dataset_x = np.linspace(-4, 4, cant_puntos_p)
    dataset_y = fa(dataset_x)

    # armo el polinomio de lagrange
    lagrangiano = sci.lagrange(dataset_x, dataset_y)

    error_absoluto_máximo = error_máximo(coords_x, coords_y, coords_x, lagrangiano(coords_x))
    print("error absoluto para", cant_puntos_p,"puntos:", error_absoluto_máximo)

    error_prom = error_promedio(coords_x, coords_y, coords_x, lagrangiano(coords_x))
    print("error promedio para", cant_puntos_p, "puntos:", error_prom)

    plt.plot(coords_x, errores_sobre_dom(coords_x, coords_y, coords_x, lagrangiano(coords_x)))
    plt.show()

    plt.plot(coords_x, coords_y)
    plt.plot(coords_x, lagrangiano(coords_x))

    plt.show()




if __name__ == "__main__":
    main()
