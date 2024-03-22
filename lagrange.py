import scipy.interpolate 
import numpy as np
import matplotlib.pyplot as plt


def fa(x:float) -> float:
    return (0.3**abs(x))*np.sin(4*x) - np.tanh(2*x) + 2

def fb(x: float) -> float:
    return 1

def error_absoluto(f1, f2, intervalo:list) -> list:
    return f1(intervalo) - f2(intervalo)

def error_absoluto_máximo( f1, f2, intervalo:list) -> float:
    return max(error_absoluto(f1,f2, intervalo))

def error_promedio(f1, f2, intervalo) -> float:
    return sum(error_absoluto(f1,f2, intervalo))/len(intervalo)


def graficar_aproximación(f, aprox, intervalo):
    plt.plot(intervalo, f(intervalo))
    plt.plot(intervalo, aprox(intervalo))
    plt.show()

def calcular_grado_min_error(f, intervalo, inicio, fin):
    min = 0
    min_error = 0
    for i in range(2, 21):
        cx = np.linspace(inicio, fin, i)
        cy = f(cx)
        pol_lagrange = scipy.interpolate.lagrange( cx, cy)
        if i == 2 or min_error > error_absoluto_máximo(f, pol_lagrange, intervalo):
            min = i
            min_error = error_absoluto_máximo(f, pol_lagrange, intervalo)
    return min

def graf_error_según_n_puntos(f, intervalo, inicio, fin):
    error = []
    for i in range(2, 21):
        cx = np.linspace(inicio, fin, i)
        cy = f(cx)
        pol_lagrange = scipy.interpolate.lagrange( cx, cy)
        error.append(error_absoluto_máximo(f, pol_lagrange, intervalo))
    plt.plot(range(2, 21), error)
    plt.xlabel("Cantidad de puntos tomados por el polinomio")
    plt.ylabel("Error absoluto respecto a la función")

def graf_error_prom_n_puntos(f, intervalo, inicio, fin):
    error = []
    for i in range(2, 21):
        cx = np.linspace(inicio, fin, i)
        cy = f(cx)
        pol_lagrange = scipy.interpolate.lagrange( cx, cy)
        error.append(error_promedio(f, pol_lagrange, intervalo))
        print(error_promedio(f, pol_lagrange, intervalo))
    plt.plot(range(2, 21), error)
    plt.xlabel("Cantidad de puntos tomados por el polinomio")
    plt.ylabel("Error promedio respecto a la función")
# esto da la función fa
inicio = -4  # Valor inicial del intervalo
fin = 4    # Valor final del intervalo
numero_elementos = 1000  # cantidad de divisiones del intervalo
coords_x = np.linspace(inicio, fin, numero_elementos)
coords_y = fa(coords_x)

# print(calcular_grado_min_error(fa, coords_x, inicio, fin))
graf_error_según_n_puntos(fa, coords_x, inicio, fin)
# plt.show()

graf_error_prom_n_puntos(fa, coords_x, inicio, fin)
plt.show()


abs_error = []
for i in range(2,31):
    c_x3 = np.linspace( inicio, fin, i)
    c_y3 = fa(c_x3)
    pol_lagrange = scipy.interpolate.lagrange( c_x3, c_y3)
    error_i = error_absoluto_máximo(fa, pol_lagrange, coords_x)
    if i == 7: print(error_i)
    abs_error.append(error_i)
    # if i==7:
    #     plt.plot(coords_x, error_absoluto(fa, pol_lagrange, coords_x))
    #     plt.plot(coords_x, coords_y)
    #     plt.plot(coords_x, pol_lagrange(coords_x))
    #     plt.axhline(y=0, color='b', linestyle='-')
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(range(2,31), abs_error)
# idea: quiero calcular cuanto le estoy errando al error real por calcularlo como lo hago.
# derivo y veo el máximo valor de la derivada. 
# continúo la recta tangente en ese punto 

# plt.show()
# plt.title("Evolución del error según el grado del polinomio de Lagrange")
# plt.xlabel("Grado del polinomio")
# plt.ylabel("Máximo error")
# plt.plot(range(2,21), abs_error)
# plt.show()


# idea para calcular puntos no equiespaciados: nodos de chebyshev

# primero voy a usar la proyección del la circunferencia de radio 4 sobre el eje x (cos)

error_c2 = []
for i in range(2, 31):
    x3 = np.linspace(np.pi, np.pi*2, i)
    x3 = np.cos(x3) * 4
    y3 = fa(x3)

    pol_lagrange = scipy.interpolate.lagrange( x3, y3)
    # print(error_absoluto_máximo(fa, pol_lagrange, coords_x))
    error_c2.append(error_absoluto_máximo(fa, pol_lagrange, coords_x))
    if i == 10:
        plt.plot()

plt.subplot(1, 2, 2)
plt.plot(range(2,31), error_c2)
plt.show()



# plt.plot(coords_x, error_absoluto(fa, pol_lagrange, coords_x))
# plt.plot(coords_x, coords_y)
# plt.plot(coords_x, pol_lagrange(coords_x))
# plt.axhline(y=0, color='b', linestyle='-')

# plt.show()