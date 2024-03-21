from scipy.interpolate import lagrange
import numpy as np
import matplotlib.pyplot as plt

inicio = -4  # Valor inicial del intervalo
fin = 4    # Valor final del intervalo
numero_elementos = 100  # Número de elementos equiespaciados

# crea la lista de elementos equiespaciados
def fa(x:float) -> float:
    return (0.3**abs(x))*np.sin(4*x) - np.tanh(2*x) + 2

def fb(x: float) -> float:
    return 1

def error_absoluto(f1, f2, intervalo:list) -> list:
    return f1(intervalo) - f2(intervalo)

def error_absoluto_máximo( f1, f2, intervalo:list) -> float:
    return max(error_absoluto(f1,f2, intervalo))

def lagrangiano( x:float, coords_x:list[float], coords_y:list[float]) -> float:
    # tiene que evaluar el polinomio de lagrange en el punto x.
    # suma de todas las bases multiplicadas por f(xk)

    return

def bases_del_lagrangiano( x:float, coords_x:list[float]) -> float:
    numerador = 1
    # for i in coords_x:
    #     for j in coords_x:
            
    denominador = 1
    return numerador/denominador

coords_x = np.linspace(inicio, fin, numero_elementos)
coords_y = fa(coords_x)

# print(coords_y[::2])

# coords_x2 = coords_x[::2]
# coords_y2 = coords_y[::2]

# pol_lagrange = lagrange( coords_x2, coords_y2)

# print( error_absoluto_máximo(fa, pol_lagrange, coords_x))

plt.plot(coords_x, coords_y)
# plt.plot(coords_x2, coords_y2)

plt.show()

