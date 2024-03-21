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


# esto da la función fa
inicio = -4  # Valor inicial del intervalo
fin = 4    # Valor final del intervalo
numero_elementos = 1000  # cantidad de divisiones del intervalo
coords_x = np.linspace(inicio, fin, numero_elementos)
coords_y = fa(coords_x)

for i in range(2,11):
    c_x3 = np.linspace( inicio, fin, i)
    c_y3 = fa(c_x3)
    pol_lagrange = scipy.interpolate.lagrange( c_x3, c_y3)
    print(i, error_absoluto_máximo(fa, pol_lagrange, coords_x))
    if i==7:
        plt.figure(figsize=(8, 6))
        plt.plot(coords_x, coords_y)
        plt.plot(coords_x, pol_lagrange(coords_x))

plt.show()



