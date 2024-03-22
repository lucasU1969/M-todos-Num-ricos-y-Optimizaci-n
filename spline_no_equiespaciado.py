import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

def fa(x:float) -> float:
    return (0.3**abs(x))*np.sin(4*x) - np.tanh(2*x) + 2

def fa_prima(x) -> float:
    if(x <= 0):
        return (-0.3**(-x))*np.log(0.3)*np.sin(4*x) + 0.3**(-x)*4*np.cos(4*x) + 2/(np.cosh(2*x)*np.cosh(2*x))
    return (-0.3**(x))*np.log(0.3)*np.sin(4*x) + (0.3**(x))*4*np.cos(4*x) + 2/(np.cosh(2*x)*np.cosh(2*x))

def error_absoluto(f1, f2, intervalo:list) -> list:
    return f1(intervalo) - f2(intervalo)

def error_absoluto_máximo( f1, f2, intervalo:list) -> float:
    return max(error_absoluto(f1,f2, intervalo))

def calc_puntos_criticos(f, x:list[float]) -> list[float]:
    pc = []
    for i in range(len(x) -1):
        if ((fa_prima(x[i])*fa_prima(x[i+1])) < 0):
            pc.append(x[i])
    return pc


inicio = -4  # Valor inicial del intervalo
fin = 4    # Valor final del intervalo
numero_elementos = 10000  # cantidad de divisiones del intervalo
coords_x = np.linspace(inicio, fin, numero_elementos)
coords_y = fa(coords_x)


c_x3 = calc_puntos_criticos(fa_prima, coords_x)
c_x3.append(4)
c_x3.insert(0, -4)
c_x3 = np.array(c_x3)
c_y3 = fa(c_x3)
inter_cubica = scipy.interpolate.CubicSpline( c_x3, c_y3)
print(error_absoluto_máximo(fa, inter_cubica, coords_x))

plt.plot(coords_x, coords_y)
plt.plot(coords_x, inter_cubica(coords_x))
plt.show()
# pc_mayores_a_0 = calc_puntos_criticos(fa_prima, )
print(calc_puntos_criticos(fa_prima, coords_x))