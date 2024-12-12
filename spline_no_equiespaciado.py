import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import scipy.interpolate as spi
<<<<<<< HEAD
=======
import ej1a

# quiero graficar la evolución del error de los splines equiespaciados y no equiespaciados.
>>>>>>> origin/Ej01_a


def fa(x:float) -> float:
    return (0.3**(np.abs(x)))*np.sin(4*x) - np.tanh(2*x) + 2

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


<<<<<<< HEAD
def puntos_criticos(f, x): 
    # usar bisección o algún otro método de búsqueda de raíces para encontrar raíces de la derivada. 
    # si no usar bisección y aproximar el error  

    return

=======
>>>>>>> origin/Ej01_a
coords_x = np.linspace(-4, 4, 100)
coords_y = fa(coords_x)

plt.title("Interpolación por splines cúbicos")
plt.plot(coords_x, coords_y, label="Función original", color='k')

<<<<<<< HEAD
c_x3 = calc_puntos_criticos(fa_prima, coords_x)
c_x3.append(4)
c_x3.insert(0, -4)
c_x3 = np.array(c_x3)
c_y3 = fa(c_x3)
inter_cubica = scipy.interpolate.CubicSpline( c_x3, c_y3)
print(error_absoluto_máximo(fa, inter_cubica, coords_x))

plt.plot(coords_x, coords_y)
plt.plot(coords_x, inter_cubica(coords_x))
=======

errores_equi = ej1a.errores_relativos_equiespaciados(coords_x, coords_y)


x_censadas = np.linspace(-2.478136535, 2.478136535, 8)
x_censadas = x_censadas + (1/10)*np.power(x_censadas, 3)
y_censadas = fa(x_censadas)

plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x_censadas, y_censadas, label="Puntos no equidistantes", color='b')
splines_cúbicos = spi.CubicSpline(x_censadas, y_censadas, bc_type='clamped')
plt.plot(coords_x, splines_cúbicos(coords_x), label="Splines Cúbicos")

plt.legend()
>>>>>>> origin/Ej01_a
plt.show()
# pc_mayores_a_0 = calc_puntos_criticos(fa_prima, )
print(calc_puntos_criticos(fa_prima, coords_x))