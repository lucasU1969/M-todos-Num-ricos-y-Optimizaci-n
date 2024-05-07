import matplotlib.pyplot as plt
import scipy.interpolate as sci
import numpy as np
import scipy.optimize as opt
import funciones as f
# Definir polinomios de intersección y funciones auxiliares
def t1x_pol_intersec(x:float) -> float:
    return -7.66296299*(x**3) + 17.02260561*(x**2) + 1.88588466*x + 1.62372089

def t1y_pol_intersec(x:float) -> float:
    return -0.014973*(x**3) + -0.03458137*(x**2) + 0.96540245*x + 2.91025123

def t2x_pol_intersec(x:float) -> float:
    return -0.16666667*(x**3) + 1.5*(x**2) + 3.66666667*x + 5

def t2y_pol_intersec(x:float) -> float:
    return -0.05*(x**3) + 0.65*(x**2) + -2.6*x + 5

def intersección_x(x:float) -> float:
    return t1x_pol_intersec(x) - t2x_pol_intersec(x)

def intersección_y(x:float) -> float:
    return t1y_pol_intersec(x) - t2y_pol_intersec(x)

# Carga de datos
with open("mnyo_ground_truth.csv", "r") as file:
    lines = file.readlines()
    gt_x = []
    gt_y = []
    for i in lines:
        x , y = i.split()
        gt_x.append(float(x))
        gt_y.append(float(y))

plt.plot(gt_x, gt_y, color="k", label="Trayectoria")

with open("mnyo_mediciones.csv", "r") as file:
    lines = file.readlines()
    coords_x = []
    coords_y = []
    for i in lines:
        x,y = i.split()
        coords_x.append(float(x))
        coords_y.append(float(y))

# busco las coordenadas de la segunda trayectoria
with open("mnyo_mediciones2.csv", "r") as file:
    lines = file.readlines()
    coords2_x = []
    coords2_y = []
    for i in lines:
        x,y = i.split()
        coords2_x.append(float(x))
        coords2_y.append(float(y))

t = np.lisnpace(0, 10, 100) # Lista de valores de tiempo para la interpolación
w = np.linspace(0, 4, 40)
intervalo1 = np.linspace(min(t), max(t), 100)

# Definición de funciones y polinomios
lagrangiano_x = sci.lagrange(t, coords_x)
lagrangiano_y = sci.lagrange(t, coords_y)

spline_l_x = sci.interp1d(t, coords_x)
spline_l_y = sci.interp1d(t, coords_y)

splines_x = sci.CubicSpline(t, coords_x)
splines_y = sci.CubicSpline(t, coords_y)


interv = np.linspace(0, 1, 10)

# Crear figura y subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Subplot 1: Convergencia del error absoluto en función de las iteraciones
axs[0, 0].set_title("Convergencia del error absoluto en función de las iteraciones")
axs[0, 0].set_xlabel("t")
axs[0, 0].set_ylabel("Error absoluto")
axs[0, 0].scatter(coords_x, coords_y, color='k', label="Puntos censados de la trayectoria")
axs[0, 0].plot(lagrangiano_x(intervalo1), lagrangiano_y(intervalo1), label="Polinomio de Lagrange")
axs[0, 0].plot(spline_l_x(intervalo1), spline_l_y(intervalo1), label="Splines lineales")
axs[0, 0].plot(splines_x(intervalo1), splines_y(intervalo1), label="Splines cúbicos")
axs[0, 0].legend()

# Subplot 2: Error absoluto sobre el dominio
axs[0, 1].set_title("Error absoluto sobre el dominio")
axs[0, 1].set_xlabel("Intervalo")
axs[0, 1].set_ylabel("Error absoluto")
axs[0, 1].plot(intervalo1, f.errores_sobre_intervalo(gt_x, gt_y, lagrangiano_x(intervalo1), lagrangiano_y(intervalo1)), label='Polinomio de Lagrange')
axs[0, 1].plot(intervalo1, f.errores_sobre_intervalo(gt_x, gt_y, spline_l_x(intervalo1), spline_l_y(intervalo1)), label='Splines Lineales')
axs[0, 1].plot(intervalo1, f.errores_sobre_intervalo(gt_x, gt_y, splines_x(intervalo1), splines_y(intervalo1)), label='Splines Cúbicos')
axs[0, 1].legend()

# Subplot 3: Trayectoria y puntos
axs[1, 0].set_title("Trayectoria y puntos")
axs[1, 0].set_xlabel("x")
axs[1, 0].set_ylabel("y")
axs[1, 0].plot(gt_x, gt_y, color="k", label="Trayectoria")
axs[1, 0].scatter(coords_x, coords_y, color='r', label="Trayectoria censada")
axs[1, 0].legend()

# Subplot 4: Intersección de trayectorias
axs[1, 1].set_title("Intersección de las trayectorias")
axs[1, 1].set_xlabel("x")
axs[1, 1].set_ylabel("y")
axs[1, 1].plot(splines_x(intervalo1), splines_y(intervalo1), label="T1 interpolada con splines cúbicos")
axs[1, 1].plot(t1x_pol_intersec(interv), t1y_pol_intersec(interv), label="Polinomio de la intersección (t1)")
axs[1, 1].plot(t2x_pol_intersec(interv), t2y_pol_intersec(interv), label="Polinomio de la intersección (t2)")
axs[1, 1].legend()

# Ajustar diseño y mostrar el gráfico
plt.tight_layout()
plt.show()