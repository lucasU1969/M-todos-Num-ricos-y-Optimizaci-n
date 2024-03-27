import matplotlib.pyplot as plt
import scipy.interpolate as sci
import numpy as np
from ej1a import norma
import scipy.optimize as opt

def max_distance(x1:list[float], y1:list[float], x2:list[float], y2:list[float]) -> float:
    # las listas recibidas tienen la misma longitud
    max_dis = 0
    for i in range(len(x1)):
        if i == 0 or max_dis < np.sqrt((x1[i] - x2[i])**2 + (y1[i] - y2[i])**2):
            max_dis = np.sqrt((x1[i] - x2[i])**2 + (y1[i] - y2[i])**2)
    return max_dis

def distancia_promedio(x1:list[float], y1:list[float], x2:list[float], y2:list[float]) -> float:
    distancias = 0
    for i in range(len(x1)):
        distancias += norma(x1[i], y1[i], x2[i], y2[i])
    return distancias/len(x1)

def t1x_pol_intersec(x:float) -> float:
    # polinomio de la x de la primera trayectoria
    return -7.66296299*(x**3) + 17.02260561*(x**2) + 1.88588466*x + 1.62372089

def t1y_pol_intersec(x:float) -> float:
    # polinomio de la y de la primera trayectoria
    return -0.014973*(x**3) + -0.03458137*(x**2) + 0.96540245*x + 2.91025123

def t2x_pol_intersec(x:float) -> float:
    # polinmio de la x de la segunda trayectoria
    return -0.16666667*(x**3) + 1.5*(x**2) + 3.66666667*x + 5

def t2y_pol_intersec(x:float) -> float:
    # polinomio de la y de la segunda trayectoria
    return -0.05*(x**3) + 0.65*(x**2) + -2.6*x + 5

def intersección_x(x:float) -> float:
    return t1x_pol_intersec(x) - t2x_pol_intersec(x)

def intersección_y(x:float) -> float:
    return t1y_pol_intersec(x) - t2y_pol_intersec(x)

# gráfico del ground truth
with open("mnyo_ground_truth.csv", "r") as file:
    lines = file.readlines()
    gt_x = []
    gt_y = []
    for i in lines:
        x , y = i.split()
        gt_x.append(float(x))
        gt_y.append(float(y))

plt.plot(gt_x, gt_y, color="k", label="trayectoria")

# obtener las coordenadas x e y de la trayectoria.
# estos son los únicos puntos de la trayectoria que conozco.
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

# hay que interpolar ambas coords de la trayectoria
t = range(0, 10)
intervalo1 = np.linspace(0,9, 100)

# acá esta hecho el lagrangiano todavía
lagrangiano_x = sci.lagrange(t, coords_x)
lagrangiano_y = sci.lagrange(t, coords_y)
# plt.plot(lagrangiano_x(intervalo), lagrangiano_y(intervalo), label="polinomio de lagrange")
print("error máximo con polinomio de lagrange:", max_distance(gt_x, gt_x, lagrangiano_x(intervalo1), lagrangiano_y(intervalo1)))
print("error promedio con polinomio de lagrange:", distancia_promedio(gt_x, gt_x, lagrangiano_x(intervalo1), lagrangiano_y(intervalo1)))


# acá lo voy a hacer con splines lineales
spline_l_x = sci.interp1d(t, coords_x)
spline_l_y = sci.interp1d(t, coords_y) 
# plt.plot(spline_l_x(intervalo1), spline_l_y(intervalo1), label="splines lineales")
print("error máximo con splines lineales:", max_distance(gt_x, gt_y, spline_l_x(intervalo1), spline_l_y(intervalo1)))
print("error promedio con splines lineales:", distancia_promedio(gt_x, gt_y, spline_l_x(intervalo1), spline_l_y(intervalo1)))

# hago splines cúbicos
splines_x = sci.CubicSpline(t, coords_x)
splines_y = sci.CubicSpline(t, coords_y)
plt.plot(splines_x(intervalo1), splines_y(intervalo1),label="splines cúbicos")
print("error máximo con splines cúbicos:", max_distance(gt_x, gt_y, splines_x(intervalo1), splines_y(intervalo1)))
print("error promedio con splines cúbicos:", distancia_promedio(gt_x, gt_y, splines_x(intervalo1), splines_y(intervalo1)))


# interpolo la segunda trayectoria usando splines
w = range(0, 4) 
intervalo2 = np.linspace(0, 3, 100)

splines2_x = sci.CubicSpline(w, coords2_x)
splines2_y =sci.CubicSpline(w, coords2_y)
plt.plot(splines2_x(intervalo2), splines2_y(intervalo2), label="trayectoria2") 


# splines que me dan la intersección
interv = np.linspace(0, 1, 10)
plt.plot(t1x_pol_intersec(interv), t1y_pol_intersec(interv), label="spline1_intersección")
plt.plot(t2x_pol_intersec(interv), t2y_pol_intersec(interv), label="spline2_intersección")

# la coordenada x de la primera trayectoria tiene que ser igual a la de la segunda
# la coordenada y de la primera trayectoria tiene que ser igual a la de la segunda
# utilizo el método de newton para calcular la raíz de las intersecciones.
root_x = t1x_pol_intersec(opt.newton(intersección_x, 0.5))
root_y = t1y_pol_intersec(opt.newton(intersección_y, 0.5))



print(f"intersección: ({root_x}, {root_y})")

plt.scatter(root_x, root_y, label="intersección")

plt.legend()
plt.show()


