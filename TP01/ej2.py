import matplotlib.pyplot as plt
import scipy.interpolate as sci
import numpy as np
import funciones as f

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

def t1x_derivada(x:float):
    return -22.988888*(x**2) + 34.04521122*x + 1.88588466

def t1y_derivada(x:float):
    return -0.044919*(x**2) + -0.06916274*x + 0.96540245

def t2x_derivada(x:float):
    return -0.5*(x**2) + 3*x + 3.66666667

def t2y_derivada(x:float):
    return -0.15*(x**2) + 1.3*x - 2.6

def coord_x_intersección_derivada(x:float):
    return t1x_derivada(x) - t2x_derivada(x)

def coord_y_intersección_derivada(x:float):
    return t1y_derivada(x) - t2y_derivada(x)

# gráfico del ground truth
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

# hay que interpolar ambas coords de la trayectoria
t = range(0, 10)
intervalo1 = np.linspace(0,9, 100)


plt.title("Convergencia del error absoluto en función de las iteraciones")
plt.xlabel("t")
plt.ylabel("Error absoluto")
lagrangiano_x = sci.lagrange(t, coords_x)
lagrangiano_y = sci.lagrange(t, coords_y)
plt.scatter(coords_x, coords_y,color='k', label="Puntos censados de la trayectoria")
plt.plot(lagrangiano_x(intervalo1), lagrangiano_y(intervalo1), label="Polinomio de lagrange")
print("error máximo con polinomio de lagrange:", f.error_máximo(gt_x, gt_x, lagrangiano_x(intervalo1), lagrangiano_y(intervalo1)))
print("error promedio con polinomio de lagrange:", f.error_promedio(gt_x, gt_x, lagrangiano_x(intervalo1), lagrangiano_y(intervalo1)))



spline_l_x = sci.interp1d(t, coords_x)
spline_l_y = sci.interp1d(t, coords_y) 
plt.plot(spline_l_x(intervalo1), spline_l_y(intervalo1), label="Splines lineales")
print("error máximo con splines lineales:", f.error_máximo(gt_x, gt_y, spline_l_x(intervalo1), spline_l_y(intervalo1)))
print("error promedio con splines lineales:", f.error_promedio(gt_x, gt_y, spline_l_x(intervalo1), spline_l_y(intervalo1)))


splines_x = sci.CubicSpline(t, coords_x)
splines_y = sci.CubicSpline(t, coords_y)
plt.plot(splines_x(intervalo1), splines_y(intervalo1),label="T1 interpolada con splines cúbicos")
print("error máximo con splines cúbicos:", f.error_máximo(gt_x, gt_y, splines_x(intervalo1), splines_y(intervalo1)))
print("error promedio con splines cúbicos:", f.error_promedio(gt_x, gt_y, splines_x(intervalo1), splines_y(intervalo1)))

plt.show()

plt.title("Error absoluto sobre el dominio")
plt.plot(intervalo1, f.errores_sobre_intervalo(gt_x, gt_y, lagrangiano_x(intervalo1), lagrangiano_y(intervalo1)), label='Polinomio de Lagrange')
plt.plot(intervalo1, f.errores_sobre_intervalo(gt_x, gt_y, spline_l_x(intervalo1), spline_l_y(intervalo1)), label='Splines Lineales')
plt.plot(intervalo1, f.errores_sobre_intervalo(gt_x, gt_y, splines_x(intervalo1), splines_y(intervalo1)), label='Splines Cúbicos')
plt.show()


# interpolo la segunda trayectoria usando splines
w = range(0, 4) 
intervalo2 = np.linspace(0, 3, 100)

splines2_x = sci.CubicSpline(w, coords2_x)
splines2_y =sci.CubicSpline(w, coords2_y)
plt.plot(splines2_x(intervalo2), splines2_y(intervalo2), label="T2 interpolada con splines cúbicos") 


# splines que me dan la intersección
interv = np.linspace(0, 1, 10)
plt.title("Intersección de las trayectorias")
plt.plot(splines_x(intervalo1), splines_y(intervalo1),label="T1 interpolada con splines cúbicos")
plt.plot(t1x_pol_intersec(interv), t1y_pol_intersec(interv), label="polinomio de la intersección (t1)")
plt.plot(t2x_pol_intersec(interv), t2y_pol_intersec(interv), label="Polinomio de la intersección (t2)")
plt.legend()
plt.show()

root_x = t1x_pol_intersec(f.newton_raphson(intersección_x, coord_x_intersección_derivada, 0.5, 1e-5, True))
root_y = t1y_pol_intersec(f.newton_raphson(intersección_y, coord_y_intersección_derivada, 0.5, 1e-5))

root_x = t1x_pol_intersec(f.bisect(intersección_x, 0, 1, 1e-5, True))
root_y = t1y_pol_intersec(f.bisect(intersección_y, 0, 1, 1e-5))


print(f"intersección: ({root_x}, {root_y})")

plt.ylabel("Error absoluto")
plt.xlabel("Iteraciones")
plt.legend()
plt.show()


