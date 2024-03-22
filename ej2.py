import matplotlib.pyplot as plt
import scipy.interpolate as sci
import numpy as np

def max_distance(x1:list[float], y1:list[float], x2:list[float], y2:list[float]) -> float:
    # las listas recibidas tienen la misma longitud
    max_dis = 0
    for i in range(len(x1)):
        if i == 0 or max_dis < np.sqrt((x1[i] - x2[i])**2 + (y1[i] - y2[i])**2):
            # max_x = x1[i]
            # max_y = y1[i]
            # max_x2 = x2[i]
            # max_y2 = y2[i]
            max_dis = np.sqrt((x1[i] - x2[i])**2 + (y1[i] - y2[i])**2)
    # print("(", max_x, ", ", max_y, ") y (", max_x2, ", ", max_y2, ")")
    return max_dis

# gráfico del ground truth
with open("mnyo_ground_truth.csv", "r") as file:
    lines = file.readlines()
    gt_x = []
    gt_y = []
    for i in lines:
        x , y = i.split()
        gt_x.append(float(x))
        gt_y.append(float(y))

    plt.plot(gt_x, gt_y)

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
intervalo = np.linspace(0,9, 100)

# acá esta hecho el lagrangiano
# lagrangiano_x = sci.lagrange(t, coords_x)
# lagrangiano_y = sci.lagrange(t, coords_y)
# plt.plot(lagrangiano_x(intervalo), lagrangiano_y(intervalo))

# hago splines cúbicos
splines_x = sci.CubicSpline(t, coords_x)
splines_y = sci.CubicSpline(t, coords_y)

plt.plot(splines_x(intervalo), splines_y(intervalo))

# calculo el error máximo de aproximación de la trayectoria por medio de splines
print(max_distance(gt_x, gt_y, splines_x(intervalo), splines_y(intervalo)))

plt.plot( 5.63746987873047 ,  1.0, marker='o')
plt.plot( 8.001721605855845 ,  0.9547441622189221, marker='o')

# interpolo la segunda trayectoria usando splines
w = range(0, 4) 
intervalo = np.linspace(0, 3, 100)

splines2_x = sci.CubicSpline(w, coords2_x)
splines2_y =sci.CubicSpline(w, coords2_y)
plt.plot(splines2_x(intervalo), splines2_y(intervalo), label="trayectoria2")

# grafico la segunda trayectoria (splines de grado 1)
plt.plot(coords2_x, coords2_y)

plt.show()


