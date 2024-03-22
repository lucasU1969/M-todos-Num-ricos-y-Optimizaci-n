import matplotlib.pyplot as plt
import scipy.interpolate as sci
import numpy as np

with open("mnyo_mediciones.csv", "r") as file:

    # obtener las coordenadas x e y de la trayectoria
    lines = file.readlines()
    coords_x = []
    coords_y = []
    for i in lines:
        x,y = i.split()
        y = y

        coords_x.append(float(x))
        coords_y.append(float(y))
    print(coords_x)
    print(coords_y)

    # gráfico de la trayectoria
    plt.plot(coords_x, coords_y)


    # hay que interpolar ambas coords de la trayectoria
    t = range(0, 10)
    intervalo = np.linspace(0,9, 100)
    lagrangiano_x = sci.lagrange(t, coords_x)
    lagrangiano_y = sci.lagrange(t, coords_y)

    # plt.plot(lagrangiano_x(intervalo), lagrangiano_y(intervalo))

    # hago splines cúbicos
    splines_x = sci.CubicSpline(t, coords_x)
    splines_y = sci.CubicSpline(t, coords_y)

    plt.plot(splines_x(intervalo), splines_y(intervalo))
    plt.show()

    