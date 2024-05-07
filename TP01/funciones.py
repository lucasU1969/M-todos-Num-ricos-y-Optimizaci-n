import numpy as np
import scipy.interpolate as sci
import matplotlib.pyplot as plt

def fa(x:float) -> float:
    return (0.3**abs(x))*np.sin(4*x) - np.tanh(2*x) + 2

def g_0(x:float) -> float:
    return np.cos(x) * 4

def distancia(x0:float, y0:float, x1:float, y1:float) -> float:
    """
    retorna la distancia entre los puntos (x0, y0) y (x1, y1)
    """
    return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)

def error_máximo(x0:list[float], y0:list[float], x1:list[float], y1:list[float]) -> float:
    """
    x0 e y0 son listas con las coordenadas de una función y x1 e y1 son listas con las coordenadas de otra función
    a cada x0[i] le corresponde y0[i] y a cada x1[i] le corresponde y1[i]
    se calcula el error máximo entre (x0[i], y0[i]) y (x1[i], y1[i])
    suponemos qeu todas las listas tienen la misma longitud
    """
    return np.max(distancia(x0, y0, x1, y1))

def error_absoluto(x0:float, x1:float) -> float:
    return np.abs(x0 - x1)

def errores_sobre_intervalo(x0:list[float], y0:list[float], x1:list[float], y1:list[float]) -> list[float]:
    """
    retorna una lista con las distancias entre los puntos (x0[i], y0[i]) y (x1[i], y1[i])
    """
    return [distancia(x0[i], y0[i], x1[i], y1[i]) for i in range(len(x0))]

def error_promedio(x0:list[float], y0:list[float], x1:list[float], y1:list[float]) -> float:
    """
    retorna la distancia promedio entre los puntos (x0[i], y0[i]) y (x1[i], y1[i])
    """
    return np.mean(errores_sobre_intervalo(x0, y0, x1, y1))

def minimizar_error_promedio(f, intervalo, interpolación, max_cantidad_puntos:int=20):
    """
    calcula la cantidad de puntos que se deben tomar para encontrar el mínimo error promedio
    """
    errores_prom = []
    for i in range(2, max_cantidad_puntos + 1):
        x_censados = np.linspace(intervalo[0], intervalo[-1], i)
        y_censados = f(x_censados)
        interpolación = interpolación(x_censados, y_censados)
        errores_prom.append(error_promedio(intervalo, interpolación(intervalo), intervalo, interpolación(intervalo)))
    return np.argmin(errores_prom) + 2

def ev_errores_prom(x:list[float], y:list[float], interpolación, f_original, g):
    errores_promedio = []
    for i in range(2, 21):
        xcensadas = np.linspace(x[0], x[len(x) - 1], i)
        xcensadas = np.array([g(x) for x in xcensadas])
        ycensadas = f_original(xcensadas)
        inter = interpolación(xcensadas, ycensadas)
        errores_promedio.append(error_promedio(x, y, x, inter(x)))
    return errores_promedio

def newton_raphson(f, df, x0, tol, graf=False):
    x_approx = [x0]
    x = x0
    ite = 0
    while abs(f(x_approx[-1])) > tol and ite < 50:
        x_approx.append(x_approx[-1] - f(x_approx[-1])/df(x_approx[-1]))
    if graf:
        plt.plot(range(len(x_approx)), f(np.array(x_approx)), label='Newton-Raphson')
    return x_approx[-1]

def bisect(f, a, b, tol=1e-5, graf=False):
    vals = []
    while abs(f(a)) > tol:
        c = (a + b)/2
        vals.append(c)
        if f(c) == 0:
            return c
        if f(a)*f(c) < 0:
            b = c
        else:
            a = c
    if graf:
        plt.plot(range(len(vals)), f(np.array(vals)), label='Bisección')
    return c


def ev_error_lagrange_no_equiespaciado(coords_x, coords_y):
    errores_prom = []
    for i in range(2, 21):
        x3 = np.cos(np.linspace(np.pi, np.pi*2, i))*4
        y3 = fa(x3)
        pol_lagrange = sci.lagrange(x3, y3)
        errores_prom.append(error_promedio(coords_x, coords_y, coords_x, pol_lagrange(coords_x)))
    return errores_prom

def ev_error_lagrange_equiespaciado(coords_x, coords_y):
    errores_prom = []
    for i in range(2, 21):
        x3 = np.linspace(-4, 4, i)
        y3 = fa(x3)
        pol_lagrange = sci.lagrange(x3, y3)
        errores_prom.append(error_promedio(coords_x, coords_y, coords_x, pol_lagrange(coords_x)))
    return errores_prom

def ev_error_splines_equiespaciados(coords_x, coords_y):
    errores_prom = []
    for i in range(2, 21):
        x3 = np.linspace(-4, 4, i)
        y3 = fa(x3)
        pol_lagrange = sci.CubicSpline(x3, y3, bc_type='clamped')
        errores_prom.append(error_promedio(coords_x, coords_y, coords_x, pol_lagrange(coords_x)))
    return errores_prom

def ev_error_splines_no_equiespaciados(coords_x, coords_y):
    errores_prom = []
    for i in range(2, 21):
        xcensadas = np.linspace(-2.478136535, 2.478136535, 8)
        xcensadas = xcensadas + (1/10)*np.power(xcensadas, 3)
        ycensadas = fa(xcensadas)
        pol_lagrange = sci.CubicSpline(xcensadas, ycensadas, bc_type='clamped')
        errores_prom.append(error_promedio(coords_x, coords_y, coords_x, pol_lagrange(coords_x)))
    return errores_prom