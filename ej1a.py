import numpy as np
import scipy.interpolate as sci
import matplotlib.pyplot as plt

def fa(x:float) -> float:
    return (0.3**abs(x))*np.sin(4*x) - np.tanh(2*x) + 2

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# EStAS SON LAS FUNCIONES QUE VAN EN EL INFORME

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

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def error_relativo(x0:float, y0:float, x1:float, y1:float) -> float:
    return distancia(x0, y1, x1, y1) / distancia(x0, y0, 0, 0)

def error_relativo_promedio(x0:list[float], y0:list[float], x1:list[float], y1:list[float]) -> float:
    # se supone que todas las listas tienen la misma longitud
    sum = 0
    for i in range(len(x0)):
        sum += error_relativo(x0[i], y0[i], x1[i], y1[i])
    return sum/len(x0)

def errores_relativos_no_equiespaciados(x:list[float], y:list[float]):
    errores_rel = []
    for i in range(2, 21):
        x3 = np.linspace(np.pi, np.pi*2, i)
        x3 = np.cos(x3) * 4
        y3 = fa(x3)
        pol_lagrange = sci.lagrange(x3, y3)
        errores_rel.append(error_relativo_promedio(x, y, x, pol_lagrange(x)))
    return errores_rel

def errores_relativos_equiespaciados(x:list[float], y:list[float]):
    errores_rel = []
    for i in range(2, 21):
        x3 = np.linspace(-4, 4, i)
        y3 = fa(x3)
        pol_lagrange = sci.lagrange(x3, y3)
        errores_rel.append(error_relativo_promedio(x, y, x, pol_lagrange(x)))
    return errores_rel

def errores_promedio_no_equiespaciados(x:list[float], y:list[float]):
    errores_prom = []
    for i in range(2, 21):
        x3 = np.linspace(np.pi, np.pi*2, i)
        x3 = np.cos(x3) * 4
        y3 = fa(x3)
        pol_lagrange = sci.lagrange(x3, y3)
        errores_prom.append(error_promedio(x, y, x, pol_lagrange(x)))
    return errores_prom

def errores_absolutos_no_equiespaciados(x:list[float], y:list[float]) -> list[float]:
    errores_abs = []
    for i in range(2, 21):
        x3 = np.linspace(np.pi, np.pi*2, i)
        x3 = np.cos(x3) * 4
        y3 = fa(x3)
        pol_lagrange = sci.lagrange(x3, y3)
        errores_abs.append(error_absoluto(x, y, x, pol_lagrange(x)))
    return errores_abs

def errores_promedio(x, y):

    errores_prom = []
    for i in range(2, 21):
        x3 = np.linspace(-4, 4, i)
        y3 = fa(x3)
        pol_lagrange = sci.lagrange(x3, y3)
        errores_prom.append(error_promedio(x, y, x, pol_lagrange(x)))
    return errores_prom

def errores_absolutos(x, y):
    errores_abs = []
    for i in range(2, 21):
        x3 = np.linspace(-4, 4, i)
        y3 = fa(x3)
        pol_lagrange = sci.lagrange(x3, y3)
        errores_abs.append(error_máximo(x, y, x, pol_lagrange(x)))
    return errores_abs

def errores_relativos_prom(x, y):
    errores_rel = []
    for i in range(2, 21):
        x3 = np.linspace(-4, 4, i)
        y3 = fa(x3)
        pol_lagrange = sci.lagrange(x3, y3)
        errores_rel.append((x, y, x, pol_lagrange(x)))
    return errores_rel

# def error_relativo

def ev_error_splines_equiespaciados(x, y):
    errores_rel = []
    for i in range(2, 21):
        x3 = np.linspace(-4, 4 ,i)
        y3 = fa(x3)
        spline = sci.CubicSpline(x3, y3)
        errores_rel.append(error_promedio(x, y, x, spline(x)))
    return errores_rel

def ev_error_splines_no_equiespaciados(x, y):
    errores = []
    for i in range(2, 21):
        x_censadas = np.linspace(-2.478136535, 2.478136535, i)
        x_censadas = x_censadas + (1/10)*np.power(x_censadas, 3)
        y_censadas = fa(x_censadas)
        spline = sci.CubicSpline(x_censadas, y_censadas) 
        errores.append(error_promedio(x, y, x, spline(x)))
    return errores  

def main():

    coords_x = np.linspace(-4, 4, 800)
    coords_y = fa(coords_x)

    dataset_x = np.linspace(-4, 4, 8)
    dataset_y = fa(dataset_x)

    dataset1_x = np.linspace(-4, 4, 12)
    dataset1_y = fa(dataset1_x)

    dataset2_x = np.linspace(-4, 4, 16)
    dataset2_y = fa(dataset2_x)

    # errores_prom = errores_promedio(coords_x, coords_y)
    # OBS IMPORTANTE: PARA LA COMPUTADORA EL PROMEDIO DEL ERROR RELATIVO ES 0
    # errores_abs = errores_absolutos(coords_x, coords_y)
    # plt.yscale('log')
    # plt.plot(range(2,21), errores_prom)
    # plt.plot(range(2,21), errores_abs)
    # plt.show()

    # plt.plot()

    lagrangiano = sci.lagrange(dataset_x, dataset_y)
    # lagrangiano1 = sci.lagrange(dataset1_x, dataset1_y)
    # lagrangiano2 = sci.lagrange(dataset2_x, dataset2_y)

    # plt.title("Ejemplo del efecto de Runge")
    # plt.plot(coords_x, errores_sobre_dom(coords_x, coords_y, coords_x, lagrangiano(coords_x)), label='Error del polinomio de grado 7')
    # plt.plot(coords_x, errores_sobre_dom(coords_x, coords_y, coords_x, lagrangiano1(coords_x)), label='Error del polinomio de grado 11')
    # plt.plot(coords_x, errores_sobre_dom(coords_x, coords_y, coords_x, lagrangiano2(coords_x)), label='Error del polinomio de grado 15')
    # plt.xlabel('x')
    # plt.ylabel('Error absoluto')
    # plt.yscale('symlog')


    # print(error_relativo_promedio(coords_x, coords_y, coords_x, lagrangiano(coords_x)))

    # error_absoluto_máximo = error_máximo(coords_x, coords_y, coords_x, lagrangiano(coords_x))

    # print("Error absoluto máximo:", error_absoluto_máximo)

    error_prom = error_promedio(coords_x, coords_y, coords_x, lagrangiano(coords_x))
    # error_pro = np.mean(array(errores_absolutos))
    print("Error promedio:", error_prom)
    # plt.title('Error absoluto sobre el intervalo')
    # plt.xlabel('x')
    # plt.ylabel('Error')
    # plt.plot(coords_x, errores_sobre_dom(coords_x, coords_y, coords_x, lagrangiano(coords_x)), label='Error absoluto sobre el intervalo')
    # plt.scatter(dataset_x, [0]*8, label='Puntos equiespaciados', color='k')
    # plt.legend()
    # plt.show()

    # plt.title("Interpolación por polinomio de Lagrange")
    # plt.plot(coords_x, coords_y, color='k', label='Función original')
    # plt.plot(coords_x, lagrangiano(coords_x), label='Polinomio de Lagrange de grado 7')
    # plt.scatter(dataset_x, dataset_y, label='Puntos equiespaciados')
    # plt.xlabel('x')
    # plt.ylabel('y')

# quiero graficar la evolución del error sobre el dominio cuando n cambia
    



    # plt.legend()
    # plt.show()




if __name__ == "__main__":
    main()
