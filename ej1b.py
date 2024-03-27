import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

# Definir una función bidimensional (por ejemplo, z = x^2 + y^2)
def funcion(x, y):
    return np.cos(x) + np.sin(y)


def fb(x1: float, x2: float) -> float:
    return 0.75*((np.e)**((-(10*x1 - 2)**2)/4 - ((9*x2 - 2)**2)/4)) \
        + 0.65*((np.e)**(- ((9*x1 + 1)**2)/9 - ((10*x2 + 1)**2)/2)) \
        + 0.55*((np.e)**(- ((9*x1 - 6)**2)/4 - ((9*x2 - 3)**2)/4)) \
        - 0.01*((np.e)**(- ((9*x1 - 7)**2)/4 - ((9*x2 - 3)**2)/4))



# Generar la cuadrícula de puntos de prueba
x = np.linspace(-1, 1, 5)  # 5 puntos en el eje x
y = np.linspace(-1, 1, 5)  # 5 puntos en el eje y
X, Y = np.meshgrid(x, y)

# Calcular los valores de la función en la cuadrícula
Z = fb(X, Y)


x_fino = np.linspace(-1, 1, 100)  # 100 puntos en el eje x
y_fino = np.linspace(-1, 1, 100)  # 100 puntos en el eje y
z_fino = fb(x_fino, y_fino)
f_interp_cúbica = interp2d(x, y, Z, kind='cubic')  
Z_interp_cúbica = f_interp_cúbica(x_fino, y_fino)
f_interp_lineal = interp2d(x, y, Z, kind='linear')
z_interp_lineal = f_interp_lineal(x_fino, y_fino)

def error_absoluto_cúbica(x1:float, x2:float):
    return np.abs(fb(x1, x2) - f_interp_cúbica(x1, x2))

def error_absoluto_lineal(x1:float, x2:float):
    return np.abs(fb(x1, x2) - f_interp_lineal(x1, x2))


error_cúbica = np.abs(z_fino - Z_interp_cúbica)
error_lineal = np.abs(z_fino - z_interp_lineal)

print("error máximo de la interpolación bilineal:", np.max(error_lineal))
print("error promedio de la interpolación bilineal:", np.mean(error_lineal))
print("error máximo de la interpolación bicúbica:", np.max(error_cúbica))
print("error promedio de la interpolación bicúbica:", np.mean(error_cúbica))


# falta el criterio de selección de puntos no equiespaciados
x_censadas = np.linspace(-11/10, 11/10, 10)
x_censadas = x_censadas + (1/10)*np.power(x_censadas, 3)
y_censadas = np.linspace(-11/10, 11/10, 10)
y_censadas = y_censadas + (1/10)*np.power(y_censadas, 3)
x_censadas, y_censadas = np.meshgrid(x_censadas, y_censadas)
z_censadas = fb(x_censadas, y_censadas)


interp_cúbica_no_equiespaciada = interp2d(x_censadas, y_censadas, z_censadas, kind='cubic')
z_interp_cúbica_no_equiespaciada = interp_cúbica_no_equiespaciada(x_fino, y_fino)
error_cúbica_no_equiespaciada = np.abs(z_fino - z_interp_cúbica_no_equiespaciada)

print("error máximo de la interpolación bicúbica no equiespaciada:", np.max(error_cúbica_no_equiespaciada))
print("error promedio de la interpolación bicúbica no equiespaciada:", np.mean(error_cúbica_no_equiespaciada))

# ------------------------------

# fig = plt.figure()
# ax1 = fig.add_subplot(122, projection='3d')
# ax0 = fig.add_subplot(121, projection='3d')

# X_fino, Y_fino = np.meshgrid(x_fino, y_fino)

# ax0.plot_surface(X_fino, Y_fino, error_lineal, cmap='viridis')
# ax1.plot_surface(X_fino, Y_fino, error_cúbica, cmap='viridis')
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# ax1.set_zlabel('Error Absoluto de Interpolación Bicúbica')

# ax0.set_xlabel('X')
# ax0.set_ylabel('Y')
# ax0.set_zlabel('Error Absoluto de Interpolación Bilineal')
# fig.suptitle('Error Absoluto de los métodos de interpolación')
# plt.show()

# ------------------------------

# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.contourf(x_fino, y_fino, error_lineal, cmap='viridis')
# plt.colorbar(label='Valor del error absoluto')
# plt.title('Error de la interpolación lineal')

# plt.subplot(1, 2, 2)
# plt.contourf(x_fino, y_fino, error_cúbica, cmap='viridis')
# plt.colorbar(label='Valor del error absoluto')
# plt.title('Error de la interpolación cúbica')

# plt.tight_layout()
# plt.show()


#---------------------------------
# voy a calcular el error para distintos n

# error_p_lineal = []
# error_p_cúbica = []

# for i in range(5, 21):
#     x = np.linspace(-1, 1, i)  # 5 puntos en el eje x
#     y = np.linspace(-1, 1, i)  # 5 puntos en el eje y
#     X, Y = np.meshgrid(x, y)

#     # Calcular los valores de la función en la cuadrícula
#     Z = fb(X, Y)
#     f_interp_cúbica = interp2d(x, y, Z, kind='cubic')  
#     Z_interp_cúbica = f_interp_cúbica(x_fino, y_fino)
#     f_interp_lineal = interp2d(x, y, Z, kind='linear')
#     z_interp_lineal = f_interp_lineal(x_fino, y_fino)

#     error_p_lineal.append(np.mean(np.abs(z_fino - z_interp_lineal)))
#     error_p_cúbica.append(np.mean(np.abs(z_fino - Z_interp_cúbica)))

# plt.plot(np.power(np.array(range(5,21)), 2), error_p_lineal, label="Error por interpolación bilineal")
# plt.plot(np.power(np.array(range(5,21)), 2), error_p_cúbica, label="Error por interpolación bicubica")

# plt.ylabel("Error promedio")
# plt.xlabel("Cantidad de puntos")
# plt.legend()
# plt.show()