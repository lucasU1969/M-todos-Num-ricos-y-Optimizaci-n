import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

def fb(x1: float, x2: float) -> float:
    return 0.75*((np.e)**((-(10*x1 - 2)**2)/4 - ((9*x2 - 2)**2)/4)) \
        + 0.65*((np.e)**(- ((9*x1 + 1)**2)/9 - ((10*x2 + 1)**2)/2)) \
        + 0.55*((np.e)**(- ((9*x1 - 6)**2)/4 - ((9*x2 - 3)**2)/4)) \
        - 0.01*((np.e)**(- ((9*x1 - 7)**2)/4 - ((9*x2 - 3)**2)/4))

x = np.linspace(-1, 1, 10) 
y = np.linspace(-1, 1, 10) 
X, Y = np.meshgrid(x, y)
Z = fb(X, Y)

x_values = np.linspace(-1, 1, 100) 
y_values = np.linspace(-1, 1, 100)
X_values, Y_values = np.meshgrid(x_values, y_values)
Z_values = fb(X_values, Y_values)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_values, Y_values, Z_values, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Función Original')
plt.show()

f_interp_cúbica = interp2d(X, Y, Z, kind='cubic')  
Z_interp_cúbica = f_interp_cúbica(x_values, y_values)
f_interp_lineal = interp2d(x, y, Z, kind='linear')
z_interp_lineal = f_interp_lineal(x_values, y_values)


error_cúbica = np.abs(Z_values - Z_interp_cúbica)
error_lineal = np.abs(Z_values - z_interp_lineal)

print("error máximo de la interpolación bilineal:", np.max(error_lineal))
print("error promedio de la interpolación bilineal:", np.mean(error_lineal))
print("error máximo de la interpolación bicúbica:", np.max(error_cúbica))
print("error promedio de la interpolación bicúbica:", np.mean(error_cúbica))

x_grid, y_grid = np.meshgrid(x, y)

plt.scatter(x_grid, y_grid, color='blue', marker='o')
plt.plot(x_grid, y_grid, color='blue')
plt.plot(y_grid, x_grid, color='blue')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Puntos censados equiespaciados')
plt.grid(True)
plt.show()


x_censadas = np.linspace(-0.9216989942, 0.9216989942, 10)
x_censadas = x_censadas + (1/10)*np.power(x_censadas, 3)
y_censadas = np.linspace(-0.9216989942, 0.9216989942, 10)
y_censadas = y_censadas + (1/10)*np.power(y_censadas, 3)
x_censadas, y_censadas = np.meshgrid(x_censadas, y_censadas)
z_censadas = fb(x_censadas, y_censadas)


interp_cúbica_no_equiespaciada = interp2d(x_censadas, y_censadas, z_censadas, kind='cubic')
z_interp_cúbica_no_equiespaciada = interp_cúbica_no_equiespaciada(x_values, y_values)
error_cúbica_no_equiespaciada = np.abs(Z_values - z_interp_cúbica_no_equiespaciada)

interp_lineal_no_equiespaciada = interp2d(x_censadas, y_censadas, z_censadas, kind='linear')
z_interp_lineal_no_equiespaciada = interp_lineal_no_equiespaciada(x_values, y_values)
error_lineal_no_equiespaciada = np.abs(Z_values - z_interp_lineal_no_equiespaciada)

print("error máximo de la interpolación bicúbica no equiespaciada:", np.max(error_cúbica_no_equiespaciada))
print("error promedio de la interpolación bicúbica no equiespaciada:", np.mean(error_cúbica_no_equiespaciada))
print("error máximo de la interpolación bilineal no equiespaciada:", np.max(error_lineal_no_equiespaciada))
print("error promedio de la interpolación bilineal no equiespaciada:", np.mean(error_lineal_no_equiespaciada))
x_grid, y_grid = np.meshgrid(x_censadas, y_censadas)

plt.scatter(x_grid, y_grid, color='blue', marker='o')
plt.plot(x_grid, y_grid, color='blue')
plt.plot(y_grid, x_grid, color='blue')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Puntos censados no equiespaciados')
plt.grid(True)
plt.show()
# ------------------------------

fig = plt.figure()
ax1 = fig.add_subplot(122, projection='3d')
ax0 = fig.add_subplot(121, projection='3d')

X_fino, Y_fino = np.meshgrid(x_values, y_values)

ax0.plot_surface(X_fino, Y_fino, error_lineal, cmap='viridis')
ax1.plot_surface(X_fino, Y_fino, error_cúbica, cmap='viridis')

ax1.title.set_text('Interpolación Bicúbica')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Error Absoluto')

ax0.title.set_text('Interpolación Bilineal')
ax0.set_xlabel('X')
ax0.set_ylabel('Y')
ax0.set_zlabel('Error Absoluto')
fig.suptitle('Error Absoluto de los métodos de interpolación')
plt.show()

# ------------------------------

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.contourf(x_values, y_values, error_lineal, cmap='viridis')
plt.colorbar(label='Valor del error absoluto')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Error de la interpolación lineal')

plt.subplot(1, 2, 2)
plt.contourf(x_values, y_values, error_cúbica, cmap='viridis')
plt.colorbar(label='Valor del error absoluto')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Error de la interpolación cúbica')

plt.tight_layout()
plt.show()


#---------------------------------
# bilineal equiespaciada vs no equiespaciada

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.contourf(x_values, y_values, error_lineal, cmap='viridis')
plt.colorbar(label='Valor del error absoluto')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Error de la interpolación lineal equiespaciada')

plt.subplot(1, 2, 2)
plt.contourf(x_values, y_values, error_lineal_no_equiespaciada, cmap='viridis')
plt.colorbar(label='Valor del error absoluto')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Error de la interpolación lineal no equiespaciada')

plt.tight_layout()
plt.show()


#---------------------------------
# bicúbica equiespaciada vs no equiespaciada

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.contourf(x_values, y_values, error_cúbica, cmap='viridis')
plt.colorbar(label='Valor del error absoluto')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Error de la interpolación cúbica equiespaciada')

plt.subplot(1, 2, 2)
plt.contourf(x_values, y_values, error_cúbica_no_equiespaciada, cmap='viridis')
plt.colorbar(label='Valor del error absoluto')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Error de la interpolación cúbica no equiespaciada')

plt.tight_layout()
plt.show()

# --------------------------------
# bicúbica no equiespaciada vs bilineal no equiespaciada

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.contourf(x_values, y_values, error_cúbica_no_equiespaciada, cmap='viridis')
plt.colorbar(label='Valor del error absoluto')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Error de la interpolación cúbica no equiespaciada')

plt.subplot(1, 2, 2)
plt.contourf(x_values, y_values, error_lineal_no_equiespaciada, cmap='viridis')
plt.colorbar(label='Valor del error absoluto')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Error de la interpolación lineal no equiespaciada')

plt.tight_layout()
plt.show()


# --------------------------------
# error según la cantidad de puntos

error_p_lineal = []
error_p_cúbica = []

for i in range(5, 21):
    x = np.linspace(-1, 1, i)  # 5 puntos en el eje x
    y = np.linspace(-1, 1, i)  # 5 puntos en el eje y
    X, Y = np.meshgrid(x, y)

    # Calcular los valores de la función en la cuadrícula
    Z = fb(X, Y)
    f_interp_cúbica = interp2d(x, y, Z, kind='cubic')  
    Z_interp_cúbica = f_interp_cúbica(x_values, y_values)
    f_interp_lineal = interp2d(x, y, Z, kind='linear')
    z_interp_lineal = f_interp_lineal(x_values, y_values)

    error_p_lineal.append(np.mean(np.abs(Z_values - z_interp_lineal)))
    error_p_cúbica.append(np.mean(np.abs(Z_values - Z_interp_cúbica)))

plt.title("Error promedio en función de la cantidad de puntos")
plt.plot(np.power(np.array(range(5,21)), 2), error_p_lineal, label="Error por interpolación bilineal")
plt.plot(np.power(np.array(range(5,21)), 2), error_p_cúbica, label="Error por interpolación bicúbica")

plt.ylabel("Error promedio")
plt.xlabel("Cantidad de puntos")
plt.legend()
plt.show()

# --------------------------------
# grafico las aproximaciones en 2d

# # Graficar los resultados
plt.figure(figsize=(12, 6))

# Gráfico de contorno para la interpolación cúbica
plt.subplot(1, 2, 1)
plt.contourf(x_values, y_values, Z_interp_cúbica, cmap='viridis')
plt.colorbar()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Interpolación Cúbica')

# Gráfico de contorno para la interpolación lineal
plt.subplot(1, 2, 2)
plt.contourf(x_values, y_values, z_interp_lineal, cmap='viridis')
plt.colorbar()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Interpolación Lineal')

plt.tight_layout()
plt.show()