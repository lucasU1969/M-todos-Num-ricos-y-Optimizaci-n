import numpy as np
import matplotlib.pyplot as plt

# # Definir los ángulos equiespaciados para los puntos en la circunferencia
# num_puntos = 100
# angulos_circunferencia = np.linspace(np.pi, 2*np.pi, num_puntos)

# # Coordenadas de los puntos en la circunferencia unitaria
# x_circunferencia = np.cos(angulos_circunferencia)
# y_circunferencia = np.sin(angulos_circunferencia)

# # Definir los puntos equiespaciados en el círculo
# num_puntos_equispaciados = 10
# angulos_puntos = np.linspace(np.pi, 2*np.pi, num_puntos_equispaciados)
# x_puntos = np.cos(angulos_puntos)
# y_puntos = np.sin(angulos_puntos)

# # Calcular el coseno de cada punto
# cosenos_puntos = np.cos(angulos_puntos)

# # Calcular las coordenadas del punto de intersección
# x_interseccion = 1
# y_interseccion = np.cos(np.pi/10)

# # Crear el gráfico con proporción cuadrada
# plt.figure(figsize=(6, 6))

# # Graficar la circunferencia unitaria suave
# plt.plot(x_circunferencia, y_circunferencia, 'g-', label='Circunferencia unitaria')

# # Graficar los puntos sobre la circunferencia
# plt.scatter(x_puntos, y_puntos, color='green', zorder=3)

# # Graficar los puntos equiespaciados sobre el eje x
# plt.scatter(cosenos_puntos, np.zeros_like(cosenos_puntos), color='green', zorder=3)

# # Graficar las líneas que conectan los puntos
# for i in range(num_puntos_equispaciados):
#     plt.plot([cosenos_puntos[i], cosenos_puntos[i]], [0, y_puntos[i]], 'g--', zorder=1)

# # Graficar la línea del eje x
# plt.axhline(y=0, color='black', linestyle='-', linewidth=1, zorder=1)

# # Configurar límites de los ejes
# plt.xlim(-1.2, 1.2)
# plt.ylim(-1.2, 1.2)

# # Configurar etiquetas y título
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Efecto de función coseno sobre la circunferencia unitaria')
# plt.legend()

# # Mostrar el gráfico
# plt.grid(True)
# plt.show()



# Definir la función
# def funcion(x):
#     return (1/10) * x**3 + x

# # Generar valores para x
# x = np.linspace(-4, 4, 1000)

# # Calcular los valores de y usando la función
# y = funcion(x)

# # Graficar la función
# plt.figure(figsize=(8, 6))
# plt.plot(x, y, label=r'$\frac{1}{10}x^3 + x$', color='blue')
# plt.xlabel('x')
# plt.ylabel('y')

# plt.scatter(np.linspace(-4, 4, 8), funcion(np.linspace(-4, 4, 8)), color='red', label='Puntos equiespaciados')
# # plt.title('Gráfico de la función $\frac{1}{10}x^3 + x$')
# plt.legend()
# plt.grid(True)
# plt.show()

# Definir la función
def funcion(x):
    return (1/10) * x**3 + x

# Generar valores para x
x = np.linspace(-2.478136535, 2.478136535, 1000)

# Calcular los valores de y usando la función
y = funcion(x)

# Puntos equiespaciados
x_points = np.linspace(-2.478136535, 2.478136535, 8 )
y_points = funcion(x_points)

# Graficar la función y los puntos
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r'$\frac{1}{10}x^3 + x$', color='blue')
plt.scatter(x_points, y_points, color='k', label='Puntos no equiespaciados')

# Proyección de los puntos sobre el eje y y marcar el eje
for i in range(len(x_points)):
    plt.plot([x_points[i], 0], [y_points[i], y_points[i]], linestyle='--', color='k')
    plt.scatter([0], [y_points[i]], color='black', zorder=5)  # Marcar el eje y

plt.axvline(0, color='black', linewidth=0.5)  # Línea vertical en x=0
plt.xlabel('x')
plt.ylabel('y')
plt.title('Criterio para puntos no equiespaciados')
plt.legend()
plt.grid(True)
plt.show()