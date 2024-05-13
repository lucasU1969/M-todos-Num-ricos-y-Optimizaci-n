import numpy as np
import matplotlib.pyplot as plt

# Definir la función que representa el campo vectorial
def vector_field(x, y):
    u = x  # Componente x del vector
    v = y  # Componente y del vector
    return u, v

# Crear una malla de puntos
x = np.linspace(-3, 3, 20)
y = np.linspace(-3, 3, 20)
X, Y = np.meshgrid(x, y)

# Calcular los componentes del campo vectorial en cada punto de la malla
U, V = vector_field(X, Y)

# Crear la figura y el eje
fig, ax = plt.subplots(figsize=(8, 6))

# Graficar el campo vectorial
q = ax.quiver(X, Y, U, V, pivot='mid', angles='xy', scale_units='xy', scale=1)

# Ajustar los límites del eje
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])

# Agregar una leyenda que muestra la escala del campo vectorial
plt.quiverkey(q, 0.9, 0.95, 1, '1', labelpos='E', coordinates='figure')

# Agregar etiquetas a los ejes
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Mostrar la figura
plt.show()


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Definir la función que representa el campo vectorial
def vector_field(x, y):
    u = x  # Componente x del vector
    v = y  # Componente y del vector
    return u, v

# Crear una malla de puntos
x = np.linspace(-3, 3, 20)
y = np.linspace(-3, 3, 20)
X, Y = np.meshgrid(x, y)

# Calcular los componentes del campo vectorial en cada punto de la malla
U, V = vector_field(X, Y)

# Crear la figura y el eje
fig, ax = plt.subplots(figsize=(8, 6))

# Graficar el campo vectorial con Seaborn
q = sns.quiverplot(X, Y, U, V, ax=ax)

# Ajustar los límites del eje
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])

# Agregar una leyenda que muestra la escala del campo vectorial
plt.quiverkey(q, 0.9, 0.95, 1, '1', labelpos='E', coordinates='figure')

# Agregar etiquetas a los ejes
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Mostrar la figura
plt.show()
