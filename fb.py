import numpy as np
import matplotlib.pyplot as plt

def fb(x1: float, x2: float) -> float:
    x = (0.75*np.e)**((-(10*x1 - 2)**2)/4 - ((9*x2 - 2)**2)/4) \
        + (0.65*np.e)**(- ((9*x1 + 1)**2)/9 - ((10*x2 + 1)**2)/2) \
        + (0.55*np.e)**(- ((9*x1 - 6)**2)/4 - ((9*x2 - 3)**2)/4) \
        - (0.01*np.e)**(- ((9*x1 - 7)**2)/4 - ((9*x2 - 3)**2)/4)
    return x
coords_x = np.linspace(-1, 1, 10)




plt.xlabel('X')
plt.ylabel('Y')
plt.title('Funcion fb')
plt.show()
