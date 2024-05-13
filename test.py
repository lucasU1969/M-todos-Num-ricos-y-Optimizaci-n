# grafico de las isoclinas para los nuevos valores de los parámetros: 
# Para entender la dinámica general del sistema, seleccionamos \(K_1\), \(K_2\), \(\alpha_{12}\) y \(\alpha_{21}\) representativos de cada uno de los casos para observar el comportamiento general de las soluciones. En el caso 1,  \(\frac {K_1}{\alpha_12} > K_2\) y \(K_1 > \frac {K_2}{\alpha_{21}}\) por lo que estableceremos \(K_1 = 1000\), \(K_2 = 800\), \(\alpha_{12} = 1.3\) y \(\alpha_{21} = 0.9\) de manera que cumplan con las restricciones anteriores. Por otro lado, para el caso 2, nos restringimos a \(\frac {K_1}{\alpha_12} < K_2\) y \(K_1 < \frac {K_2}{\alpha_{21}}\) y, valores para los parámetros que cumplen dichos límites son \(K_1 = 500\), \(K_2 = 1200\), \(\alpha_{12} = 0.5\)  y \(\alpha_{21} = 3\). El caso 3 presenta los requisitos \(\frac {K_1}{\alpha_12} > K_2\) y \(K_1 < \frac {K_2}{\alpha_{21}}\), por lo que proponemos \(K_1 = 1500\), \(K_2 = 500\), \(\alpha_{12} = 2.1\) y \(\alpha_{21} = 0.2\). Finalmente, para el caso 4 proponemos \(K_1 = 300\), \(K_2 = 400\), \(\alpha_{12} = 1\) y \(\alpha_{21} = 1.5\) para satisfacer las condiciones \(\frac {K_1}{\alpha_12} < K_2\) y \(K_1 > \frac {K_2}{\alpha_{21}}\). Las isoclinas de estos sistemas se representan en la Figura 2.  

import numpy as np
import matplotlib.pyplot as plt

#constantes caso 1
K1_1 = 1200
K2_1 = 400
alpha12_1 = 2.7
alpha21_1 = 0.5

#constantes caso 2
K1_2 = 500
K2_2 = 1200
alpha12_2 = 0.5
alpha21_2 = 2

#constantes caso 4
K1_3 = 1500
K2_3 = 800
alpha12_3 = 1.5
alpha21_3 = 0.4

#constantes caso 3
K1_4 = 1200
K2_4 = 800
alpha12_4 = 2.7
alpha21_4 = 1.5


def isoclina_de_N1(N1, K1, alpha12): 
    return (K1 - N1)/alpha12

def isoclina_de_N2(N1, K2, alpha21): 
    return K2 - N1*alpha21

#gráfico de los cuatro casos en subplots en fila
N1 = np.linspace(0, 2000, 1000)

fig, axs = plt.subplots(1, 4)

axs[0].plot(N1, isoclina_de_N1(N1, K1_1, alpha12_1), color='blue', label='dN1/dt = 0')
axs[0].plot(N1, isoclina_de_N2(N1, K2_1, alpha21_1), color='red', label='dN2/dt = 0')
axs[0].set_title('Caso 1')
axs[0].set_xlabel('N1')
axs[0].set_ylabel('N2')
axs[0].legend()
axs[0].set_xlim(left=0)  # Set x-axis limit to start from 0
axs[0].set_ylim(bottom=0)  # Set y-axis limit to start from 0
axs[0].grid(True)  # Add grid

axs[1].plot(N1, isoclina_de_N1(N1, K1_2, alpha12_2), color='blue', label='dN1/dt = 0')
axs[1].plot(N1, isoclina_de_N2(N1, K2_2, alpha21_2), color='red', label='dN2/dt = 0')
axs[1].set_title('Caso 2')
axs[1].set_xlabel('N1')
axs[1].set_ylabel('N2')
axs[1].legend()
axs[1].set_xlim(left=0)  # Set x-axis limit to start from 0
axs[1].set_ylim(bottom=0)  # Set y-axis limit to start from 0
axs[1].grid(True)  # Add grid

axs[2].plot(N1, isoclina_de_N1(N1, K1_4, alpha12_4), color='blue', label='dN1/dt = 0')
axs[2].plot(N1, isoclina_de_N2(N1, K2_4, alpha21_4), color='red', label='dN2/dt = 0')
axs[2].set_title('Caso 3')
axs[2].set_xlabel('N1')
axs[2].set_ylabel('N2')
axs[2].legend()
axs[2].set_xlim(left=0)  # Set x-axis limit to start from 0
axs[2].set_ylim(bottom=0)  # Set y-axis limit to start from 0
axs[2].grid(True)  # Add grid

axs[3].plot(N1, isoclina_de_N1(N1, K1_3, alpha12_3), color='blue', label='dN1/dt = 0')
axs[3].plot(N1, isoclina_de_N2(N1, K2_3, alpha21_3), color='red', label='dN2/dt = 0')
axs[3].set_title('Caso 4')
axs[3].set_xlabel('N1')
axs[3].set_ylabel('N2')
axs[3].legend()
axs[3].set_xlim(left=0)  # Set x-axis limit to start from 0
axs[3].set_ylim(bottom=0)  # Set y-axis limit to start from 0
axs[3].grid(True)  # Add grid

plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.1)
plt.show()

