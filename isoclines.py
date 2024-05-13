import matplotlib.pyplot as plt
import numpy as np

# Define the parameters
K1 = 8
K2 = 7
alpha12 = 1
alpha21 = 1
alpha_21 = 0.8
alpha_12 = 0.8

# Define the functions
def i(N1):
    return K2 - alpha21*N1

def j(N1):
    return K1/alpha12 - N1/alpha12

def k(N1):
    return K2 - alpha_21*N1

# Generate N1 values
N1 = np.linspace(0, 10, 400)

# Generate i and j values
i_values = i(N1)
j_values = j(N1)
k_values = k(N1)

# Create the subplots
fig, axs = plt.subplots(1,4)

# Plot the lines with the isocline of N2 above the isocline of N1 for the first subplot
axs[0].plot(N1, j_values, color='blue', label=r'Isoclina de $N_{1}$')
axs[0].plot(N1, i_values, color='red', label=r'Isoclina de $N_{2}$')

# Mark the intersections with the axes for N1 isocline for the first subplot
axs[0].scatter([0, K1], [K1/alpha12, 0], color='blue')
axs[0].annotate(r'$\frac{K_1}{\alpha_{12}}$', (0, K1/alpha12), textcoords="offset points", xytext=(-10,-10), ha='center', color='blue', fontsize=14)
axs[0].annotate(r'$K_1$', (K1, 0), textcoords="offset points", xytext=(-10,-20), ha='center', color='blue', fontsize=16)

# Mark the intersections with the axes for N2 isocline for the first subplot
axs[0].scatter([0, K2/alpha21], [K2, 0], color='red')
axs[0].annotate(r'$K_2$', (0, K2), textcoords="offset points", xytext=(-10,-10), ha='center', color='red', fontsize=14)
axs[0].annotate(r'$\frac{K_2}{\alpha_{21}}$', (K2/alpha21, 0), textcoords="offset points", xytext=(-10,-20), ha='center', color='red', fontsize=14)
axs[0].set_title(r'$\frac{K_1}{\alpha_{12}} > K_2$ y $K_1 > \frac{K_2}{\alpha_{21}}$')

# Plot the lines with the isocline of N2 above the isocline of N1 for the second subplot
axs[1].plot(N1, i_values, color='blue', label=r'Isoclina de $N_{1}$')
axs[1].plot(N1, j_values, color='red', label=r'Isoclina de $N_{2}$')

# Mark the intersections with the axes for N1 isocline for the second subplot
axs[1].scatter([0, K1], [K1/alpha12, 0], color='red')
axs[1].annotate(r'$K_2$', (0, K1/alpha12), textcoords="offset points", xytext=(-10,-10), ha='center', color='red', fontsize=14)
axs[1].annotate(r'$\frac{K_2}{\alpha_{21}}$', (K1, 0), textcoords="offset points", xytext=(-10,-20), ha='center', color='red', fontsize=14)

# Mark the intersections with the axes for N2 isocline for the second subplot
axs[1].scatter([0, K2/alpha21], [K2, 0], color='blue')
axs[1].annotate(r'$\frac{K_1}{\alpha_{12}}$', (0, K2), textcoords="offset points", xytext=(-10,-10), ha='center', color='blue', fontsize=14)
axs[1].annotate(r'$K_1$', (K2/alpha21, 0), textcoords="offset points", xytext=(-10,-20), ha='center', color='blue', fontsize=14)
axs[1].set_title(r'$\frac{K_1}{\alpha_{12}} < K_2$ y $K_1 < \frac{K_2}{\alpha_{21}}$')


# Plot the lines with the isocline of N1 above the isocline of N2 for the third subplot
axs[2].plot(N1, k_values, color='blue', label=r'Isoclina de $N_{1}$')
axs[2].plot(N1, j_values, color='red', label=r'Isoclina de $N_{2}$')

# Mark the intersections with the axes for N1 isocline for the third subplot
axs[2].scatter([0, K2/alpha_21], [K2, 0], color='blue')
axs[2].annotate(r'$\frac{K_1}{\alpha_{12}}$', (0, K2), textcoords="offset points", xytext=(-10,-10), ha='center', color='blue', fontsize=14)
axs[2].annotate(r'$K_1$', (K2/alpha_21 +1, 0), textcoords="offset points", xytext=(-10,-20), ha='center', color='blue', fontsize=16)

# Mark the intersections with the axes for N2 isocline for the third subplot
axs[2].scatter([0, K1], [K1/alpha12, 0], color='red')
axs[2].annotate(r'$K_2$', (0, K1/alpha12), textcoords="offset points", xytext=(-10,-10), ha='center', color='red', fontsize=14)
axs[2].annotate(r'$\frac{K_2}{\alpha_{21}}$', (K1, 0), textcoords="offset points", xytext=(-10,-20), ha='center', color='red', fontsize=14)
axs[2].set_title(r'$\frac{K_1}{\alpha_{12}} < K_2$ y $K_1 > \frac{K_2}{\alpha_{21}}$')


# Plot the lines with the isocline of N1 above the isocline of N2 for the fourth subplot
axs[3].plot(N1, j_values, color='blue', label=r'Isoclina de $N_{1}$')
axs[3].plot(N1, k_values, color='red', label=r'Isoclina de $N_{2}$')

# Mark the intersections with the axes for N1 isocline for the fourth subplot
axs[3].scatter([0, K1], [K1/alpha12, 0], color='blue')
axs[3].annotate(r'$\frac{K_1}{\alpha_{12}}$', (0, K1/alpha12), textcoords="offset points", xytext=(-10,-10), ha='center', color='blue', fontsize=14)
axs[3].annotate(r'$K_1$', (K1, 0), textcoords="offset points", xytext=(-10,-20), ha='center', color='blue', fontsize=16)

# Mark the intersections with the axes for N2 isocline for the fourth subplot
axs[3].scatter([0, K2/alpha_21], [K2, 0], color='red')
axs[3].annotate(r'$K_2$', (0, K2), textcoords="offset points", xytext=(-10,-10), ha='center', color='red', fontsize=14)
axs[3].annotate(r'$\frac{K_2}{\alpha_{21}}$', (K2/alpha_21 +1, 0), textcoords="offset points", xytext=(-10,-20), ha='center', color='red', fontsize=14)
axs[3].set_title(r'$\frac{K_1}{\alpha_{12}} > K_2$ y $K_1 < \frac{K_2}{\alpha_{21}}$')


# Set common properties for both subplots
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel(r'$N_{1}$')
    ax.set_ylabel(r'$N_{2}$')
    ax.legend()
    ax.grid(True)

plt.subplots_adjust(wspace=0.3, hspace=0.1)
plt.show()