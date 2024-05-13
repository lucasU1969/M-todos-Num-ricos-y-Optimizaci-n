import numpy as np
import matplotlib.pyplot as plt

# Define the ODE system
def dN1dt(N1, N2, r1, K1, alpha12):
    return r1*N1*(1 - (N1 + alpha12*N2)/K1)

def dN2dt(N1, N2, r2, K2, alpha21):
    return r2*N2*(1 - (N2 + alpha21*N1)/K2)

# Define the isoclines
def N1_isocline(N1, K1, alpha12):
    return (-N1 + K1) / alpha12

def N2_isocline(N1, K2, alpha21):
    return K2 - (alpha21 * N1)

def plot_phase_diagram(r1, K1, alpha12, r2, K2, alpha21, N1_range, N2_range):
    # Define the grid
    N1, N2 = np.meshgrid(N1_range, N2_range)

    # Calculate the derivatives
    dN1 = dN1dt(N1, N2, r1, K1, alpha12)
    dN2 = dN2dt(N1, N2, r2, K2, alpha21)

    # Plot the vector field
    plt.streamplot(N1, N2, dN1, dN2, color='k')

    # Plot the isoclines
    N1_values = np.linspace(0, K1, 100)
    plt.plot(N1_values, N1_isocline(N1_values, K1, alpha12), label='dN1/dt = 0')
    plt.plot(N1_values, N2_isocline(N1_values, K2, alpha21), label='dN2/dt = 0')

    # Add intersection between isoclines (equilibrium point)
    # plt.scatter( (K1 - alpha12*K2)/(1 - alpha12*alpha21), K2 - alpha21*(K1 - alpha12*K2)/(1 - alpha12*alpha21), color='red', label='Punto de equilibrio')

    plt.xlabel('N1')
    plt.ylabel('N2')
    plt.title('Phase diagram with isoclines')
    plt.legend()
    plt.show()

def plot_dN1dt_N1(N1, N2, r1, K1, alpha12):
    plt.plot(N1, dN1dt(N1, N2, r1, K1, alpha12))
    plt.xlabel('N1')
    plt.ylabel('dN1/dt')
    plt.title('dN1/dt vs N1')
    plt.show()


# First the blue isocline is above the orange one and they instersect in the first quadrant
plot_phase_diagram(0.1, 100, 1, 0.1, 20, -1, np.linspace(0, 120, 100), np.linspace(0, 120, 100))

# First the orange isocline is above the blue one and they instersect in the first quadrant
plot_phase_diagram(0.1, 70, 0.875, 0.1, 100, 1, np.linspace(0, 120, 100), np.linspace(0, 120, 100))

# # The isoclines does not intersect in the first quadrant but the blue is above
plot_phase_diagram(0.1, 80, 1, 0.1, 100, 1, np.linspace(0, 120, 100), np.linspace(0, 120, 100))

# # The isoclines does not intersect in the first quadrant but the orange is above
plot_phase_diagram(0.1, 100, 1, 0.1, 70, 1, np.linspace(0, 120, 100), np.linspace(0, 120, 100))

# plot_dN1dt_N1(np.linspace(0, 100, 100), 10, 0.1, 0.3, 0.5)