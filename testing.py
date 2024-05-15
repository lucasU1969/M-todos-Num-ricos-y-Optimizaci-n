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

def runge_kutta_ode_sys(h:float, odes:list, initial_conditions:list[float, float], t0:float, t_max:float):
    approximations = [initial_conditions] ## [[N1_0, N2_0]]
    t = t0
    while t < t_max:
        k1 = [h * ode(t, approximations[-1][0], approximations[-1][1]) for ode in odes]
        k2 = [h * ode(t + h/2, approximations[-1][0] + k1[0]/2, approximations[-1][1] + k1[1]/2) for ode in odes]
        k3 = [h * ode(t + h/2, approximations[-1][0] + k2[0]/2, approximations[-1][1] + k2[1]/2) for ode in odes]
        k4 = [h * ode(t + h, approximations[-1][0] + k3[0], approximations[-1][1] + k3[1]) for ode in odes]
        next_approximation = [approximations[-1][i] + (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6 for i in range(2)]
        approximations.append(next_approximation)
        t += h
    return np.array(approximations)


def plot_solutions_grid(h, odes, initial_conditions_grid, t0, t_max, subplt:list[int,int], ax):

    for initial_conditions in initial_conditions_grid:
        solutions = runge_kutta_ode_sys(h, odes, initial_conditions, t0, t_max)
        ax[subplt[0], subplt[1]].plot(solutions[:, 0], solutions[:, 1], color='green')
        # Agregar flechas para indicar la dirección
        for i in range(0, len(solutions) - 1, max(1, len(solutions) // 2)):
            ax[subplt[0], subplt[1]].arrow(solutions[i, 0], solutions[i, 1], 
                      solutions[i+1, 0] - solutions[i, 0], 
                      solutions[i+1, 1] - solutions[i, 1], 
                      head_width= 15, head_length=20, color='green')

    plt.xlabel('N1')
    plt.ylabel('N2')
    plt.title('Solutions of the ODE system')
    plt.grid(True)

# Define the ODE system with parameters
def dN1dt(N1, N2, r1, K1, alpha12):
    return r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)

def dN2dt(N1, N2, r2, K2, alpha21):
    return r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)


# Create a figure with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(20, 16))

# List of cases
cases = [
    {
        'r1': 0.1, 'K1': 1200, 'alpha12': 2.7, 'r2': 0.1, 'K2': 400, 'alpha21': 0.5,
        'N1_values': np.linspace(0, 1400, 6), 'N2_values': np.linspace(0, 600, 10),
        'ylim': (0, 600), 'xlim': (0, 1400), 'equilibrium_points': [(0, 0), (1200, 0), (0, 400)],
        'stable_points': [ (1200, 0), (0, 400)],
        'unstable_points': [(0, 0)]
    },
    # Add other cases here...
    {
        'r1': 0.1, 'K1': 500, 'alpha12': 0.5, 'r2': 0.1, 'K2': 1200, 'alpha21': 2,
        'N1_values': np.linspace(0, 800, 10), 'N2_values': np.linspace(0, 1400, 8),
        'ylim': (0, 1400), 'xlim': (0, 800), 'equilibrium_points': [(0, 0), (500, 0), (0, 1200)],
        'stable_points': [ (500, 0), (0, 1200)],
        'unstable_points': [(0, 0)]
    },

    {
        'r1': 0.1, 'K1': 1200, 'alpha12': 2.7, 'r2': 0.1, 'K2': 800, 'alpha21': 1.5,
        'N1_values': np.linspace(0, 1400, 8), 'N2_values': np.linspace(0, 800, 10),
        'ylim': (0, 800), 'xlim': (0, 1400), 'equilibrium_points': [(0, 0), (1200, 0), (0, 800), (19200/61, 20000/61)],
        'stable_points': [ (1200, 0), (0, 800)],
        'unstable_points': [(0, 0), (19200/61, 20000/61)]
    },

    {
        'r1': 0.1, 'K1': 1500, 'alpha12': 1.5, 'r2': 0.1, 'K2': 800, 'alpha21': 0.4,
        'N1_values': np.linspace(0, 2100, 8), 'N2_values': np.linspace(0, 1200, 10),
        'ylim': (0, 1200), 'xlim': (0, 2100), 'equilibrium_points': [(0, 0), (1500, 0), (0, 800), (750, 500)],
        'stable_points': [(0, 0), (1500, 0), (0, 800), (750, 500)],
        'unstable_points': []

    }
]

for i, case in enumerate(cases):
    # Calculate row and column index for the subplot
    row = i // 2
    col = i % 2

    # Set the current axes to the subplot
    plt.sca(axs[row, col])

    # Run the existing code for each case
    # ...
    # You need to replace the # ... comment with the existing code for each case, and add the other cases to the cases list. Make sure to replace all plt. calls with axs[row, col]. (for example, plt.plot() becomes axs[row, col].plot()), except for plt.sca(), plt.figure(), and plt.show().

    # Parameters
    r1 = case['r1']
    K1 = case['K1']
    alpha12 = case['alpha12']
    r2 = case['r2']
    K2 = case['K2']
    alpha21 = case['alpha21']
    params = (r1, K1, alpha12, r2, K2, alpha21)

    # Create a meshgrid of initial conditions
    N1_values = case['N1_values']
    N2_values = case['N2_values']
    N1_grid, N2_grid = np.meshgrid(N1_values, N2_values)
    initial_conditions_grid = np.column_stack([N1_grid.ravel(), N2_grid.ravel()])
    # Other parameters for Runge-Kutta
    h = 0.005
    t0 = 0
    t_max = 50
    odes = [lambda t, N1, N2: r1 * N1 * (1 - N1 / K1 - alpha12 * N2 / K1),
            lambda t, N1, N2: r2 * N2 * (1 - N2 / K2 - alpha21 * N1 / K2)]
    
    # Plot the solutions
    plot_solutions_grid(h, odes, initial_conditions_grid, t0, t_max, (row, col), axs)

    iso_N1 = np.linspace(0, K1, 100)

    axs[row, col].plot(iso_N1, N1_isocline(iso_N1, K1, alpha12), linestyle='-', color='pink', label='dN1/dt = 0')
    axs[row, col].plot(iso_N1, N2_isocline(iso_N1, K2, alpha21), linestyle='-', color='lightblue', label='dN2/dt = 0')

    # axs[row,col].scatter([point[0] for point in case['equilibrium_points']], [point[1] for point in case['equilibrium_points']], color='red', label='Puntos de equilibrio')
    # hacer los puntos de equilibrio estables con un circulo relleno y los inestables con un circulo vacio
    axs[row, col].scatter([point[0] for point in case['stable_points']], [point[1] for point in case['stable_points']], color='red', label='Puntos de equilibrio estables', marker='o')
    axs[row, col].scatter([point[0] for point in case['unstable_points']], [point[1] for point in case['unstable_points']], color='red', label='Puntos de equilibrio inestables', marker='o', facecolors='none')

    axs[row, col].legend()
    axs[row, col].set_ylim(case['ylim'])
    axs[row, col].set_xlim(case['xlim'])

# FALTA CLASIFICAR BIEN LOS PUNTOS DE EQUILIBRIO



# Show the figure
plt.show()
# -----------------------------------------------------------------------------------------------------------
# diagramas de fase pero con el campo vectorial para los cuatro casos: 

def plot_vector_field(K1, K2, r1, r2, alpha12, alpha21, N1_range, N2_range, ax):
    N1, N2 = np.meshgrid(N1_range, N2_range)
    dN1 = dN1dt(N1, N2, r1, K1, alpha12)
    dN2 = dN2dt(N1, N2, r2, K2, alpha21)
    dN1 = dN1 / np.sqrt(dN1**2 + dN2**2)
    dN2 = dN2 / np.sqrt(dN1**2 + dN2**2)
    ax.quiver(N1, N2, dN1, dN2, scale=20, color='lightgreen')
    ax.set_xlabel('N1')
    ax.set_ylabel('N2')
    ax.set_title('Phase diagram with isoclines')
    ax.legend()
    ax.set_ylim(0, 600)
    ax.set_xlim(0, 1500)

# Create a figure with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(20, 16))

for i, case in enumerate(cases):
    # Calculate row and column index for the subplot
    row = i // 2
    col = i % 2

    # Set the current axes to the subplot
    plt.sca(axs[row, col])

    # Run the existing code for each case
    # ...
    # You need to replace the # ... comment with the existing code for each case, and add the other cases to the cases list. Make sure to replace all plt. calls with axs[row, col]. (for example, plt.plot() becomes axs[row, col].plot()), except for plt.sca(), plt.figure(), and plt.show().

    # Parameters
    r1 = case['r1']
    K1 = case['K1']
    alpha12 = case['alpha12']
    r2 = case['r2']
    K2 = case['K2']
    alpha21 = case['alpha21']
    params = (r1, K1, alpha12, r2, K2, alpha21)

    # Create a meshgrid of initial conditions
    N1_values = case['N1_values']
    N2_values = case['N2_values']
    N1_grid, N2_grid = np.meshgrid(N1_values, N2_values)
    initial_conditions_grid = np.column_stack([N1_grid.ravel(), N2_grid.ravel()])
    # Other parameters for Runge-Kutta
    h = 0.005
    t0 = 0
    t_max = 50
    odes = [lambda t, N1, N2: r1 * N1 * (1 - N1 / K1 - alpha12 * N2 / K1),
            lambda t, N1, N2: r2 * N2 * (1 - N2 / K2 - alpha21 * N1 / K2)]
    
    # Plot the solutions
    plot_solutions_grid(h, odes, initial_conditions_grid, t0, t_max, (row, col), axs)

    iso_N1 = np.linspace(0, K1, 100)

    axs[row, col].plot(iso_N1, N1_isocline(iso_N1, K1, alpha12), linestyle='-', color='pink', label='dN1/dt = 0')
    axs[row, col].plot(iso_N1, N2_isocline(iso_N1, K2, alpha21), linestyle='-', color='lightblue', label='dN2/dt = 0')

    # axs[row,col].scatter([point[0] for point in case['equilibrium_points']], [point[1] for point in case['equilibrium_points']], color='red', label='Puntos de equilibrio')
    # hacer los puntos de equilibrio estables con un circulo relleno y los inestables con un circulo vacio
    axs[row, col].scatter([point[0] for point in case['stable_points']], [point[1] for point in case['stable_points']], color='red', label='Puntos de equilibrio estables', marker='o')
    axs[row, col].scatter([point[0] for point in case['unstable_points']], [point[1] for point in case['unstable_points']], color='red', label='Puntos de equilibrio inestables', marker='o', facecolors='none')

    plot_vector_field(K1, K2, r1, r2, alpha12, alpha21, np.linspace(0, 2100, 20), np.linspace(0, 1400, 20), axs[row, col])

    axs[row, col].legend()
    axs[row, col].set_ylim(case['ylim'])
    axs[row, col].set_xlim(case['xlim'])

# Show the figure
plt.show()


#------------------------------------------------------------------------------------------------------------


# GRÁFICOS QUE FALTAN DEL EJERCICIO 2
#  - variando todos los parámetros. 
