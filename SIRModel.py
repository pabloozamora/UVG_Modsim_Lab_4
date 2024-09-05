import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation
from matplotlib import colors

class SIRModel:
    def __init__(self, M, N, T, I0, rad, beta, gamma):
        self.M = M
        self.N = N
        self.T = T
        self.I0 = I0
        self.rad = rad
        self.beta = beta
        self.gamma = gamma
        self.grid = np.zeros((M, N), dtype=int)
        self.grid_history = []  # Historial del grid
        self.population_counts = []  # Historial de S, I, R
    
    # Función para inicializar el grid con celdas infectadas aleatoriamente
    def initialize_grid(self):
        grid = np.zeros((self.M, self.N), dtype=int)  # Todas las celdas son susceptibles inicialmente (0)
        
        # Elegir I0 celdas aleatorias para ser infectadas
        infected_cells = random.sample([(i, j) for i in range(self.M) for j in range(self.N)], self.I0)
        for cell in infected_cells:
            grid[cell] = 1  # Infectadas (estado 1)
        
        return grid

    # Función para obtener la vecindad de una celda (i, j) con radio rad
    def get_neighborhood(self, grid, i, j):
        M, N = grid.shape
        neighborhood = grid[max(0, i-self.rad):min(M, i+rad+1), max(0, j-self.rad):min(N, j+self.rad+1)]
        return neighborhood

    # Función para ejecutar la simulación
    def simulate_sir(self):
        grid = self.initialize_grid()  # Inicializar el grid con infecciones aleatorias
        
        for t in range(self.T):
            new_grid = grid.copy()  # Hacer una copia del grid actual para la próxima iteración
            S_count = 0
            I_count = 0
            R_count = 0
            
            for i in range(self.M):
                for j in range(self.N):
                    if grid[i, j] == 0:  # Susceptible
                        neighborhood = self.get_neighborhood(grid, i, j)  # Pasar el grid actual
                        infected_neighbors = np.sum(neighborhood == 1)
                        total_neighbors = neighborhood.size
                        
                        # Probabilidad de infección
                        if infected_neighbors > 0:
                            infection_probability = self.beta * (infected_neighbors / total_neighbors)
                            if random.random() < infection_probability:
                                new_grid[i, j] = 1  # Infectar la celda
                    
                    elif grid[i, j] == 1:  # Infectado
                        # Probabilidad de recuperación
                        if random.random() < self.gamma:
                            new_grid[i, j] = 2  # Recuperar la celda
            
            # Actualizar el grid y registrar las cantidades de S, I, R
            grid = new_grid  # Actualizar el grid con el nuevo estado
            
            # Contar los susceptibles, infectados y recuperados
            S_count = np.sum(grid == 0)
            I_count = np.sum(grid == 1)
            R_count = np.sum(grid == 2)
            
            # Almacenar el historial del grid y de los conteos
            self.grid_history.append(grid.copy())
            self.population_counts.append((S_count, I_count, R_count))

    # Función para graficar el historial de S, I, R
    def plot_sir_history(self):
        population_counts = np.array(self.population_counts)
        time = np.arange(self.T)
        
        plt.figure(figsize=(10, 6))
        plt.plot(time, population_counts[:, 0], label="Susceptibles (S)", color='blue')
        plt.plot(time, population_counts[:, 1], label="Infectados (I)", color='red')
        plt.plot(time, population_counts[:, 2], label="Recuperados (R)", color='green')
        plt.xlabel("Tiempo")
        plt.ylabel("Cantidad de individuos")
        plt.title("Evolución de las poblaciones S, I, R")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    # Función para graficar la simulación
    def animate_grid(self):
        fig, ax = plt.subplots()
        img = ax.imshow(self.grid_history[0], cmap='viridis', vmin=0, vmax=2)

        def update(frame):
            img.set_data(self.grid_history[frame])
            ax.set_title(f'Día {frame}')
            return [img]

        ani = animation.FuncAnimation(fig, update, frames=len(self.grid_history), blit=False)
        return ani
        
# Parámetros del modelo

M = 50  # Tamaño del grid en filas
N = 50  # Tamaño del grid en columnas
T = 100  # Tiempo de simulación
I0 = 10  # Número inicial de infectados
rad = 1  # Radio de interacción
beta = 0.7  # Probabilidad de infección
gamma = 0.1  # Probabilidad de recuperación

# Crear el modelo y ejecutar la simulación

model = SIRModel(M, N, T, I0, rad, beta, gamma)
model.simulate_sir()
ani = model.animate_grid()
plt.show()
