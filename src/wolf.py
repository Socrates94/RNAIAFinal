import numpy as np

class Wolf:
    """Representa un lobo en el algoritmo GWO."""

    def __init__(self, dim, lb, ub):
        """
        Inicializa un lobo con posición aleatoria.
        
        Args:
            dim: Dimensiones del problema.
            lb: Límite inferior.
            ub: Límite superior.
        """
        self.dim = dim
        self.position = np.random.uniform(0, 1, dim) * (ub - lb) + lb
        self.fitness = float("inf")
        self.role = "omega"
        
    def evaluate(self, obj_func):
        """Calcula el fitness del lobo."""
        self.fitness = obj_func(self.position)
        return self.fitness
        
    def clip_bounds(self, lb, ub):
        """Asegura que la posición esté dentro de los límites."""
        self.position = np.clip(self.position, lb, ub)
        
    def update_position(self, alpha_pos, beta_pos, delta_pos, a):
        """Actualiza la posición basada en los líderes Alpha, Beta y Delta."""
        r1_alpha = np.random.random(self.dim)
        r2_alpha = np.random.random(self.dim)
        A1 = 2 * a * r1_alpha - a
        C1 = 2 * r2_alpha
        D_alpha = np.abs(C1 * alpha_pos - self.position)
        X1 = alpha_pos - A1 * D_alpha
        
        r1_beta = np.random.random(self.dim)
        r2_beta = np.random.random(self.dim)
        A2 = 2 * a * r1_beta - a
        C2 = 2 * r2_beta
        D_beta = np.abs(C2 * beta_pos - self.position)
        X2 = beta_pos - A2 * D_beta
        
        r1_delta = np.random.random(self.dim)
        r2_delta = np.random.random(self.dim)
        A3 = 2 * a * r1_delta - a
        C3 = 2 * r2_delta
        D_delta = np.abs(C3 * delta_pos - self.position)
        X3 = delta_pos - A3 * D_delta
        
        self.position = (X1 + X2 + X3) / 3