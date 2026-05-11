from conexion_db import get_connection
import numpy as np
import matplotlib.pyplot as plt
from wolf import Wolf

def objective_function(x):
    """Calcula la función esfera para un vector x."""
    return np.sum(x**2)

def gwo(obj_func, lb, ub, dim, search_agents_no, max_iter):
    """
    Implementación del algoritmo Grey Wolf Optimizer (GWO).
    
    Args:
        obj_func: Función objetivo a minimizar.
        lb: Límite inferior.
        ub: Límite superior.
        dim: Dimensiones del problema.
        search_agents_no: Número de agentes de búsqueda.
        max_iter: Iteraciones máximas.
    """
    pack = [Wolf(dim, lb, ub) for _ in range(search_agents_no)]
    
    alpha = Wolf(dim, lb, ub)
    alpha.fitness = float("inf")
    alpha.role = "alpha"
    
    beta = Wolf(dim, lb, ub)
    beta.fitness = float("inf")
    beta.role = "beta"
    
    delta = Wolf(dim, lb, ub)
    delta.fitness = float("inf")
    delta.role = "delta"
    
    convergence_curve = np.zeros(max_iter)
    
    for l in range(max_iter):
        for wolf in pack:
            wolf.clip_bounds(lb, ub)
            fitness = wolf.evaluate(obj_func)
            
            if fitness < alpha.fitness:
                delta.fitness, delta.position = beta.fitness, beta.position.copy()
                beta.fitness, beta.position = alpha.fitness, alpha.position.copy()
                alpha.fitness, alpha.position = fitness, wolf.position.copy()
                
            elif fitness < beta.fitness and fitness > alpha.fitness:
                delta.fitness, delta.position = beta.fitness, beta.position.copy()
                beta.fitness, beta.position = fitness, wolf.position.copy()
                
            elif fitness < delta.fitness and fitness > beta.fitness and fitness > alpha.fitness:
                delta.fitness, delta.position = fitness, wolf.position.copy()
        
        a = 2 - l * (2 / max_iter)
        
        for wolf in pack:
            wolf.update_position(alpha.position, beta.position, delta.position, a)
                
        convergence_curve[l] = alpha.fitness
        
    return alpha.fitness, alpha.position, convergence_curve

if __name__ == "__main__":
    dim = 10
    search_agents_no = 30
    max_iter = 100
    lb = -100
    ub = 100
    
    best_score, best_pos, convergence = gwo(objective_function, lb, ub, dim, search_agents_no, max_iter)
    
    print(f"[INFO] Fitness: {best_score:.6f}")
    print(f"[INFO] Posicion alfa: {np.round(best_pos, 4)}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(convergence, color='b', linewidth=2)
    plt.title('Curva de Convergencia GWO')
    plt.xlabel('Iteraciones')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.show()

