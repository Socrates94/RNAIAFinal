"""
paralelizador.py — Módulo para evaluación concurrente de individuos

Este módulo encapsula la lógica para ejecutar las evaluaciones de fitness
(entrenamiento del MLP) de varios lobos al mismo tiempo utilizando todos
los núcleos disponibles del procesador.
"""
from joblib import Parallel, delayed
from mlp_trainer import evaluar_fitness

def evaluar_poblacion_paralelo(lobos, X_train, y_train, X_val, y_val, n_jobs=-1):
    """
    Toma una lista de lobos y evalúa su fitness en paralelo.
    
    Args:
        lobos (list[Wolf]): Lista de lobos que necesitan evaluación.
        X_train, y_train, X_val, y_val: Datos de entrenamiento y validación.
        n_jobs (int): Número de núcleos a usar. -1 significa todos los disponibles.
        
    Returns:
        list[Wolf]: La misma lista de lobos, pero con el atributo .fitness ya actualizado.
    """
    # Si la lista está vacía, no hacemos nada
    if not lobos:
        return lobos
        
    # joblib.Parallel crea un 'Pool' de procesos trabajadores.
    # delayed(evaluar_fitness) le dice a joblib qué función ejecutar.
    # Se ejecuta evaluar_fitness para cada lobo en la lista al mismo tiempo.
    resultados_fitness = Parallel(n_jobs=n_jobs)(
        delayed(evaluar_fitness)(lobo, X_train, y_train, X_val, y_val) 
        for lobo in lobos
    )
    
    # Asignamos los resultados devueltos a los objetos originales
    for lobo, fitness in zip(lobos, resultados_fitness):
        lobo.fitness = fitness
        
    return lobos
