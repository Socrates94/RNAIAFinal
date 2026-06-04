"""
hybrid_optimizer.py — Orquestador del híbrido AG + GWO

Optimiza los hiper-parámetros del MLPClassifier combinando:
  1. Algoritmo Genético (AG): exploración global (cruce + mutación).
  2. Grey Wolf Optimizer (GWO): explotación local (movimiento hacia líderes).

Retorna los mejores hiper-parámetros encontrados y los historiales
necesarios para el análisis estadístico (Punto 4 del examen).
"""

import os
import warnings
import numpy as np
from wolf import Wolf
from ga import nueva_generacion
from gwo import paso_gwo
from paralelizador import evaluar_poblacion_paralelo


def optimizar(X_train, y_train, X_val, y_val,
              N=10, MAX_ITER=20,
              prob_cruce=0.8, prob_mutacion=0.15,
              verbose=True):
    """
    Ejecuta el optimizador híbrido AG + GWO para encontrar los mejores
    hiper-parámetros del MLPClassifier.

    Args:
        X_train (ndarray): Features de entrenamiento.
        y_train (ndarray): Etiquetas de entrenamiento.
        X_val (ndarray): Features de validación.
        y_val (ndarray): Etiquetas de validación.
        N (int): Tamaño de la población de lobos.
        MAX_ITER (int): Número máximo de iteraciones.
        prob_cruce (float): Probabilidad de cruce en el AG.
        prob_mutacion (float): Probabilidad de mutación por gen.
        verbose (bool): Si True, imprime progreso por iteración.

    Returns:
        tuple:
            - mejor_wolf (Wolf): Mejor lobo encontrado (Alpha final).
            - historial_mejor (list[float]): Mejor fitness por iteración.
            - historial_poblacion (list[list[float]]): Fitness de toda la
              población por iteración (para tendencia central).
            - historial_a (list[float]): Coeficiente 'a' por iteración.
    """
    # ------------------------------------------------------------------
    # FASE 1: Inicialización
    # ------------------------------------------------------------------
    if verbose:
        print("=" * 60)
        print("[HÍBRIDO AG+GWO] Inicializando optimización")
        print(f"  Población: {N} lobos")
        print(f"  Iteraciones máximas: {MAX_ITER}")
        print(f"  Prob. cruce: {prob_cruce}")
        print(f"  Prob. mutación: {prob_mutacion}")
        print("=" * 60)

    # ConvergenceWarning es esperado: algunos lobos tendrán max_iter bajo
    warnings.filterwarnings("ignore", category=UserWarning)

    # Crear población inicial aleatoria
    poblacion = [Wolf() for _ in range(N)]

    # Evaluar fitness de toda la población inicial en paralelo
    evaluar_poblacion_paralelo(poblacion, X_train, y_train, X_val, y_val)

    # Ordenar por fitness descendente (mejor primero)
    poblacion.sort(key=lambda w: w.fitness, reverse=True)

    # Asignar roles iniciales
    alpha = poblacion[0]
    beta  = poblacion[1]
    delta = poblacion[2]
    alpha.role, beta.role, delta.role = "alpha", "beta", "delta"

    # Inicializar historiales
    historial_mejor = []
    historial_poblacion = []
    historial_a = []

    if verbose:
        print(f"\n[ITER 00] Mejor fitness inicial: {alpha.fitness:.4f}")
        print(f"  Alpha: {alpha.to_dict()}")

    # ------------------------------------------------------------------
    # FASE 2: Ciclo principal híbrido
    # ------------------------------------------------------------------
    for t in range(1, MAX_ITER + 1):
        # ──────────────────────────────────────────────────────────────
        # Actualizar coeficiente 'a' al inicio de cada iteración
        # ──────────────────────────────────────────────────────────────
        a = 2.0 - 2.0 * (t / MAX_ITER)  # Decrece linealmente de 2 a 0
        # ──────────────────────────────────────────────────────────────
        # BLOQUE AG: Exploración global
        # ──────────────────────────────────────────────────────────────
        poblacion = nueva_generacion(
            poblacion,
            prob_cruce=prob_cruce,
            prob_mutacion=prob_mutacion,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )
        # nueva_generacion() ya retorna la población ordenada

        # BLOQUE GWO: Explotación local
        # ──────────────────────────────────────────────────────────────

        poblacion = paso_gwo(
            poblacion,
            a=a,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )

        # ──────────────────────────────────────────────────────────────
        # Actualizar jerarquía
        # ──────────────────────────────────────────────────────────────
        poblacion.sort(key=lambda w: w.fitness, reverse=True)
        alpha = poblacion[0]
        beta  = poblacion[1]
        delta = poblacion[2]
        alpha.role, beta.role, delta.role = "alpha", "beta", "delta"

        # ──────────────────────────────────────────────────────────────
        # Registrar estadísticas (Punto 4 del examen)
        # ──────────────────────────────────────────────────────────────
        fitness_actual = [lobo.fitness for lobo in poblacion]

        mejor_fitness = alpha.fitness
        media_fitness = np.mean(fitness_actual)
        mediana_fitness = np.median(fitness_actual)
        desviacion_fitness = np.std(fitness_actual)

        historial_mejor.append(mejor_fitness)
        historial_poblacion.append(fitness_actual)
        historial_a.append(a)

        if verbose:
            print(f"[ITER {t:02d}] Mejor: {mejor_fitness:.4f}  "
                  f"Media: {media_fitness:.4f}  "
                  f"Mediana: {mediana_fitness:.4f}  "
                  f"Desv: {desviacion_fitness:.4f}  "
                  f"a: {a:.3f}")

    # ------------------------------------------------------------------
    # FASE 3: Resultado final
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 60)
        print("[HÍBRIDO AG+GWO] Optimización finalizada")
        print(f"  Mejor fitness: {alpha.fitness:.4f}")
        print(f"  Mejores hiper-parámetros: {alpha.to_dict()}")
        print("=" * 60)

    return alpha.to_dict(), historial_mejor, historial_poblacion, historial_a
