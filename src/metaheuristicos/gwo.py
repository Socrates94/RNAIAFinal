"""
gwo.py — Operadores del Grey Wolf Optimizer (GWO)

Parte del híbrido AG + GWO para optimización de hiper-parámetros del
MLPClassifier. Proporciona la función paso_gwo() que mueve los lobos
omega hacia Alpha, Beta y Delta.

Usa update_position_gwo() y clip_bounds() de la clase Wolf.
"""

from paralelizador import evaluar_poblacion_paralelo


def paso_gwo(poblacion, a, X_train, y_train, X_val, y_val):
    """
    Ejecuta un ciclo de caza del GWO sobre la población:
        1. Identifica a Alpha (índice 0), Beta (1) y Delta (2).
        2. Mueve SOLO a los omegas (índices 3..N-1) hacia ellos.
        3. Re-evalúa el fitness de cada omega movido.
        4. Retorna la población actualizada (sin reordenar aún).

    Args:
        poblacion (list[Wolf]): Población actual, ordenada por fitness.
        a (float): Coeficiente de exploración/explotación (2 → 0).
        X_train (ndarray): Features de entrenamiento.
        y_train (ndarray): Etiquetas de entrenamiento.
        X_val (ndarray): Features de validación.
        y_val (ndarray): Etiquetas de validación.

    Returns:
        list[Wolf]: Misma población con los omegas actualizados.
    """
    N = len(poblacion)

    # Si la población es muy pequeña, no hay omegas que mover
    if N < 4:
        return poblacion

    # Identificar líderes
    alpha = poblacion[0]
    beta  = poblacion[1]
    delta = poblacion[2]

    omegas_movidos = []
    # Mover solo los omegas (índices 3 en adelante)
    for i in range(3, N):
        omega = poblacion[i]

        # Mover hacia los líderes usando las ecuaciones GWO
        omega.update_position_gwo(
            alpha_pos=alpha,
            beta_pos=beta,
            delta_pos=delta,
            a=a
        )

        # Ajustar a límites (redundante pero seguro: update_position_gwo ya lo hace)
        omega.clip_bounds()
        
        omegas_movidos.append(omega)

    # Re-evaluar fitness de todos los omegas en paralelo
    evaluar_poblacion_paralelo(omegas_movidos, X_train, y_train, X_val, y_val)

    return poblacion
