"""
ga.py — Operadores del Algoritmo Genético (AG)

Parte del híbrido AG + GWO para optimización de hiper-parámetros del
MLPClassifier. Proporciona selección por torneo, cruce uniforme,
mutación y la función principal nueva_generacion().

Todas las funciones respetan los rangos definidos en hiperparametros.py.
"""

import random
from wolf import Wolf
from hiperparametros import gen_aleatorio, BOUNDS
from paralelizador import evaluar_poblacion_paralelo


# ---------------------------------------------------------------------------
# Operadores Genéticos
# ---------------------------------------------------------------------------

def torneo_binario(poblacion):
    """
    Selecciona un padre mediante torneo binario: elige dos lobos al
    azar y retorna el de mayor fitness.

    Args:
        poblacion (list[Wolf]): Población actual.

    Returns:
        Wolf: Padre seleccionado.
    """
    i = random.randint(0, len(poblacion) - 1)
    j = random.randint(0, len(poblacion) - 1)
    while j == i:
        j = random.randint(0, len(poblacion) - 1)

    if poblacion[i].fitness > poblacion[j].fitness:
        return poblacion[i]
    return poblacion[j]


def cruce_uniforme(padre1, padre2):
    """
    Cruce uniforme gen por gen con probabilidad 0.5.
    Cada gen del hijo tiene igual probabilidad de heredarse de padre1 o padre2.

    Args:
        padre1 (Wolf): Primer padre.
        padre2 (Wolf): Segundo padre.

    Returns:
        tuple[Wolf, Wolf]: Dos hijos resultado del cruce.
    """
    hijo1 = Wolf()
    hijo2 = Wolf()

    # Lista de todos los genes definidos en el espacio de búsqueda
    nombres_genes = list(BOUNDS.keys())

    for gen in nombres_genes:
        if random.random() < 0.5:
            setattr(hijo1, gen, getattr(padre1, gen))
            setattr(hijo2, gen, getattr(padre2, gen))
        else:
            setattr(hijo1, gen, getattr(padre2, gen))
            setattr(hijo2, gen, getattr(padre1, gen))

    return hijo1, hijo2


def mutar(individuo, prob_mutacion=0.15):
    """
    Aplica mutación gen por gen con probabilidad prob_mutacion.
    Cada gen mutado se reinicializa aleatoriamente dentro de sus rangos
    definidos en hiperparametros.py.

    Args:
        individuo (Wolf): Individuo a mutar (se modifica in-place).
        prob_mutacion (float): Probabilidad de mutar cada gen.
    """
    for nombre_gen in BOUNDS:
        if random.random() < prob_mutacion:
            nuevo_valor = gen_aleatorio(nombre_gen)
            setattr(individuo, nombre_gen, nuevo_valor)


# ---------------------------------------------------------------------------
# Ciclo principal del AG dentro de una iteración del híbrido
# ---------------------------------------------------------------------------

def nueva_generacion(poblacion, prob_cruce, prob_mutacion,
                     X_train, y_train, X_val, y_val):
    """
    Ejecuta un ciclo completo del Algoritmo Genético:
        1. Conserva élite (Alpha, Beta, Delta).
        2. Genera el resto mediante torneo + cruce + mutación.
        3. Evalúa el fitness de cada nuevo individuo.
        4. Retorna la nueva población ordenada por fitness descendente.

    Args:
        poblacion (list[Wolf]): Población actual (ordenada, índices 0,1,2 = élite).
        prob_cruce (float): Probabilidad de aplicar cruce.
        prob_mutacion (float): Probabilidad de mutación por gen.
        X_train (ndarray): Features de entrenamiento.
        y_train (ndarray): Etiquetas de entrenamiento.
        X_val (ndarray): Features de validación.
        y_val (ndarray): Etiquetas de validación.

    Returns:
        list[Wolf]: Nueva población ordenada por fitness descendente.
    """
    N = len(poblacion)
    nueva_poblacion = []

    # --- Elitismo: Alpha, Beta y Delta pasan como copias independientes ---
    nueva_poblacion.extend([w.copy() for w in poblacion[:3]])

    # --- Generar el resto de la población ---
    hijos_generados = []
    while len(nueva_poblacion) + len(hijos_generados) < N:
        # Selección
        padre1 = torneo_binario(poblacion)
        padre2 = torneo_binario(poblacion)

        # Cruce
        if random.random() < prob_cruce:
            hijo1, hijo2 = cruce_uniforme(padre1, padre2)
        else:
            hijo1 = padre1.copy()
            hijo2 = padre2.copy()

        # Mutación
        mutar(hijo1, prob_mutacion)
        mutar(hijo2, prob_mutacion)

        hijos_generados.append(hijo1)
        if len(nueva_poblacion) + len(hijos_generados) < N:
            hijos_generados.append(hijo2)

    # Evaluación en paralelo
    evaluar_poblacion_paralelo(hijos_generados, X_train, y_train, X_val, y_val)
    
    nueva_poblacion.extend(hijos_generados)

    # --- Ordenar y retornar ---
    nueva_poblacion.sort(key=lambda w: w.fitness, reverse=True)
    return nueva_poblacion