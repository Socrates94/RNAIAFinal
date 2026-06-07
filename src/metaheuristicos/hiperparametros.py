"""
Define el espacio de búsqueda de hiper-parámetros del MLPClassifier.
Es la única fuente de verdad para rangos y decodificación.
Todos los metaheurísticos y mlp_trainer deben importar desde aquí.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Espacio de búsqueda
# ---------------------------------------------------------------------------

BOUNDS = {
    # Continuos en escala logarítmica
    "alpha": {
        "tipo": "log_continuo",
        "min": 1e-4,
        "max": 1e-1,
    },
    "learning_rate_init": {
        "tipo": "log_continuo",
        "min": 1e-4,
        "max": 1e-1,
    },
    # Enteros
    "n_neuronas_capa1": {
        "tipo": "entero",
        "min": 10,
        "max": 200,
    },
    "n_neuronas_capa2": {
        "tipo": "entero",
        "min": 0,      # 0 = sin segunda capa oculta
        "max": 200,
    },
    "max_iter": {
        "tipo": "entero",
        "min": 100,
        "max": 500,
    },
    # Categóricos (representados como índices enteros internamente)
    "batch_size": {
        "tipo": "categorico",
        "opciones": [16, 32, 64, 128, 256],
    },
    "activacion": {
        "tipo": "categorico",
        "opciones": [0, 1, 2],   # 0=relu  1=tanh  2=logistic
    },
    "solver": {
        "tipo": "categorico",
        "opciones": [0, 1],      # 0=adam  1=sgd
    },
}

# Mapeos de índice entero a string que entiende sklearn
MAPA_ACTIVACION = {0: "relu", 1: "tanh", 2: "logistic"}
MAPA_SOLVER     = {0: "adam", 1: "sgd"}

# ---------------------------------------------------------------------------
# Inicialización aleatoria de un gen
# ---------------------------------------------------------------------------

def gen_aleatorio(nombre):
    """
    Genera un valor aleatorio válido para el gen indicado,
    respetando el tipo y rango definidos en BOUNDS.

    Args:
        nombre (str): Clave del gen en BOUNDS.

    Returns:
        float | int: Valor aleatorio dentro del rango permitido.
    """
    spec = BOUNDS[nombre]
    tipo = spec["tipo"]

    if tipo == "log_continuo":
        log_min = np.log10(spec["min"])
        log_max = np.log10(spec["max"])
        return 10 ** np.random.uniform(log_min, log_max)

    if tipo == "entero":
        return np.random.randint(spec["min"], spec["max"] + 1)

    if tipo == "categorico":
        return np.random.choice(spec["opciones"])

    raise ValueError(f"Tipo de gen desconocido: {tipo}")


def clip_gen(nombre, valor):
    """
    Ajusta un valor al rango válido de su gen.
    Útil tras la actualización de posición en GWO.

    Args:
        nombre (str): Clave del gen en BOUNDS.
        valor  (float | int): Valor a ajustar.

    Returns:
        float | int: Valor ajustado dentro del rango.
    """
    spec = BOUNDS[nombre]
    tipo = spec["tipo"]

    if tipo == "log_continuo":
        return float(np.clip(valor, spec["min"], spec["max"]))

    if tipo == "entero":
        return int(np.clip(round(valor), spec["min"], spec["max"]))

    if tipo == "categorico":
        opciones = np.array(spec["opciones"])
        # Snap al valor de opciones más cercano
        return int(opciones[np.argmin(np.abs(opciones - valor))])

    raise ValueError(f"Tipo de gen desconocido: {tipo}") # Si no se encuentra el tipo de gen, se lanza un error

# ---------------------------------------------------------------------------
# Decodificador: Wolf → dict para MLPClassifier
# ---------------------------------------------------------------------------

def decodificar(wolf):
    """
    Convierte los genes de un lobo a un diccionario listo para
    pasarse directamente a MLPClassifier de sklearn.

    Args:
        wolf: Instancia de Wolf con los atributos de genes.

    Returns:
        dict: Hiper-parámetros para MLPClassifier.
    """
    if wolf.n_neuronas_capa2 == 0:
        capas = (wolf.n_neuronas_capa1,)
    else:
        capas = (wolf.n_neuronas_capa1, wolf.n_neuronas_capa2)

    return {
        "hidden_layer_sizes":  capas,
        "activation":          MAPA_ACTIVACION[wolf.activacion],
        "solver":              MAPA_SOLVER[wolf.solver],
        "alpha":               float(wolf.alpha),
        "learning_rate_init":  float(wolf.learning_rate_init),
        "batch_size":          int(wolf.batch_size),
        "max_iter":            int(wolf.max_iter),
        "early_stopping":      False,
        "random_state":        42,
        "verbose":             False,
    }
