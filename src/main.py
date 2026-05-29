import os
import sys

# Asegura que los imports relativos funcionen desde cualquier CWD
_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.join(_SRC, "metaheuristicos"))

from analisis_exploratorio import (
    cargar_datos,
    informacion_general,
    distribucion_clases,
    histogramas_features,
    heatmap_correlacion,
    boxplots_por_clase,
    RUTA_CSV,
)
from preprocessing import preprocesar
from split_data import split_data
from mlp_trainer import entrenar, evaluar_final, graficar_perdida
from metrics import evaluar, graficar_confusion
from visualizer import (
    graficar_evolucion_fitness,
    graficar_tendencia_central,
    graficar_coeficiente_a,
    RUTA_SALIDA,
)

# ---------------------------------------------------------------------------
# Configuracion global — activar/desactivar etapas individualmente
# ---------------------------------------------------------------------------

ETAPAS = {
    "eda":           False,  # Correr aparte con: python src/analisis_exploratorio.py
    "preprocessing": True,
    "split":         True,
    "optimizacion":  True,   # Usa hiperparametros por defecto mientras no hay metaheuristicos
    "entrenamiento": True,
    "evaluacion":    True,
}


# ---------------------------------------------------------------------------
# Etapas del pipeline
# ---------------------------------------------------------------------------

def etapa_eda(df):
    print("\n" + "=" * 60)
    print("ETAPA 1 — Analisis Exploratorio de Datos (EDA)")
    print("=" * 60)

    informacion_general(df)
    distribucion_clases(df)
    histogramas_features(df)
    heatmap_correlacion(df)
    boxplots_por_clase(df)

    print("\n[INFO] EDA completado.")


def etapa_preprocessing(df):
    print("\n" + "=" * 60)
    print("ETAPA 2 — Preprocesamiento")
    print("=" * 60)

    X, y, scaler, le = preprocesar(df)

    print("[INFO] Preprocesamiento completado.")
    return X, y, scaler, le


def etapa_split(X, y):
    print("\n" + "=" * 60)
    print("ETAPA 3 — Division de datos (70 / 15 / 15)")
    print("=" * 60)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print("[INFO] Split completado.")
    return X_train, X_val, X_test, y_train, y_val, y_test


def etapa_optimizacion(X_train, y_train, X_val, y_val):
    """
    Stub — se implementa cuando metaheuristicos/ este listo.
    Retorna los mejores hiperparametros y los historiales de fitness
    para las graficas de convergencia.
    """
    print("\n" + "=" * 60)
    print("ETAPA 4 — Optimizacion hibrida GWO-GA (PENDIENTE)")
    print("=" * 60)
    print("[WARN] Etapa no implementada. Usando hiperparametros por defecto.")

    mejores_hiperparametros = {
        "hidden_layer_sizes": (100, 50),
        "activation":         "relu",
        "solver":             "adam",
        "alpha":              1e-4,
        "learning_rate_init": 1e-3,
        "batch_size":         32,
        "max_iter":           300,
        "early_stopping":     True,
        "validation_fraction": 0.1,
        "random_state":       42,
        "verbose":            False,
    }
    historial_mejor     = []
    historial_poblacion = []
    historial_a         = []

    return mejores_hiperparametros, historial_mejor, historial_poblacion, historial_a


def etapa_entrenamiento(mejores_hiperparametros, X_train, y_train, X_test, y_test):
    print("\n" + "=" * 60)
    print("ETAPA 5 — Entrenamiento final con mejores hiperparametros")
    print("=" * 60)

    modelo, accuracy = evaluar_final(
        mejores_hiperparametros, X_train, y_train, X_test, y_test
    )

    ruta_perdida = os.path.join(RUTA_SALIDA, "curva_perdida.png")
    graficar_perdida(modelo, ruta_perdida)

    print("[INFO] Entrenamiento final completado.")
    return modelo


def etapa_evaluacion(modelo, X_test, y_test, le):
    print("\n" + "=" * 60)
    print("ETAPA 6 — Evaluacion sobre conjunto de test")
    print("=" * 60)

    nombres_clases = list(le.classes_)
    resultados = evaluar(modelo, X_test, y_test, nombres_clases)

    ruta_confusion = os.path.join(RUTA_SALIDA, "matriz_confusion.png")
    graficar_confusion(modelo, X_test, y_test, nombres_clases, ruta_confusion)

    print("[INFO] Evaluacion completada.")
    return resultados


def etapa_visualizacion(historial_mejor, historial_poblacion, historial_a):
    print("\n" + "=" * 60)
    print("ETAPA 7 — Graficas del optimizador")
    print("=" * 60)

    if historial_mejor:
        graficar_evolucion_fitness(historial_mejor)

    if historial_poblacion:
        graficar_tendencia_central(historial_poblacion)

    if historial_a:
        graficar_coeficiente_a(historial_a)

    print("[INFO] Graficas del optimizador guardadas.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Proyecto: Optimizacion hibrida GWO/GA para MLP — Dry Bean")
    print("=" * 60)

    print(f"\n[INFO] Cargando dataset desde: {RUTA_CSV}")
    df = cargar_datos(RUTA_CSV)
    print(f"[INFO] Filas: {df.shape[0]}, Columnas: {df.shape[1]}")

    if ETAPAS["eda"]:
        etapa_eda(df)

    X = y = scaler = le = None
    if ETAPAS["preprocessing"]:
        X, y, scaler, le = etapa_preprocessing(df)

    X_train = X_val = X_test = y_train = y_val = y_test = None
    if ETAPAS["split"] and X is not None:
        X_train, X_val, X_test, y_train, y_val, y_test = etapa_split(X, y)

    mejores_hp = historial_mejor = historial_poblacion = historial_a = None
    if ETAPAS["optimizacion"] and X_train is not None:
        mejores_hp, historial_mejor, historial_poblacion, historial_a = etapa_optimizacion(
            X_train, y_train, X_val, y_val
        )

    modelo = None
    if ETAPAS["entrenamiento"] and mejores_hp is not None:
        modelo = etapa_entrenamiento(mejores_hp, X_train, y_train, X_test, y_test)

    if ETAPAS["evaluacion"] and modelo is not None:
        etapa_evaluacion(modelo, X_test, y_test, le)

    if historial_mejor is not None:
        etapa_visualizacion(historial_mejor, historial_poblacion, historial_a)

    print("\n" + "=" * 60)
    print("[INFO] Pipeline finalizado.")
    print("=" * 60)


if __name__ == "__main__":
    main()
    sys.exit(0)