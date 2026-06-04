import os
import sys

# Asegura que los imports relativos funcionen desde cualquier CWD
_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.join(_SRC, "metaheuristicos"))

from analisis_exploratorio import cargar_datos, RUTA_CSV
from preprocessing import preprocesar
from split_data import split_data
from mlp_trainer import evaluar_final, graficar_perdida
from metrics import evaluar, graficar_confusion
from visualizer import RUTA_SALIDA

def main():
    print("=" * 60)
    print("Proyecto: MLP Baseline (Sin Metaheurísticas) — Dry Bean")
    print("=" * 60)

    print(f"\n[INFO] Cargando dataset desde: {RUTA_CSV}")
    df = cargar_datos(RUTA_CSV)
    
    print("\n" + "=" * 60)
    print("ETAPA 1 — Preprocesamiento")
    print("=" * 60)
    X, y, scaler, le = preprocesar(df)
    
    print("\n" + "=" * 60)
    print("ETAPA 2 — Division de datos (70 / 15 / 15)")
    print("=" * 60)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    print("\n" + "=" * 60)
    print("ETAPA 3 — Entrenamiento Baseline")
    print("=" * 60)
    
    # Hiperparámetros por defecto para comparar contra el optimizador
    hiperparametros_baseline = {
        "hidden_layer_sizes": (100, 50),
        "activation":         "relu",
        "solver":             "adam",
        "alpha":              0.0001,
        "learning_rate_init": 0.001,
        "batch_size":         32,
        "max_iter":           300,
        "early_stopping":     True,
        "validation_fraction": 0.1,
        "random_state":       42,
        "verbose":            True
    }
    
    print(f"[INFO] Hiperparámetros del Baseline: {hiperparametros_baseline}")
    
    modelo, accuracy = evaluar_final(
        hiperparametros_baseline, X_train, y_train, X_test, y_test
    )
    
    # Se guardan con nombres distintos para no sobreescribir las de main.py
    ruta_perdida = os.path.join(RUTA_SALIDA, "curva_perdida_baseline.png")
    graficar_perdida(modelo, ruta_perdida)
    
    print("\n" + "=" * 60)
    print("ETAPA 4 — Evaluacion sobre conjunto de test")
    print("=" * 60)
    nombres_clases = list(le.classes_)
    resultados = evaluar(modelo, X_test, y_test, nombres_clases)
    
    ruta_confusion = os.path.join(RUTA_SALIDA, "matriz_confusion_baseline.png")
    graficar_confusion(modelo, X_test, y_test, nombres_clases, ruta_confusion)

    print("\n" + "=" * 60)
    print("[INFO] Ejecución Baseline finalizada.")
    print("=" * 60)

if __name__ == "__main__":
    main()
