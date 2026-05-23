import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

def evaluar(modelo, X_test, y_test, nombres_clases):
    """
    Evalúa el modelo sobre el conjunto de test y reporta métricas completas.

    Args:
        modelo: MLPClassifier entrenado.
        X_test (ndarray): features de test.
        y_test (ndarray): etiquetas reales.
        nombres_clases (list): nombres legibles de cada clase.

    Returns:
        dict con accuracy, f1_macro y f1_weighted.
    """
    y_pred = modelo.predict(X_test)

    acc        = accuracy_score(y_test, y_pred)
    f1_macro   = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    print("=" * 60)
    print("[METRICAS] Resultados sobre conjunto de test")
    print("=" * 60)
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  F1 Macro:    {f1_macro:.4f}")
    print(f"  F1 Weighted: {f1_weighted:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=nombres_clases))

    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


def graficar_confusion(modelo, X_test, y_test, nombres_clases, ruta_salida):
    """Genera y guarda la matriz de confusión como heatmap."""
    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=nombres_clases,
        yticklabels=nombres_clases,
        ax=ax
    )
    ax.set_title("Matriz de confusion — Dry Bean Dataset")
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    plt.tight_layout()
    fig.savefig(ruta_salida)
    plt.close()
    print(f"[INFO] Matriz de confusion guardada en: {ruta_salida}")
