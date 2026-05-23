import os
import numpy as np
import matplotlib.pyplot as plt

# Ruta de salida para graficas del optimizador
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_SALIDA   = os.path.join(BASE_DIR, "Graficas", "Resultados")


def graficar_evolucion_fitness(historial_fitness, ruta_salida=None):
    """
    Grafica la evolucion del mejor fitness por iteracion del optimizador hibrido.

    Args:
        historial_fitness (list[float]): mejor fitness registrado en cada iteracion.
        ruta_salida (str): ruta completa del archivo PNG. Si es None, usa RUTA_SALIDA.
    """
    if ruta_salida is None:
        ruta_salida = os.path.join(RUTA_SALIDA, "evolucion_fitness.png")

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

    iteraciones = list(range(1, len(historial_fitness) + 1))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(iteraciones, historial_fitness, color="steelblue", linewidth=2, marker="o", markersize=3)
    ax.set_title("Evolucion del fitness — GWO-GA")
    ax.set_xlabel("Iteracion")
    ax.set_ylabel("Accuracy en validacion")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig.savefig(ruta_salida)
    plt.close()
    print(f"[INFO] Grafica de evolucion guardada en: {ruta_salida}")


def graficar_tendencia_central(historial_poblacion, ruta_salida=None):
    """
    Grafica media y desviacion estandar del fitness de la poblacion por iteracion.
    Util para analizar la convergencia del optimizador.

    Args:
        historial_poblacion (list[list[float]]): fitness de todos los individuos
                                                  por cada iteracion.
        ruta_salida (str): ruta de salida del PNG.
    """
    if ruta_salida is None:
        ruta_salida = os.path.join(RUTA_SALIDA, "tendencia_central.png")

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

    medias  = [np.mean(gen)  for gen in historial_poblacion]
    stds    = [np.std(gen)   for gen in historial_poblacion]
    iters   = list(range(1, len(medias) + 1))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(iters, medias, color="steelblue", linewidth=2, label="Media")
    ax.fill_between(
        iters,
        [m - s for m, s in zip(medias, stds)],
        [m + s for m, s in zip(medias, stds)],
        alpha=0.25, color="steelblue", label="Desv. estandar"
    )
    ax.set_title("Tendencia central del fitness — GWO-GA")
    ax.set_xlabel("Iteracion")
    ax.set_ylabel("Accuracy en validacion")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig.savefig(ruta_salida)
    plt.close()
    print(f"[INFO] Grafica de tendencia central guardada en: {ruta_salida}")
