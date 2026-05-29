import os
import numpy as np
import matplotlib.pyplot as plt

# Ruta de salida para gráficas del optimizador
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_SALIDA = os.path.join(BASE_DIR, "Graficas", "Resultados")


def graficar_evolucion_fitness(historial_fitness, ruta_salida=None):
    """
    Gráfica la evolución del mejor fitness por iteración del
    optimizador híbrido GWO-GA.

    Args:
        historial_fitness (list[float]): Mejor fitness en cada iteración.
        ruta_salida (str): Ruta completa del archivo PNG.
    """
    if ruta_salida is None:
        ruta_salida = os.path.join(RUTA_SALIDA, "evolucion_fitness.png")

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

    iteraciones = list(range(1, len(historial_fitness) + 1))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(iteraciones, historial_fitness,
            color="steelblue", linewidth=2, marker="o", markersize=3)
    ax.set_title("Evolución del mejor fitness — Híbrido AG + GWO", fontsize=13)
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Accuracy en validación")
    ax.grid(True, linestyle="--", alpha=0.6)

    # Línea horizontal del mejor fitness encontrado
    if historial_fitness:
        mejor = max(historial_fitness)
        ax.axhline(y=mejor, color="green", linestyle="--", alpha=0.5,
                   label=f"Mejor: {mejor:.4f}")
        ax.legend()

    plt.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close()
    print(f"[INFO] Evolución del fitness guardada en: {ruta_salida}")


def graficar_tendencia_central(historial_poblacion, ruta_salida=None):
    """
    Gráfica media, mediana y desviación estándar del fitness de la
    población por iteración. Cubre el requisito de "comportamiento
    estadístico de tendencia central" del Punto 4 del examen.

    El optimizador pasa el fitness crudo de toda la población en cada
    iteración; las estadísticas se calculan aquí para no acoplar
    hybrid_optimizer a un formato de salida específico.

    Args:
        historial_poblacion (list[list[float]]): Fitness de todos los
            individuos por cada iteración.
        ruta_salida (str): Ruta de salida del PNG.
    """
    if ruta_salida is None:
        ruta_salida = os.path.join(RUTA_SALIDA, "tendencia_central.png")

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

    iteraciones = list(range(1, len(historial_poblacion) + 1))
    medias = [np.mean(gen) for gen in historial_poblacion]
    medianas = [np.median(gen) for gen in historial_poblacion]
    desviaciones = [np.std(gen) for gen in historial_poblacion]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Media con banda de desviación estándar
    ax.plot(iteraciones, medias, color="steelblue", linewidth=2, label="Media")
    ax.fill_between(
        iteraciones,
        [m - s for m, s in zip(medias, desviaciones)],
        [m + s for m, s in zip(medias, desviaciones)],
        alpha=0.2, color="steelblue", label="±1 Desv. estándar"
    )

    # Mediana
    ax.plot(iteraciones, medianas, color="darkorange", linewidth=2,
            linestyle="--", label="Mediana")

    ax.set_title("Tendencia central del fitness de la población — Híbrido AG + GWO",
                 fontsize=13)
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Accuracy en validación")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close()
    print(f"[INFO] Tendencia central guardada en: {ruta_salida}")


def graficar_coeficiente_a(historial_a, ruta_salida=None):
    """
    Gráfica la evolución del coeficiente 'a' del GWO a lo largo
    de las iteraciones. Muestra la transición de exploración
    (a=2) a explotación (a=0).

    Args:
        historial_a (list[float]): Valor de 'a' en cada iteración.
        ruta_salida (str): Ruta de salida del PNG.
    """
    if ruta_salida is None:
        ruta_salida = os.path.join(RUTA_SALIDA, "coeficiente_a.png")

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

    iteraciones = list(range(1, len(historial_a) + 1))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iteraciones, historial_a, color="crimson", linewidth=2)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5,
               label="|A|=1 (umbral exploración/explotación)")
    ax.set_title("Evolución del coeficiente 'a' — GWO", fontsize=13)
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Coeficiente a")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close()
    print(f"[INFO] Coeficiente 'a' guardado en: {ruta_salida}")