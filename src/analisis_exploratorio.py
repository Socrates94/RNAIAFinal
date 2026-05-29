import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_CSV      = os.path.join(BASE_DIR, "DryBeanDataset", "Dry_Bean_Dataset.csv")
RUTA_GRAFICAS = os.path.join(BASE_DIR, "Graficas", "EDA")


def cargar_datos(ruta):
    df = pd.read_csv(ruta)
    return df


def informacion_general(df):
    print("=" * 60)
    print("[BLOQUE 1] Informacion general del dataset")
    print("=" * 60)

    print(f"\n[INFO] Filas: {df.shape[0]}, Columnas: {df.shape[1]}")

    print("\n-- Tipos de datos por columna --")
    print(df.dtypes)

    print("\n-- Valores nulos por columna --")
    nulos = df.isnull().sum()
    print(nulos)
    print(f"\nTotal de celdas nulas: {nulos.sum()}")

    print("\n-- Filas duplicadas --")
    duplicados = df.duplicated().sum()
    print(f"Total de filas duplicadas: {duplicados}")

    print("\n-- Estadisticas descriptivas --")
    print(df.describe())


def distribucion_clases(df):
    print("=" * 60)
    print("[BLOQUE 2] Distribucion de clases")
    print("=" * 60)

    conteo = df["Class"].value_counts()

    print("\n-- Muestras por clase --\n")
    for clase, n in conteo.items():
        porcentaje = n / len(df) * 100
        print(f"  {clase:<12} {n:>5} muestras  ({porcentaje:.1f}%)")

    os.makedirs(RUTA_GRAFICAS, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    conteo.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Distribucion de clases — Dry Bean Dataset")
    ax.set_xlabel("Variedad de frijol")
    ax.set_ylabel("Numero de muestras")
    ax.tick_params(axis="x", rotation=45)

    for bar in ax.patches:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            str(int(bar.get_height())),
            ha="center", va="bottom", fontsize=9
        )

    plt.tight_layout()
    ruta = os.path.join(RUTA_GRAFICAS, "distribucion_clases.png")
    fig.savefig(ruta)
    plt.close()
    print(f"\n[INFO] Grafica guardada en: {ruta}\n")


def histogramas_features(df):
    print("=" * 60)
    print("[BLOQUE 3] Histogramas de features numericas")
    print("=" * 60)

    features = [c for c in df.columns if c != "Class"]

    cols = 4
    rows = -(-len(features) // cols)  # ceil sin import math
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    axes = axes.flatten()

    for i, col in enumerate(features):
        axes[i].hist(df[col], bins=40, color="steelblue", edgecolor="white")
        axes[i].set_title(col, fontsize=9)
        axes[i].set_ylabel("Frecuencia")

    plt.suptitle("Distribucion de features numericas — Dry Bean Dataset", fontsize=13)
    plt.tight_layout()

    ruta = os.path.join(RUTA_GRAFICAS, "histogramas_features.png")
    fig.savefig(ruta)
    plt.close()
    print(f"[INFO] Grafica guardada en: {ruta}\n")


def heatmap_correlacion(df):
    print("=" * 60)
    print("[BLOQUE 4] Heatmap de correlacion entre features")
    print("=" * 60)

    features = [c for c in df.columns if c != "Class"]
    correlacion = df[features].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        correlacion,
        annot=True, fmt=".2f",
        cmap="coolwarm", center=0,
        linewidths=0.5, ax=ax
    )
    ax.set_title("Matriz de correlacion — Dry Bean Dataset", fontsize=13)
    plt.tight_layout()

    ruta = os.path.join(RUTA_GRAFICAS, "heatmap_correlacion.png")
    fig.savefig(ruta)
    plt.close()
    print(f"[INFO] Grafica guardada en: {ruta}\n")

    print("-- Pares con correlacion alta (|r| > 0.95) --")
    for i in range(len(correlacion.columns)):
        for j in range(i + 1, len(correlacion.columns)):
            r = correlacion.iloc[i, j]
            if abs(r) > 0.95:
                print(f"  {correlacion.columns[i]} <-> {correlacion.columns[j]}: {r:.4f}")


def boxplots_por_clase(df):
    print("=" * 60)
    print("[BLOQUE 5] Boxplots por clase")
    print("=" * 60)

    features = [c for c in df.columns if c != "Class"]

    cols = 4
    rows = -(-len(features) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.5))
    axes = axes.flatten()

    for i, col in enumerate(features):
        df.boxplot(column=col, by="Class", ax=axes[i], grid=False)
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=45)

    plt.suptitle("Distribucion por clase — todas las features", fontsize=13)
    plt.tight_layout()

    ruta = os.path.join(RUTA_GRAFICAS, "boxplots_por_clase.png")
    fig.savefig(ruta)
    plt.close()
    print(f"[INFO] Grafica guardada en: {ruta}\n")


if __name__ == "__main__":
    df = cargar_datos(RUTA_CSV)

    informacion_general(df)
    distribucion_clases(df)
    histogramas_features(df)
    heatmap_correlacion(df)
    boxplots_por_clase(df)
