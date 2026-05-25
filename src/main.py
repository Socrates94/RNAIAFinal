import os
import sys

# Asegura que los imports relativos funcionen desde cualquier CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

# ---------------------------------------------------------------------------
# Configuracion global
# ---------------------------------------------------------------------------

ETAPAS = {
    "eda":           True,
    "preprocessing": True,
}


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


def main():
    print("=" * 60)
    print("Proyecto: Optimizacion hibrida GWO/GA para MLP — Dry Bean")
    print("=" * 60)

    # Carga unica del dataset compartida entre etapas
    print(f"\n[INFO] Cargando dataset desde: {RUTA_CSV}")
    df = cargar_datos(RUTA_CSV)
    print(f"[INFO] Filas: {df.shape[0]}, Columnas: {df.shape[1]}")

    if ETAPAS["eda"]:
        etapa_eda(df)

    if ETAPAS["preprocessing"]:
        X, y, scaler, le = etapa_preprocessing(df)


if __name__ == "__main__":
    main()
