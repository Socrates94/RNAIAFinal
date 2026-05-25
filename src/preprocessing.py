import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

RUTA_CSV = os.path.join("DryBeanDataset", "Dry_Bean_Dataset.csv")

FEATURES = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
    "AspectRation", "Eccentricity", "ConvexArea", "EquivDiameter",  # nombre original UCI (typo del dataset)
    "Extent", "Solidity", "roundness", "Compactness",
    "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4",
]

def cargar_datos():
    """Lee el CSV y elimina filas duplicadas."""
    df = pd.read_csv(RUTA_CSV)
    df = df.drop_duplicates()
    print(f"[INFO] Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df

def preprocesar(df):
    """
    Separa X e y, normaliza las features y codifica las etiquetas.

    Returns:
        X (ndarray): features normalizadas.
        y (ndarray): etiquetas codificadas como enteros.
        scaler (StandardScaler): para transformar datos nuevos.
        le (LabelEncoder): para decodificar predicciones.
    """
    X_raw = df[FEATURES].values
    y_raw = df["Class"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print(f"[INFO] Clases: {list(le.classes_)}")
    print(f"[INFO] Shape X: {X.shape}, Shape y: {y.shape}")

    return X, y, scaler, le  