import os
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import FEATURES

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR = os.path.join(BASE_DIR, "DryBeanDataset", "split")

RANDOM_STATE = 17

def guardar_csv(X, y, subcarpeta, nombre_archivo):
    """
    Guarda los datos divididos en un archivo CSV.
    """
    ruta_carpeta = os.path.join(SPLITS_DIR, subcarpeta)
    os.makedirs(ruta_carpeta, exist_ok=True)
    
    # Reconstruimos el DataFrame con los nombres de las features
    df = pd.DataFrame(X, columns=FEATURES)
    df["Class"] = y  # La etiqueta ya está codificada
    
    ruta_completa = os.path.join(ruta_carpeta, nombre_archivo)
    df.to_csv(ruta_completa, index=False)

def split_data(X, y):
    """
    Divide X e y en train (70%), validation (15%) y test (15%).
    Usa estratificacion para mantener la proporcion de clases en cada particion.

    El conjunto de validation se usa exclusivamente como funcion de fitness
    durante la optimizacion hibrida GWO-GA. El conjunto de test permanece
    sellado hasta la evaluacion final.
    
    Además, exporta estos splits como CSV en DryBeanDataset/split/

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout, y_holdout,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_holdout
    )

    print(f"[INFO] Train:      {X_train.shape[0]} muestras (70%)")
    print(f"[INFO] Validation: {X_val.shape[0]} muestras (15%)")
    print(f"[INFO] Test:       {X_test.shape[0]} muestras (15%)")
    
    # Guardar en las carpetas correspondientes
    guardar_csv(X_train, y_train, "train", "train.csv")
    guardar_csv(X_val, y_val, "validation", "validation.csv")
    guardar_csv(X_test, y_test, "test", "test.csv")
    
    print(f"[INFO] Splits exportados como CSV en: {SPLITS_DIR}")

    return X_train, X_val, X_test, y_train, y_val, y_test
