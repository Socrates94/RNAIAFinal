from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

def split_data(X, y):
    """
    Divide X e y en train (70%), validation (15%) y test (15%).
    Usa estratificacion para mantener la proporcion de clases en cada particion.

    El conjunto de validation se usa exclusivamente como funcion de fitness
    durante la optimizacion hibrida GWO-GA. El conjunto de test permanece
    sellado hasta la evaluacion final.

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

    return X_train, X_val, X_test, y_train, y_val, y_test
