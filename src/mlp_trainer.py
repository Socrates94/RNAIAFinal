import os
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier


def crear_modelo(hidden_layer_sizes=(100, 50),
                 activation="relu",
                 solver="adam",
                 alpha=0.0001,
                 learning_rate_init=0.001,
                 batch_size="auto",
                 max_iter=300,
                 early_stopping=True,
                 validation_fraction=0.1,
                 random_state=42,
                 verbose=False):
    """
    Instancia un MLPClassifier con los hiper-parámetros dados.
    Estos hiper-parámetros definen el espacio de búsqueda del
    optimizador híbrido GWO-GA.

    Args:
        hidden_layer_sizes (tuple): Neuronas por capa oculta.
        activation (str): 'relu', 'tanh' o 'logistic'.
        solver (str): 'adam' o 'sgd'.
        alpha (float): Regularización L2.
        learning_rate_init (float): Tasa de aprendizaje inicial.
        batch_size (int o 'auto'): Tamaño de minibatch.
        max_iter (int): Épocas máximas.
        early_stopping (bool): Detener si no mejora validación.
        validation_fraction (float): Fracción para early stopping.
        random_state (int): Semilla para reproducibilidad.
        verbose (bool): Imprimir progreso del entrenamiento.

    Returns:
        MLPClassifier: Modelo sin entrenar.
    """
    modelo = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        batch_size=batch_size,
        max_iter=max_iter,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        random_state=random_state,
        verbose=verbose
    )
    return modelo


def entrenar(modelo, X_train, y_train, verbose=True):
    """
    Entrena el modelo y devuelve el mismo objeto ya ajustado.

    Args:
        modelo (MLPClassifier): Modelo sin entrenar.
        X_train (ndarray): Features de entrenamiento.
        y_train (ndarray): Etiquetas de entrenamiento.
        verbose (bool): Si True, imprime información del entrenamiento.

    Returns:
        MLPClassifier: Modelo entrenado.
    """
    modelo.fit(X_train, y_train)

    if verbose:
        print(f"[INFO] Épocas ejecutadas: {modelo.n_iter_}")
        print(f"[INFO] Pérdida final:     {modelo.loss_:.6f}")

    return modelo


def evaluar_fitness(wolf, X_train, y_train, X_val, y_val):
    """
    Función de fitness para el optimizador híbrido GWO-GA.
    Convierte un lobo a hiper-parámetros, entrena un MLP y
    devuelve el accuracy en validación.

    Esta función será llamada cientos de veces durante la
    optimización, por lo que no imprime nada y usa early_stopping
    para acelerar el entrenamiento.

    Args:
        wolf (Wolf): Individuo con genes de hiper-parámetros.
        X_train (ndarray): Features de entrenamiento.
        y_train (ndarray): Etiquetas de entrenamiento.
        X_val (ndarray): Features de validación.
        y_val (ndarray): Etiquetas de validación.

    Returns:
        float: Accuracy en validación (entre 0.0 y 1.0).
    """
    hiperparametros = wolf.to_dict()
    # Desactivar early_stopping interno: ya tenemos X_val externo para medir fitness.
    # Activarlo aqui recortaria X_train innecesariamente.
    hiperparametros["early_stopping"] = False
    hiperparametros.pop("validation_fraction", None)
    modelo = crear_modelo(**hiperparametros)
    modelo.fit(X_train, y_train)
    fitness = modelo.score(X_val, y_val)
    return fitness


def evaluar_final(hiperparametros, X_train, y_train, X_test, y_test):
    """
    Entrena el modelo final con los mejores hiper-parámetros
    y lo evalúa en el conjunto de test sellado.

    Args:
        hiperparametros (dict): Mejores hiper-parámetros encontrados.
        X_train (ndarray): Features de entrenamiento.
        y_train (ndarray): Etiquetas de entrenamiento.
        X_test (ndarray): Features de test.
        y_test (ndarray): Etiquetas de test.

    Returns:
        tuple: (modelo_entrenado, accuracy_en_test)
    """
    # wolf.to_dict() ya incluye verbose=False, no se repite para evitar TypeError.
    modelo = crear_modelo(**hiperparametros)
    modelo.fit(X_train, y_train)
    accuracy = modelo.score(X_test, y_test)

    print(f"\n[RESULTADO FINAL]")
    print(f"  Hiper-parámetros: {hiperparametros}")
    print(f"  Accuracy en test: {accuracy:.4f}")
    print(f"  Épocas ejecutadas: {modelo.n_iter_}")
    print(f"  Pérdida final:     {modelo.loss_:.6f}")

    return modelo, accuracy


def graficar_perdida(modelo, ruta_salida):
    """
    Genera y guarda la gráfica de la curva de pérdida del
    entrenamiento del MLP.

    Args:
        modelo (MLPClassifier): Modelo ya entrenado.
        ruta_salida (str): Ruta donde guardar la gráfica.
    """
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(modelo.loss_curve_, color="steelblue", linewidth=2)
    ax.set_title("Curva de pérdida — Entrenamiento MLP", fontsize=13)
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close()

    print(f"[INFO] Curva de pérdida guardada en: {ruta_salida}")
