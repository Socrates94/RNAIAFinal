import os
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

def crear_modelo(hidden_layer_sizes=(100, 50), activation="relu",
                 learning_rate_init=0.001, alpha=0.0001, max_iter=300):
    """
    Instancia un MLPClassifier con los hiperparametros dados.
    Estos hiperparametros son el espacio de busqueda del optimizador GWO-GA.

    Args:
        hidden_layer_sizes (tuple): neuronas por capa oculta.
        activation (str): funcion de activacion ('relu' o 'tanh').
        learning_rate_init (float): tasa de aprendizaje inicial.
        alpha (float): regularizacion L2.
        max_iter (int): epocas maximas.

    Returns:
        MLPClassifier sin entrenar.
    """
    modelo = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        learning_rate_init=learning_rate_init,
        alpha=alpha,
        max_iter=max_iter,
        random_state=42,
        verbose=False
    )
    return modelo


def entrenar(modelo, X_train, y_train):
    """Entrena el modelo y devuelve el mismo objeto ya ajustado."""
    modelo.fit(X_train, y_train)
    print(f"[INFO] Epocas ejecutadas: {modelo.n_iter_}")
    print(f"[INFO] Perdida final:     {modelo.loss_:.6f}")
    return modelo


def evaluar_fitness(hiperparametros, X_train, y_train, X_val, y_val):
    """
    Funcion de fitness para el optimizador hibrido.
    Entrena un MLP con los hiperparametros dados y retorna el accuracy en validacion.

    Args:
        hiperparametros (dict): claves hidden_layer_sizes, activation,
                                learning_rate_init, alpha.
        X_train, y_train: datos de entrenamiento.
        X_val, y_val: datos de validacion (sellados para test final).

    Returns:
        float: accuracy en validacion (mayor es mejor).
    """
    modelo = crear_modelo(**hiperparametros)
    modelo.fit(X_train, y_train)
    return modelo.score(X_val, y_val)


def graficar_perdida(modelo, ruta_salida):
    """Genera y guarda la grafica de la curva de perdida del entrenamiento."""
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(modelo.loss_curve_, color="steelblue", linewidth=2)
    ax.set_title("Curva de perdida — entrenamiento MLP")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Loss")
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(ruta_salida)
    plt.close()
    print(f"[INFO] Curva de perdida guardada en: {ruta_salida}")
