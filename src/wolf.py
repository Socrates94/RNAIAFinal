import numpy as np
import copy

class Wolf:
    """
    Representa una solución candidata en el espacio de búsqueda de
    hiper-parámetros del MLPClassifier. Compatible con AG y GWO.
    """

    def __init__(self):
        """
        Inicializa un lobo con genes aleatorios dentro de los rangos
        definidos para cada hiper-parámetro del MLPClassifier.
        """
        # --- Genes continuos (escala logarítmica) ---
        self.alpha = 10 ** np.random.uniform(-4, -1)       # [1e-4, 1e-1]
        self.learning_rate_init = 10 ** np.random.uniform(-4, -1)  # [1e-4, 1e-1]

        # --- Genes enteros ---
        self.n_neuronas_capa1 = np.random.randint(10, 201)     # [10, 200]
        self.n_neuronas_capa2 = np.random.randint(0, 201)      # [0, 200]  (0 = sin 2da capa)
        self.batch_size = np.random.choice([16, 32, 64, 128, 256])  # Potencias de 2
        self.max_iter = np.random.randint(100, 501)            # [100, 500]

        # --- Genes categóricos ---
        self.activacion = np.random.choice([0, 1, 2])  # 0=relu, 1=tanh, 2=logistic
        self.solver = np.random.choice([0, 1])          # 0=adam, 1=sgd

        # --- Metadatos ---
        self.fitness = 0.0       # Accuracy en validación (0.0 a 1.0)
        self.role = "omega"      # alpha, beta, delta, omega

    # ─────────────────────────────────────────────────────────────
    #  MÉTODOS PARA EL MLPClassifier
    # ─────────────────────────────────────────────────────────────

    def to_dict(self):
        """
        Convierte los genes del lobo a un diccionario listo para
        pasarse como argumentos al MLPClassifier de sklearn.

        Returns:
            dict: Diccionario con nombres de hiper-parámetros del MLP.
        """
        # Mapeo de categóricos a strings
        mapa_activacion = {0: "relu", 1: "tanh", 2: "logistic"}
        mapa_solver = {0: "adam", 1: "sgd"}

        # Construir hidden_layer_sizes
        if self.n_neuronas_capa2 == 0:
            capas = (self.n_neuronas_capa1,)
        else:
            capas = (self.n_neuronas_capa1, self.n_neuronas_capa2)

        return {
            "hidden_layer_sizes": capas,
            "activation": mapa_activacion[self.activacion],
            "solver": mapa_solver[self.solver],
            "alpha": self.alpha,
            "learning_rate_init": self.learning_rate_init,
            "batch_size": int(self.batch_size),
            "max_iter": int(self.max_iter),
            "early_stopping": True,
            "validation_fraction": 0.1,
            "random_state": 42,
            "verbose": False
        }

    # ─────────────────────────────────────────────────────────────
    #  MÉTODOS PARA EL ALGORITMO GENÉTICO (AG)
    # ─────────────────────────────────────────────────────────────

    def copy(self):
        """
        Crea una copia profunda del lobo. Necesaria para el cruce
        genético sin modificar los originales.

        Returns:
            Wolf: Nuevo lobo idéntico.
        """
        return copy.deepcopy(self)

    def mutar(self, prob_mutacion=0.1):
        """
        Aplica mutación gen por gen con probabilidad prob_mutacion.

        Args:
            prob_mutacion (float): Probabilidad de mutar cada gen.
        """
        # Mutar continuos (perturbación gaussiana en espacio logarítmico)
        if np.random.random() < prob_mutacion:
            self.alpha = 10 ** (np.log10(self.alpha) + np.random.normal(0, 0.5))
            self.alpha = np.clip(self.alpha, 1e-4, 1e-1)

        if np.random.random() < prob_mutacion:
            self.learning_rate_init = 10 ** (np.log10(self.learning_rate_init) + np.random.normal(0, 0.5))
            self.learning_rate_init = np.clip(self.learning_rate_init, 1e-4, 1e-1)

        # Mutar enteros (reasignar aleatorio)
        if np.random.random() < prob_mutacion:
            self.n_neuronas_capa1 = np.random.randint(10, 201)

        if np.random.random() < prob_mutacion:
            self.n_neuronas_capa2 = np.random.randint(0, 201)

        if np.random.random() < prob_mutacion:
            self.batch_size = np.random.choice([16, 32, 64, 128, 256])

        if np.random.random() < prob_mutacion:
            self.max_iter = np.random.randint(100, 501)

        # Mutar categóricos (cambiar a otra opción)
        if np.random.random() < prob_mutacion:
            self.activacion = np.random.choice([0, 1, 2])

        if np.random.random() < prob_mutacion:
            self.solver = np.random.choice([0, 1])

    # ─────────────────────────────────────────────────────────────
    #  MÉTODOS PARA EL GREY WOLF OPTIMIZER (GWO)
    # ─────────────────────────────────────────────────────────────

    def clip_bounds(self):
        """Asegura que todos los genes estén dentro de sus rangos permitidos."""
        self.alpha = np.clip(self.alpha, 1e-4, 1e-1)
        self.learning_rate_init = np.clip(self.learning_rate_init, 1e-4, 1e-1)
        self.n_neuronas_capa1 = int(np.clip(self.n_neuronas_capa1, 10, 200))
        self.n_neuronas_capa2 = int(np.clip(self.n_neuronas_capa2, 0, 200))
        # Snap al valor válido más cercano — GWO puede producir valores intermedios
        _opciones_batch = np.array([16, 32, 64, 128, 256])
        self.batch_size = int(_opciones_batch[np.argmin(np.abs(_opciones_batch - self.batch_size))])
        self.max_iter = int(np.clip(self.max_iter, 100, 500))
        self.activacion = int(np.clip(self.activacion, 0, 2))
        self.solver = int(np.clip(self.solver, 0, 1))

    def update_position_gwo(self, alpha_pos, beta_pos, delta_pos, a):
        """
        Mueve los genes del lobo omega hacia las posiciones de los
        líderes Alpha, Beta y Delta. Adaptado gen por gen.

        Args:
            alpha_pos (Wolf): Lobo Alpha.
            beta_pos (Wolf): Lobo Beta.
            delta_pos (Wolf): Lobo Delta.
            a (float): Coeficiente de exploración/explotación (2 → 0).
        """
        # Lista de genes continuos (se tratan con GWO normal)
        genes_continuos = ["alpha", "learning_rate_init"]
        for gen in genes_continuos:
            valor_actual = getattr(self, gen)
            v_alpha = getattr(alpha_pos, gen)
            v_beta = getattr(beta_pos, gen)
            v_delta = getattr(delta_pos, gen)

            # Ecuaciones GWO
            r1, r2 = np.random.random(), np.random.random()
            A1, C1 = 2 * a * r1 - a, 2 * r2
            D_alpha = abs(C1 * v_alpha - valor_actual)
            X1 = v_alpha - A1 * D_alpha

            r1, r2 = np.random.random(), np.random.random()
            A2, C2 = 2 * a * r1 - a, 2 * r2
            D_beta = abs(C2 * v_beta - valor_actual)
            X2 = v_beta - A2 * D_beta

            r1, r2 = np.random.random(), np.random.random()
            A3, C3 = 2 * a * r1 - a, 2 * r2
            D_delta = abs(C3 * v_delta - valor_actual)
            X3 = v_delta - A3 * D_delta

            setattr(self, gen, (X1 + X2 + X3) / 3)

        # Genes enteros: mismo mecanismo pero redondeando al final
        genes_enteros = ["n_neuronas_capa1", "n_neuronas_capa2", "batch_size", "max_iter"]
        for gen in genes_enteros:
            valor_actual = getattr(self, gen)
            v_alpha = getattr(alpha_pos, gen)
            v_beta = getattr(beta_pos, gen)
            v_delta = getattr(delta_pos, gen)

            r1, r2 = np.random.random(), np.random.random()
            A1, C1 = 2 * a * r1 - a, 2 * r2
            D_alpha = abs(C1 * v_alpha - valor_actual)
            X1 = v_alpha - A1 * D_alpha

            r1, r2 = np.random.random(), np.random.random()
            A2, C2 = 2 * a * r1 - a, 2 * r2
            D_beta = abs(C2 * v_beta - valor_actual)
            X2 = v_beta - A2 * D_beta

            r1, r2 = np.random.random(), np.random.random()
            A3, C3 = 2 * a * r1 - a, 2 * r2
            D_delta = abs(C3 * v_delta - valor_actual)
            X3 = v_delta - A3 * D_delta

            nuevo_valor = (X1 + X2 + X3) / 3
            setattr(self, gen, int(round(nuevo_valor)))

        # Genes categóricos: votación (el valor más cercano entre los 3 líderes)
        genes_categoricos = ["activacion", "solver"]
        for gen in genes_categoricos:
            nuevo_valor = (getattr(alpha_pos, gen) + getattr(beta_pos, gen) + getattr(delta_pos, gen)) / 3
            setattr(self, gen, int(round(np.clip(nuevo_valor, 0, 2 if gen == "activacion" else 1))))

        # Ajustar a límites
        self.clip_bounds()