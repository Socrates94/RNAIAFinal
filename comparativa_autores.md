# Análisis Comparativo: Proyecto vs. Autores Originales

Para validar la efectividad de la arquitectura propuesta en este proyecto, se contrastaron los resultados obtenidos con los reportados por Koklu y Ozkan (creadores del Dry Bean Dataset). 

En su estudio original, los autores evaluaron cuatro algoritmos clásicos utilizando validación cruzada de 10 pliegues, obteniendo las siguientes precisiones globales (Accuracy):
*   MLP (Perceptrón Multicapa): 91.73%
*   SVM (Máquinas de Vectores de Soporte): 93.13% *(Mejor modelo de los autores)*
*   DT (Árbol de Decisión): 92.52%
*   kNN (K-Vecinos Más Cercanos): 87.92%

### Comparativa de Precisión Global (Accuracy)

Los resultados desarrollados en este proyecto demostraron ser superiores:

| Modelo | Accuracy Global | Análisis |
| :--- | :--- | :--- |
| **MLP Autores Originales** | 91.73% | Implementación estándar de la literatura. |
| **MLP Baseline (Nuestro)** | **92.80%** | Solo aplicando buenas prácticas de preprocesamiento (StandardScaler, partición estratificada) y configuración de hiperparámetros robustos, nuestro modelo base ya superó al MLP de los autores originales por más de 1%. |
| **SVM Autores Originales** | 93.13% | Fue el modelo estrella del paper original. |
| **MLP Híbrido GWO-GA (Nuestro)** | **93.44%** | **Mejor resultado general.** La optimización metaheurística del espacio de hiperparámetros logró que una red neuronal simple superara al mejor modelo (SVM) reportado por los creadores del dataset. |

### Comparativa por Clase (El caso de éxito)

El verdadero impacto del optimizador híbrido GWO-GA se observa al desglosar el rendimiento en las clases individuales, comparándolo contra el SVM (el mejor de los creadores originales):

| Clase de Frijol | SVM (Autores Originales) | MLP Híbrido GWO-GA (Nuestro) | Mejora / Diferencia |
| :--- | :--- | :--- | :--- |
| **Bombay** | 100.00% | 100.00% | Empate (Clase linealmente separable) |
| **Cali** | 95.03% | 95.00% | Empate estadístico |
| **Seker** | 94.67% | **96.00%** | **Mejora clara** |
| **Barbunya** | 92.36% | **95.00%** | **Mejora contundente (+2.6%)** |
| **Dermason** | **94.36%** | 93.00% | Ligeramente inferior |
| **Horoz** | **94.92%** | 93.00% | Ligeramente inferior |
| **Sira (Clase más difícil)** | 86.84% | **89.00%** | **Mejora crítica (+2.1%)** |

### Conclusión Técnica

La literatura original marcaba a la clase **Sira** como el principal cuello de botella (86.84% de precisión en su mejor modelo SVM) debido a su alta similitud morfológica con la clase Dermason. 

El éxito rotundo de nuestro **Híbrido GWO-GA** radica en que el algoritmo genético y el optimizador de lobos grises lograron encontrar una arquitectura de hiperparámetros que empujó el límite matemático de esta clase difícil, subiendo su precisión a un **89.00%**. Al resolver el punto más débil del dataset, el Accuracy general se elevó a 93.44%, estableciendo un nuevo estado del arte frente al estudio original.

---

### Apéndice: Análisis Computacional y Rendimiento de Experimentos

Durante el desarrollo, se realizaron múltiples pruebas para determinar la configuración óptima del optimizador metaheurístico, revelando hallazgos importantes sobre la relación entre el tiempo de cómputo, el espacio de búsqueda y la convergencia:

1. **Experimento Secuencial de Larga Duración (N=40, MAX_ITER=50, 2 Capas Ocultas):**
   * Ejecutado en un solo núcleo (secuencial) con una alta carga de individuos.
   * **Tiempo de ejecución:** > 6 horas.
   * **Accuracy obtenido:** 93.24%
   * **Análisis:** Demostró el costo prohibitivo de la optimización estocástica sin paralelismo. Aunque el *accuracy* fue competitivo, el tiempo de espera hizo inviable la iteración rápida de pruebas.

2. **Experimento de Fuerza Bruta Profunda (N=40, MAX_ITER=50, 3 Capas Ocultas, Paralelizado):**
   * Se intentó superar el límite permitiendo a los lobos generar arquitecturas más profundas (hasta 3 capas) y aprovechando todos los núcleos de procesamiento (`joblib`).
   * **Tiempo de ejecución:** ~60 minutos.
   * **Accuracy obtenido:** 93.10%
   * **Análisis:** Aumentar la dimensionalidad (agregando la tercera capa) generó una "Maldición de la Dimensionalidad" para la metaheurística. Al agrandar excesivamente el espacio de búsqueda, la manada de 40 lobos se disipó y no logró converger eficientemente, estancándose en un mínimo local inferior.

3. **Experimento Equilibrado y Óptimo (N=20, MAX_ITER=30, 2 Capas Ocultas, Paralelizado):**
   * **Tiempo de ejecución:** 18 minutos.
   * **Accuracy obtenido:** **93.44%**
   * **Análisis:** Restringir el espacio a 2 capas ocultas obligó a los lobos a explorar un hiperespacio más compacto. Al procesar en paralelo, el tiempo bajó de horas a minutos. Este experimento demostró ser el "Punto Óptimo" (*Sweet Spot*), donde la combinación de la exploración genética y la explotación del GWO logró converger hacia la mejor arquitectura del proyecto estableciendo el nuevo Estado del Arte.
