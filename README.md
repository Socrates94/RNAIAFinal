# Proyecto IA — Clasificacion de Variedades de Frijol (Dry Bean Dataset)

Clasificacion de 7 variedades de frijol seco usando un MLPClassifier de scikit-learn sobre el
Dry Bean Dataset (UCI). El objetivo se logró exitosamente optimizando los hiperparametros de la red
mediante un algoritmo hibrido GWO-GA acelerado con procesamiento en paralelo.

## Estructura del proyecto

```
IA/
├── DryBeanDataset/
│   ├── Dry_Bean_Dataset.csv          # 13 611 muestras, 17 columnas
│   └── split/                        # Datos divididos y exportados
│       ├── train.csv                 # 70% (Entrenamiento)
│       ├── val.csv                   # 15% (Validación/Fitness)
│       └── test.csv                  # 15% (Evaluación Final)
│
├── Graficas/
│   ├── EDA/                          # Graficas del analisis exploratorio (no versionadas)
│   │   ├── boxplots_por_clase.png
│   │   ├── distribucion_clases.png
│   │   ├── heatmap_correlacion.png
│   │   └── histogramas_features.png
│   │
│   └── Resultados/                   # Graficas de matrices de confusion y convergencia GWO-GA
│       ├── coeficiente_a.png
│       ├── curva_perdida.png
│       ├── curva_perdida_baseline.png
│       ├── evolucion_fitness.png
│       ├── matriz_confusion.png
│       ├── matriz_confusion_baseline.png
│       └── tendencia_central.png
│
├── src/
│   ├── main.py                       # Orquestador del pipeline completo
│   ├── analisis_exploratorio.py      # EDA: estadisticas, histogramas, heatmap, boxplots
│   ├── preprocessing.py              # Carga del CSV, StandardScaler, LabelEncoder
│   ├── split_data.py                 # Split estratificado 70/15/15 (train/val/test)
│   ├── mlp_trainer.py                # MLPClassifier: crear, entrenar, evaluar_fitness
│   ├── metrics.py                    # Accuracy, F1-macro, F1-weighted, matriz de confusion
│   ├── visualizer.py                 # Graficas de evolucion → Graficas/Resultados/
│   └── metaheuristicos/              # Módulo del optimizador híbrido GWO-GA
│       ├── __init__.py
│       ├── wolf.py                   # Representacion del individuo (vector de hiperparametros)
│       ├── ga.py                     # Operadores geneticos (selección, cruce, mutación)
│       ├── gwo.py                    # Grey Wolf Optimizer (Alpha, Beta, Delta)
│       ├── paralelizador.py          # Procesamiento multi-core de fitness via joblib
│       ├── hiperparametros.py        # Definición estricta de límites (Bounds) y decodificador
│       └── hybrid_optimizer.py       # Orquestacion y pipeline principal del hibrido
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Flujo actual (sin metaheuristicos)

```
main.py
  │
  ├─ 1 ─► analisis_exploratorio.py   →  Graficas/EDA/
  │           distribucion de clases, histogramas, heatmap de correlacion, boxplots por clase
  │
  ├─ 2 ─► preprocessing.py           →  X_scaled (StandardScaler), y_encoded (LabelEncoder)
  │
  ├─ 3 ─► split_data.py              →  X_train / X_val / X_test  (70 / 15 / 15, estratificado)
  │
  ├─ 4 ─► mlp_trainer.py             →  entrena MLP con hiperparametros fijos (baseline)
  │
  └─ 5 ─► metrics.py                 →  reporte final sobre X_test (accuracy, F1, confusion matrix)
```

## Flujo completo (con metaheuristicos)

```
main.py
  │
  ├─ 1-3 ─► Etapas de Preprocesamiento y Split (descritas arriba)
  │
  ├─ 4 ─► metaheuristicos/hybrid_optimizer.py
  │           │
  │           ├─► wolf.py             → Genera N lobos iniciales (vectores de hiperparámetros)
  │           ├─► paralelizador.py    → Evalúa el fitness inicial en paralelo (multi-core)
  │           │
  │           ├─► [Bucle MAX_ITER]
  │           │     ├─► ga.py         → Aplica Selección por Torneo, Cruce Uniforme y Mutación
  │           │     ├─► gwo.py        → Recalcula distancia y mueve a Omegas hacia Alpha, Beta y Delta
  │           │     └─► paralelizador → Evalúa el nuevo fitness de la manada tras moverse
  │           │
  │           └─► Retorna el Lobo Alpha (mejor hiperparámetro encontrado)
  │
  ├─ 5 ─► mlp_trainer.entrenar()      ← Entrena la red final con X_train usando los genes del Alpha
  ├─ 6 ─► metrics.evaluar()           → Valida el modelo contra X_test (datos sellados no vistos)
  └─ 7 ─► visualizer.py               → Genera matriz de confusión y curvas de evolución en Graficas/
```

## Flujo interno del optimizador hibrido GWO-GA

El optimizador recibe `X_train`, `X_val`, `y_train`, `y_val` y una poblacion de `N` lobos.
Cada lobo es un vector de 8 genes que representan los hiperparametros del MLP:

| Gen                    | Tipo       | Rango              |
| ---------------------- | ---------- | ------------------ |
| `n_neuronas_capa1`   | Entero     | [10, 200]          |
| `n_neuronas_capa2`   | Entero     | [0, 200]           |
| `activacion`         | Categorico | relu/tanh/logistic |
| `solver`             | Categorico | adam/sgd           |
| `alpha`              | Continuo   | [0.0001, 0.1]      |
| `learning_rate_init` | Continuo   | [0.0001, 0.1]      |
| `batch_size`         | Entero     | [16, 256]          |
| `max_iter`           | Entero     | [100, 500]         |

### Fase 1 — Inicializacion

```
1. Generar N lobos con genes aleatorios dentro de sus rangos
2. Evaluar fitness de cada lobo:
       fitness = accuracy del MLPClassifier entrenado con esos hiperparametros sobre X_val
3. Ordenar poblacion por fitness descendente
4. Asignar jerarquia:
       Alpha  ← lobo[0]   (mejor fitness)
       Beta   ← lobo[1]
       Delta  ← lobo[2]
       Omega  ← lobos[3..N-1]
```

### Fase 2 — Ciclo principal (se repite MaxIter veces)

Cada iteracion hace dos cosas en orden:

#### Bloque AG — Exploracion global (actua sobre toda la poblacion)

```
a ← 2 - 2 * (iter / MaxIter)    // decrece de 2 a 0 a lo largo de las iteraciones

1. Conservar elite: Alpha, Beta y Delta pasan directos a la nueva generacion
2. Completar la poblacion con hijos generados por:
       a. Seleccion: torneo binario (dos lobos distintos, gana el de mayor fitness)
       b. Cruce uniforme (probabilidad ProbCruce):
              por cada gen, elegir aleatoriamente de que padre heredarlo
       c. Mutacion por gen (probabilidad ProbMutacion):
              - gen continuo  → perturbacion gaussiana ajustada al rango
              - gen entero    → reemplazo aleatorio dentro del rango
              - gen categorico→ categoria aleatoria entre las opciones
       d. Evaluar fitness de cada hijo contra X_val
3. Reemplazar poblacion antigua con la nueva
```

#### Bloque GWO — Explotacion local (actua solo sobre los lobos Omega)

```
Para cada lobo Omega y para cada gen j:
   Calcular 3 posiciones candidatas guiadas por Alpha, Beta y Delta:

   D_alpha = | C1 * Alpha[j] - Omega[j] |     X1 = Alpha[j] - A1 * D_alpha
   D_beta  = | C2 * Beta[j]  - Omega[j] |     X2 = Beta[j]  - A2 * D_beta
   D_delta = | C3 * Delta[j] - Omega[j] |     X3 = Delta[j] - A3 * D_delta

   Omega[j] ← (X1 + X2 + X3) / 3    // promedio de las tres estimaciones

   Ajustar al rango valido del gen
   Redondear si el gen es entero o categorico

Reevaluar fitness de cada Omega modificado
```

> Los coeficientes A y C se generan con numeros aleatorios r1, r2 en [0,1]:
> `A = 2*a*r1 - a`  |  `C = 2*r2`
> Cuando |A| < 1, el lobo converge hacia los lideres (explotacion).
> Cuando |A| > 1, se aleja para explorar otras regiones del espacio.

#### Actualizacion de jerarquia al final de cada iteracion

```
Ordenar toda la poblacion por fitness descendente
Reasignar Alpha, Beta, Delta con los 3 mejores actuales
Registrar estadisticas: mejor, peor, media, mediana, desviacion estandar del fitness
```

### Fase 3 — Resultado

```
Retornar Alpha (mejor lobo encontrado) y el historial de estadisticas por iteracion
El main.py usa los hiperparametros de Alpha para entrenar el MLP final sobre X_train
y evaluar sobre X_test (que permanece sellado durante toda la optimizacion)
```

## Dataset

| Propiedad    | Detalle                                                  |
| ------------ | -------------------------------------------------------- |
| Fuente       | UCI Machine Learning Repository                          |
| Muestras     | 13 611 originales (13 596 tras remover duplicados en preprocesado) |
| Features     | 16 morfologicas (area, perimetro, compacidad...)         |
| Clases       | 7 (BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA) |
| Desbalance   | Leve; DERMASON es la clase mayoritaria                   |
| Preprocesado | StandardScaler en X, LabelEncoder en y                   |

## Estado de implementacion

| Modulo                                  | Estado   | Descripcion                                            |
| --------------------------------------- | -------- | ------------------------------------------------------ |
| `analisis_exploratorio.py`            | Completo | EDA con 4 graficas exportadas                          |
| `preprocessing.py`                    | Completo | Carga, limpieza, normalizacion y codificacion          |
| `split_data.py`                       | Completo | Split estratificado 70/15/15 exportado a CSV           |
| `mlp_trainer.py`                      | Completo | Crear, entrenar, evaluar fitness y curva de perdida    |
| `metrics.py`                          | Completo | Accuracy, F1, matriz de confusion                      |
| `visualizer.py`                       | Completo | Graficas de evolucion y convergencia metaheurística   |
| `metaheuristicos/hiperparametros.py`  | Completo | Mapeo y límites matemáticos del espacio de búsqueda |
| `metaheuristicos/wolf.py`             | Completo | Genoma del individuo y mutación                       |
| `metaheuristicos/ga.py`               | Completo | Algoritmo Genético                                    |
| `metaheuristicos/gwo.py`              | Completo | Optimización por inteligencia de enjambre (Lobos)     |
| `metaheuristicos/paralelizador.py`    | Completo | Abstracción multi-hilo para evaluación simultánea   |
| `metaheuristicos/hybrid_optimizer.py` | Completo | Orquestacion del hibrido GWO-GA                        |
| `main.py`                             | Completo | Pipeline base y optimizado 100% funcional              |

## Configuracion rapida

```bash
cd "Proyectos finales/IA"
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Ejecutar solo el EDA

```bash
python src/analisis_exploratorio.py
```

### Ejecutar solo el preprocesado y split

```bash
python src/preprocessing.py
```

### Ejecutar el pipeline base completo (sin optimizador)

```bash
python src/main.py
```
