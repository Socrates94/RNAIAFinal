# Proyecto IA — Clasificacion de Variedades de Frijol (Dry Bean Dataset)

Optimizacion hibrida GWO-GA aplicada a un MLPClassifier de scikit-learn para clasificar 7 variedades
de frijol seco usando el Dry Bean Dataset (UCI). El optimizador busca los mejores hiperparametros
de la red neuronal evaluando el fitness sobre el conjunto de validacion.

## Estructura del proyecto

```
IA/
├── DryBeanDataset/
│   └── Dry_Bean_Dataset.csv          # 13 611 muestras, 17 columnas
│
├── Graficas/
│   ├── EDA/                          # Graficas del analisis exploratorio
│   │   ├── distribucion_clases.png
│   │   ├── histogramas_features.png
│   │   ├── heatmap_correlacion.png
│   │   └── boxplots_por_clase.png
│   └── Resultados/                   # Graficas del optimizador (punto 4 del examen)
│       ├── evolucion_fitness.png
│       └── tendencia_central.png
│
├── src/
│   ├── main.py                       # Orquesta el pipeline completo
│   ├── analisis_exploratorio.py      # EDA: estadisticas, histogramas, heatmap, boxplots
│   ├── preprocessing.py              # Carga del CSV, StandardScaler, LabelEncoder
│   ├── split_data.py                 # Split estratificado 70/15/15 (train/val/test)
│   ├── mlp_trainer.py                # MLPClassifier: crear, entrenar, evaluar_fitness, curva de perdida
│   ├── metrics.py                    # Accuracy, F1-macro, F1-weighted, matriz de confusion
│   ├── wolf.py                       # (pendiente) Individuo/lobo: genes = hiperparametros del MLP
│   ├── ga.py                         # (pendiente) Algoritmo Genetico: cruce y mutacion
│   ├── gwo.py                        # (pendiente) Grey Wolf Optimizer
│   ├── hybrid_optimizer.py           # (pendiente) Metaheuristica hibrida GWO-GA
│   └── visualizer.py                 # Graficas de evolucion del optimizador → Graficas/Resultados/
│
├── venv/                             # Entorno virtual Python (no versionado)
├── requirements.txt
├── .gitignore
└── README.md
```

## Flujo de ejecucion

```
main.py
  │
  ├─ 1 ─► analisis_exploratorio.py   →  Graficas/EDA/
  ├─ 2 ─► preprocessing.py           →  X_scaled, y_encoded, le
  ├─ 3 ─► split_data.py              →  X_train / X_val / X_test  (70/15/15, estratificado)
  ├─ 4 ─► hybrid_optimizer.py        →  mejores hiperparametros
  │           ├── gwo.py
  │           ├── ga.py
  │           ├── wolf.py
  │           └── mlp_trainer.evaluar_fitness()  ← fitness = accuracy en X_val
  ├─ 5 ─► mlp_trainer.entrenar()     ←  entrena con X_train usando los mejores hiperparametros
  ├─ 6 ─► metrics.evaluar()          →  reporte final sobre X_test (sellado)
  └─ 7 ─► visualizer.py              →  Graficas/Resultados/
```

## Dataset

| Propiedad    | Detalle                                                        |
|--------------|----------------------------------------------------------------|
| Fuente       | UCI Machine Learning Repository                                |
| Muestras     | 13 611 (sin duplicados)                                        |
| Features     | 16 morfologicas (area, perimetro, compacidad...)               |
| Clases       | 7 (BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA)      |
| Desbalance   | Leve; DERMASON es la clase mayoritaria                         |
| Preprocesado | StandardScaler en X, LabelEncoder en y                         |

## Estado de implementacion

| Archivo                   | Estado    | Descripcion                                           |
|---------------------------|-----------|-------------------------------------------------------|
| `analisis_exploratorio.py`| Completo  | EDA con 4 graficas exportadas                         |
| `preprocessing.py`        | Completo  | Carga, limpieza y normalizacion                       |
| `split_data.py`           | Completo  | Split estratificado 70/15/15 reproducible             |
| `mlp_trainer.py`          | Completo  | Crear, entrenar, evaluar_fitness, curva de perdida    |
| `metrics.py`              | Completo  | Accuracy, F1, matriz de confusion                     |
| `visualizer.py`           | Completo  | Graficas de evolucion y tendencia central             |
| `wolf.py`                 | Pendiente | Representacion del individuo (vector de hiperparametros) |
| `ga.py`                   | Pendiente | Operadores geneticos (cruce, mutacion, seleccion)     |
| `gwo.py`                  | Pendiente | Grey Wolf Optimizer                                   |
| `hybrid_optimizer.py`     | Pendiente | Orquestacion del hibrido GWO-GA                       |
| `main.py`                 | Pendiente | Pipeline principal                                    |

## Configuracion rapida

```bash
cd "Proyectos finales/IA"
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Solo EDA
python src/analisis_exploratorio.py

# Pipeline completo (cuando main.py este listo)
python src/main.py
```
