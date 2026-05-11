# Proyecto IA — Clasificación de Transacciones Blockchain

Optimización híbrida GWO-GA para una red neuronal de clasificación de 5 clases sobre datos de transacciones de criptomonedas.

## Estructura

```
IA/
├── venv/                  # Entorno virtual Python
├── dataset/               # CSV y datos crudos
├── sql/                   # Scripts SQL auxiliares
├── src/                   # Código fuente principal
│   ├── conexion_db.py     # Conexión a PostgreSQL
│   ├── preprocessing.py   # Carga y preprocesamiento
│   ├── labels.py          # Ingeniería de etiquetas (5 clases)
│   ├── train_test.py      # Split reproducible
│   ├── model.py           # Arquitectura de la red neuronal
│   ├── metrics.py         # Métricas de evaluación
│   ├── gwo.py             # Grey Wolf Optimizer
│   ├── ga.py              # Algoritmo Genético
│   └── hybrid_optimizer.py# Híbrido GWO-GA
├── notebooks/             # Exploración y análisis (Jupyter)
├── results/               # Métricas y reportes exportados
├── models/                # Modelos entrenados serializados
├── graphs/                # Gráficas generadas
├── requirements.txt
└── README.md
```

## Configuración rápida

```bash
# Crear entorno virtual (si no existe)
python -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Base de datos

- Motor: **PostgreSQL**
- Base: `examen_redes_neuronales`
- Tabla principal: `crypto_transactions`
- Credenciales en `src/conexion_db.py`
