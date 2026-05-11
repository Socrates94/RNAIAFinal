"""
importar_csv.py
===============
Script para importar crypto_scam_transaction_dataset.csv a PostgreSQL
y verificar que todo quedó correcto.

Uso:
    python3 importar_csv.py

Requisitos:
    pip install psycopg2-binary pandas sqlalchemy
"""

import os
import sys
import time
import pandas as pd
from sqlalchemy import create_engine, text

# ─────────────────────────────────────────────
#  CONFIGURACIÓN DE CONEXIÓN  ← edita si es necesario
# ─────────────────────────────────────────────
DB_CONFIG = {
    "host":     "",            # vacío = socket Unix (sin contraseña)
    "port":     5432,
    "database": "examen_redes_neuronales",
    "user":     "daniel",      # tu usuario de sistema
    "password": "",
}

CSV_PATH = os.path.join(os.path.dirname(__file__), "crypto_scam_transaction_dataset.csv")
TABLE_NAME = "crypto_transactions"

# ─────────────────────────────────────────────
#  TIPOS DE COLUMNA PARA PANDAS → POSTGRES
# ─────────────────────────────────────────────
DTYPE_MAP = {
    "transaction_id":                   "str",
    "timestamp":                        "int64",
    "blockchain":                       "str",
    "transaction_type":                 "str",
    "sender_wallet_age_days":           "int64",
    "receiver_wallet_age_days":         "int64",
    "transaction_amount_usd":           "float64",
    "gas_fee_usd":                      "float64",
    "token_type":                       "str",
    "platform":                         "str",
    "num_prev_transactions_sender":     "int64",
    "num_prev_transactions_receiver":   "int64",
    "avg_txn_interval_sender_min":      "float64",
    "is_cross_chain":                   "int64",
    "failed_txn_ratio_sender":          "float64",
    "velocity_score":                   "float64",
    "anomaly_score":                    "float64",
    "is_scam":                          "int64",
}


def get_engine():
    """Crea el engine de SQLAlchemy para PostgreSQL (socket Unix)."""
    if DB_CONFIG["host"]:
        pwd = f":{DB_CONFIG['password']}" if DB_CONFIG["password"] else ""
        url = (
            f"postgresql+psycopg2://{DB_CONFIG['user']}{pwd}"
            f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )
    else:
        # Conexión por socket Unix (peer auth) – sin contraseña
        url = (
            f"postgresql+psycopg2://{DB_CONFIG['user']}"
            f"@/{DB_CONFIG['database']}"
        )
    return create_engine(url, echo=False)


def crear_tabla(engine):
    """Crea la tabla si no existe."""
    ddl = """
    CREATE TABLE IF NOT EXISTS crypto_transactions (
        transaction_id                  VARCHAR(20)    PRIMARY KEY,
        timestamp                       BIGINT,
        blockchain                      VARCHAR(50),
        transaction_type                VARCHAR(50),
        sender_wallet_age_days          INTEGER,
        receiver_wallet_age_days        INTEGER,
        transaction_amount_usd          NUMERIC(15,2),
        gas_fee_usd                     NUMERIC(10,4),
        token_type                      VARCHAR(50),
        platform                        VARCHAR(100),
        num_prev_transactions_sender    INTEGER,
        num_prev_transactions_receiver  INTEGER,
        avg_txn_interval_sender_min     NUMERIC(10,2),
        is_cross_chain                  SMALLINT,
        failed_txn_ratio_sender         NUMERIC(5,4),
        velocity_score                  NUMERIC(8,4),
        anomaly_score                   NUMERIC(8,4),
        is_scam                         SMALLINT
    );
    CREATE INDEX IF NOT EXISTS idx_blockchain       ON crypto_transactions(blockchain);
    CREATE INDEX IF NOT EXISTS idx_transaction_type ON crypto_transactions(transaction_type);
    CREATE INDEX IF NOT EXISTS idx_is_scam          ON crypto_transactions(is_scam);
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
    print("✓ Tabla 'crypto_transactions' lista.")


def importar_csv(engine):
    """Lee el CSV e inserta los datos en la tabla."""
    print(f"→ Leyendo {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH, dtype=str)  # leemos todo como str primero

    # Convertir tipos correctamente
    for col, dtype in DTYPE_MAP.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    total = len(df)
    print(f"→ {total:,} filas leídas. Importando a PostgreSQL...")

    t0 = time.time()
    df.to_sql(
        TABLE_NAME,
        engine,
        if_exists="replace",   # reemplaza si ya existe data
        index=False,
        chunksize=500,
        method="multi",
    )
    elapsed = time.time() - t0
    print(f"✓ Importación completada en {elapsed:.1f}s  ({total:,} registros).")
    return df


def verificar(engine):
    """Muestra estadísticas básicas de la tabla importada."""
    with engine.connect() as conn:
        total      = conn.execute(text("SELECT COUNT(*) FROM crypto_transactions")).scalar()
        scams      = conn.execute(text("SELECT COUNT(*) FROM crypto_transactions WHERE is_scam = 1")).scalar()
        legit      = conn.execute(text("SELECT COUNT(*) FROM crypto_transactions WHERE is_scam = 0")).scalar()
        blockchains = conn.execute(text("SELECT blockchain, COUNT(*) AS n FROM crypto_transactions GROUP BY blockchain ORDER BY n DESC")).fetchall()

    print("\n" + "="*50)
    print("  RESUMEN DE LA BASE DE DATOS")
    print("="*50)
    print(f"  Total de transacciones : {total:>8,}")
    print(f"  Transacciones SCAM     : {scams:>8,}  ({scams/total*100:.1f}%)")
    print(f"  Transacciones legítimas: {legit:>8,}  ({legit/total*100:.1f}%)")
    print("\n  Distribución por blockchain:")
    for row in blockchains:
        print(f"    {row[0]:<20} {row[1]:>6,}")
    print("="*50)


if __name__ == "__main__":
    print("\n🚀 Importador CSV → PostgreSQL")
    print(f"   Base de datos : {DB_CONFIG['database']}")
    print(f"   Tabla destino : {TABLE_NAME}\n")

    try:
        engine = get_engine()
        crear_tabla(engine)
        importar_csv(engine)
        verificar(engine)
        print("\n✅ ¡Todo listo! Conéctate con: psql -d examen_redes_neuronales")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPosibles soluciones:")
        print("  1. Verifica que PostgreSQL esté corriendo: sudo systemctl start postgresql")
        print("  2. Edita DB_CONFIG en este script con tu usuario/contraseña correctos")
        print("  3. Crea la BD manualmente: createdb examen_redes_neuronales")
        sys.exit(1)
