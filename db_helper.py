"""
Módulo para conexión y manipulación de datos en PostgreSQL.
"""

import pandas as pd
from sqlalchemy import create_engine, text

DB_CONFIG = {
    "host":     "",
    "port":     5432,
    "database": "examen_redes_neuronales",
    "user":     "daniel",
    "password": "",
}

def _build_url():
    """Genera la URL de conexión para SQLAlchemy."""
    if DB_CONFIG["host"]:
        pwd = f":{DB_CONFIG['password']}" if DB_CONFIG["password"] else ""
        return (
            f"postgresql+psycopg2://{DB_CONFIG['user']}{pwd}"
            f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )
    return f"postgresql+psycopg2://{DB_CONFIG['user']}@/{DB_CONFIG['database']}"

engine = create_engine(_build_url(), echo=False)

def get_df(only_scam: bool = False, only_legit: bool = False, limit: int = None) -> pd.DataFrame:
    """
    Retorna el dataset como DataFrame de pandas.

    Args:
        only_scam: Filtrar solo transacciones fraudulentas.
        only_legit: Filtrar solo transacciones legítimas.
        limit: Límite de registros.
    """
    where = ""
    if only_scam:
        where = "WHERE is_scam = 1"
    elif only_legit:
        where = "WHERE is_scam = 0"

    lim = f"LIMIT {limit}" if limit else ""
    sql = f"SELECT * FROM crypto_transactions {where} {lim}"
    return pd.read_sql(sql, engine)

def query(sql: str) -> pd.DataFrame:
    """Ejecuta una consulta SQL y devuelve un DataFrame."""
    return pd.read_sql(sql, engine)

def get_features_and_target():
    """Retorna X (features) e y (target) para entrenamiento."""
    df = get_df()

    feature_cols = [
        "sender_wallet_age_days",
        "receiver_wallet_age_days",
        "transaction_amount_usd",
        "gas_fee_usd",
        "num_prev_transactions_sender",
        "num_prev_transactions_receiver",
        "avg_txn_interval_sender_min",
        "is_cross_chain",
        "failed_txn_ratio_sender",
        "velocity_score",
        "anomaly_score",
    ]

    X = df[feature_cols]
    y = df["is_scam"]
    return X, y

def get_stats() -> dict:
    """Retorna estadísticas de la tabla crypto_transactions."""
    with engine.connect() as conn:
        total  = conn.execute(text("SELECT COUNT(*) FROM crypto_transactions")).scalar()
        scams  = conn.execute(text("SELECT COUNT(*) FROM crypto_transactions WHERE is_scam=1")).scalar()
        legit  = conn.execute(text("SELECT COUNT(*) FROM crypto_transactions WHERE is_scam=0")).scalar()
    return {"total": total, "scams": scams, "legit": legit,
            "scam_pct": round(scams / total * 100, 2)}

if __name__ == "__main__":
    stats = get_stats()
    print(f"[INFO] Total: {stats['total']:,}")
    print(f"[INFO] Scams: {stats['scams']:,} ({stats['scam_pct']}%)")
    print(f"[INFO] Legit: {stats['legit']:,}")
    X, y = get_features_and_target()
    print(f"[INFO] X shape: {X.shape}")
    print(f"[INFO] Target distribution: {y.value_counts().to_dict()}")

