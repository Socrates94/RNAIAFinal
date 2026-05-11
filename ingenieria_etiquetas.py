"""
Ingeniería de etiquetas (Label Engineering) para clasificación multiclase.
Define 5 clases basadas en reglas técnicas y guarda resultados en PostgreSQL y CSV.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")

engine = create_engine("postgresql+psycopg2://daniel@/examen_redes_neuronales", echo=False)
OUT_EDA = os.path.join(os.path.dirname(__file__), "resultados", "eda")
OUT_DATA = os.path.join(os.path.dirname(__file__), "resultados")
os.makedirs(OUT_EDA, exist_ok=True)
os.makedirs(OUT_DATA, exist_ok=True)

CLASS_NAMES = {
    0: "Normal transaction",
    1: "Scam transaction",
    2: "High-frequency activity",
    3: "Whale movement",
    4: "Suspicious behavior",
}

NUMERIC_FEATURES = [
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

print("[INFO] Iniciando ingeniería de etiquetas")

df = pd.read_sql("SELECT * FROM crypto_transactions", engine)
print(f"[INFO] Registros cargados: {len(df):,}")

nulos_antes = df["avg_txn_interval_sender_min"].isnull().sum()
if nulos_antes > 0:
    mediana = df["avg_txn_interval_sender_min"].median()
    df["avg_txn_interval_sender_min"] = df["avg_txn_interval_sender_min"].fillna(mediana)
    print(f"[INFO] Imputados {nulos_antes} nulos en avg_txn_interval_sender_min con mediana {mediana:.2f}")

P99_amount   = df["transaction_amount_usd"].quantile(0.99)
P10_interval = df["avg_txn_interval_sender_min"].quantile(0.10)
P90_velocity = df["velocity_score"].quantile(0.90)
P75_anomaly  = df["anomaly_score"].quantile(0.75)
WALLET_AGE_THRESHOLD = 90

print("[INFO] Umbrales calculados:")
print(f"    amount_usd P99: {P99_amount:.2f}")
print(f"    interval P10: {P10_interval:.2f}")
print(f"    velocity P90: {P90_velocity:.4f}")
print(f"    anomaly P75: {P75_anomaly:.4f}")

label = pd.Series(np.nan, index=df.index)

mask_whale = df["transaction_amount_usd"] > P99_amount
label[mask_whale] = 3

mask_scam = (label.isna()) & (df["is_scam"] == 1)
label[mask_scam] = 1

mask_highfreq = (label.isna()) & (
    (df["avg_txn_interval_sender_min"] < P10_interval) |
    (df["velocity_score"] > P90_velocity)
)
label[mask_highfreq] = 2

mask_suspicious = (label.isna()) & (
    (df["anomaly_score"] > P75_anomaly) &
    (df["sender_wallet_age_days"] < WALLET_AGE_THRESHOLD)
)
label[mask_suspicious] = 4

label[label.isna()] = 0
df["label"] = label.astype(int)

print("[INFO] Distribucion de clases:")
total = len(df)
dist = df["label"].value_counts().sort_index()
for cls, cnt in dist.items():
    print(f"    {cls} - {CLASS_NAMES[cls]}: {cnt:,} ({cnt/total*100:.1f}%)")

for cls, cnt in dist.items():
    if cnt / total < 0.03:
        print(f"[WARN] Clase {cls} representa menos del 3% del dataset")

df_labeled = df[["transaction_id"] + NUMERIC_FEATURES + ["label"]].copy()

df_labeled.to_sql("crypto_labeled", engine, if_exists="replace", index=False, chunksize=500, method="multi")
print(f"[INFO] Tabla 'crypto_labeled' actualizada")

csv_path = os.path.join(OUT_DATA, "crypto_labeled.csv")
df_labeled.to_csv(csv_path, index=False)
print(f"[INFO] Dataset exportado a {csv_path}")

colores = ["#4CAF50", "#F44336", "#FF9800", "#9C27B0", "#2196F3"]
labels_str = [f"{k} — {v}" for k, v in CLASS_NAMES.items()]
valores = [dist.get(k, 0) for k in CLASS_NAMES.keys()]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

bars = ax1.bar(range(5), valores, color=colores, edgecolor="white", width=0.6)
ax1.set_xticks(range(5))
ax1.set_xticklabels([f"Clase {k}" for k in CLASS_NAMES.keys()], rotation=15, ha="right")
ax1.set_title("Distribucion de clases (conteo)")
ax1.set_ylabel("Registros")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
for bar, val in zip(bars, valores):
    ax1.annotate(f"{val:,}\n({val/total*100:.1f}%)", (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center", va="bottom", fontsize=8)

ax2.pie(valores, labels=labels_str, colors=colores, autopct="%1.1f%%", startangle=140, textprops={"fontsize": 8}, wedgeprops={"edgecolor": "white", "linewidth": 1.5})
ax2.set_title("Distribucion porcentual")

fig.suptitle("Distribucion de Clases Post-Ingenieria")
plt.tight_layout()
fig_path = os.path.join(OUT_EDA, "fig6_distribucion_clases.png")
fig.savefig(fig_path, bbox_inches="tight")
plt.close()

print("[INFO] Ingeniería de etiquetas completada")

