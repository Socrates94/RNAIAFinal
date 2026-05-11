"""
Análisis Exploratorio de Datos (EDA) para clasificación de transacciones.
Genera estadísticas descriptivas y visualizaciones en resultados/eda/.
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sqlalchemy import create_engine

warnings.filterwarnings("ignore")

engine = create_engine("postgresql+psycopg2://daniel@/examen_redes_neuronales", echo=False)
OUT_DIR = os.path.join(os.path.dirname(__file__), "resultados", "eda")
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})

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

print("[INFO] Iniciando EDA")

df = pd.read_sql("SELECT * FROM crypto_transactions", engine)

print(f"[INFO] Dimensiones: {df.shape}")
print("[INFO] Tipos de datos:")
print(df.dtypes.to_string())

nulos = df.isnull().sum()
nulos_pct = (nulos / len(df) * 100).round(2)
nulos_df = pd.DataFrame({"Nulos": nulos, "Porcentaje (%)": nulos_pct})
nulos_df = nulos_df[nulos_df["Nulos"] > 0]
if nulos_df.empty:
    print("[INFO] Sin valores nulos")
else:
    print("[WARN] Valores nulos detectados:")
    print(nulos_df.to_string())

dups = df.duplicated(subset="transaction_id").sum()
print(f"[INFO] Duplicados transaction_id: {dups}")

print("[INFO] Distribucion is_scam:")
vc = df["is_scam"].value_counts()
for val, cnt in vc.items():
    label = "Scam" if val == 1 else "Legitima"
    print(f"    {val} ({label}): {cnt:>6,}  ({cnt/len(df)*100:.1f}%)")

print("[INFO] Estadisticas descriptivas:")
desc = df[NUMERIC_FEATURES].describe().T
desc["missing%"] = (df[NUMERIC_FEATURES].isnull().sum() / len(df) * 100).round(2)
print(desc.to_string())

fig, ax = plt.subplots(figsize=(5, 4))
vc.plot(kind="bar", ax=ax, color=["#4CAF50", "#F44336"], edgecolor="white", width=0.5)
ax.set_title("Distribucion: is_scam")
ax.set_xticklabels(["0 — Legitima", "1 — Scam"], rotation=0)
for bar in ax.patches:
    ax.annotate(f"{int(bar.get_height()):,}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig1_distribucion_is_scam.png"))
plt.close()

fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()
for i, col in enumerate(NUMERIC_FEATURES):
    data = df[col].dropna()
    axes[i].hist(data, bins=40, color="#5C6BC0", edgecolor="white", alpha=0.85)
    axes[i].set_title(col, fontsize=8)
    axes[i].set_ylabel("Frecuencia")
    axes[i].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
for j in range(len(NUMERIC_FEATURES), len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Histogramas Features Numericas")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig2_histogramas_features.png"), bbox_inches="tight")
plt.close()

corr = df[NUMERIC_FEATURES + ["is_scam"]].corr()
fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5, ax=ax)
ax.set_title("Matriz de Correlacion")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig3_heatmap_correlacion.png"))
plt.close()

key_features = ["transaction_amount_usd", "velocity_score", "anomaly_score", "failed_txn_ratio_sender", "avg_txn_interval_sender_min", "sender_wallet_age_days"]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
df_plot = df.copy()
df_plot["is_scam_label"] = df_plot["is_scam"].map({0: "Legitima", 1: "Scam"})
for i, col in enumerate(key_features):
    sns.boxplot(data=df_plot, x="is_scam_label", y=col, palette=["#4CAF50", "#F44336"], ax=axes[i], showfliers=False)
    axes[i].set_title(col, fontsize=8)
    axes[i].set_xlabel("")
fig.suptitle("Distribucion features clave por clase")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig4_boxplots_clases.png"))
plt.close()

print("[INFO] Percentiles clave:")
p10_interval = df["avg_txn_interval_sender_min"].quantile(0.10)
p90_velocity  = df["velocity_score"].quantile(0.90)
p90_amount    = df["transaction_amount_usd"].quantile(0.90)
p99_amount    = df["transaction_amount_usd"].quantile(0.99)
p90_anomaly   = df["anomaly_score"].quantile(0.90)

print(f"    amount_usd P90: {p90_amount:.2f}")
print(f"    amount_usd P99: {p99_amount:.2f}")
print(f"    interval P10: {p10_interval:.2f}")
print(f"    velocity P90: {p90_velocity:.4f}")
print(f"    anomaly P90: {p90_anomaly:.4f}")

print("[INFO] EDA completado")

