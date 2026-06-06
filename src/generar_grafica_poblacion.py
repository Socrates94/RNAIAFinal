import os
import matplotlib.pyplot as plt
import numpy as np

# Datos empíricos corregidos
poblaciones = ["N = 20 (Paralelo)", "N = 40 (Paralelo)"]
tiempos_minutos = [18, 120]  # Se corrigió de 60 a 120 minutos (2 horas)
accuracy = [93.44, 93.10]

fig, ax1 = plt.subplots(figsize=(8, 5))

# Eje izquierdo para el Tiempo
color_tiempo = 'tab:red'
ax1.set_xlabel('Tamaño de la Población (AG)', fontsize=12)
ax1.set_ylabel('Tiempo de Convergencia (Minutos)', color=color_tiempo, fontsize=12)
barras = ax1.bar(poblaciones, tiempos_minutos, color=color_tiempo, alpha=0.6, width=0.4, label="Tiempo")
ax1.tick_params(axis='y', labelcolor=color_tiempo)
ax1.set_ylim(0, 150)

# Eje derecho para el Accuracy
ax2 = ax1.twinx()
color_acc = 'tab:blue'
ax2.set_ylabel('Exactitud / Accuracy (%)', color=color_acc, fontsize=12)
linea = ax2.plot(poblaciones, accuracy, color=color_acc, marker='o', linewidth=3, markersize=10, label="Accuracy")
ax2.tick_params(axis='y', labelcolor=color_acc)
ax2.set_ylim(92.0, 94.0)

# Etiquetas en las barras de tiempo
for bar in barras:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval} min", ha='center', va='bottom', color=color_tiempo, fontweight='bold')

# Etiquetas en la línea de accuracy
for i, acc in enumerate(accuracy):
    ax2.text(i, acc + 0.1, f"{acc}%", ha='center', va='bottom', color=color_acc, fontweight='bold')

plt.title("Impacto del Tamaño de Población: Rendimiento vs Tiempo Computacional", fontsize=14, pad=15)
fig.tight_layout()

ruta_salida = os.path.join("Graficas", "Resultados", "comparativa_poblacion.png")
plt.savefig(ruta_salida, dpi=300)
print(f"[INFO] Gráfica guardada en: {ruta_salida}")
