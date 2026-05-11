from conexion_db import get_connection
import random
import string
import statistics as stats
import matplotlib.pyplot as plt
import os

ARCHIVO_NOMBRE = "parrafo.txt"
POBLACION_TAMANO = 400
TASA_MUTACION = 0.01

ALFABETO = string.ascii_uppercase + " ,.!?"

def cargar_objetivo_desde_archivo(ruta):
    """Carga y normaliza el texto objetivo desde un archivo."""
    if not os.path.exists(ruta):
        with open(ruta, "w") as f:
            f.write("HOLA ESTE ES UN PARRAFO DE PRUEBA PARA EL ALGORITMO GENETICO.")
    
    with open(ruta, "r") as f:
        contenido = f.read().strip().upper()
        contenido = ''.join([c for c in contenido if c in ALFABETO])
        return contenido

OBJETIVO = cargar_objetivo_desde_archivo(ARCHIVO_NOMBRE)
LONGITUD_OBJETIVO = len(OBJETIVO)

def generar_individuo(longitud):
    """Genera una cadena aleatoria de longitud especificada."""
    return ''.join(random.choices(ALFABETO, k=longitud))

def calcular_fitness(individuo):
    """Calcula la cantidad de caracteres que coinciden con el objetivo."""
    return sum(1 for a, b in zip(individuo, OBJETIVO) if a == b)

def cruza(padre1, padre2):
    """Realiza cruce de un punto entre dos individuos."""
    punto = random.randint(1, len(OBJETIVO) - 1)
    return padre1[:punto] + padre2[punto:]

def mutar(individuo):
    """Aplica mutaciones aleatorias a un individuo según TASA_MUTACION."""
    lista = list(individuo)
    for i in range(len(lista)):
        if random.random() < TASA_MUTACION:
            lista[i] = random.choice(ALFABETO)    
    return ''.join(lista)

if __name__ == "__main__":
    print(f"[INFO] Archivo: {ARCHIVO_NOMBRE}")
    print(f"[INFO] Objetivo: {OBJETIVO}")

    estad_max = []
    estad_prom = []

    poblacion = [generar_individuo(LONGITUD_OBJETIVO) for _ in range(POBLACION_TAMANO)]
    mejor_individuo = ""
    generacion = 0

    try:
        while mejor_individuo != OBJETIVO:
            generacion += 1
            
            fitness_vals = [calcular_fitness(ind) for ind in poblacion]
            poblacion = sorted(poblacion, key=lambda x: calcular_fitness(x), reverse=True)
            mejor_individuo = poblacion[0]
            
            estad_max.append(max(fitness_vals))
            estad_prom.append(stats.mean(fitness_vals))
            
            if generacion % 20 == 0 or mejor_individuo == OBJETIVO:
                print(f"[PROGRESS] Gen {generacion:04d} | Fitness: {max(fitness_vals)}/{LONGITUD_OBJETIVO} | {mejor_individuo[:50]}...")

            if mejor_individuo == OBJETIVO:
                break

            elite = poblacion[:POBLACION_TAMANO//4]
            nueva_poblacion = elite[:]
            
            while len(nueva_poblacion) < POBLACION_TAMANO:
                p1, p2 = random.sample(elite, 2)
                nueva_poblacion.append(mutar(cruza(p1, p2)))
            
            poblacion = nueva_poblacion

    except KeyboardInterrupt:
        print("\n[WARN] Proceso interrumpido")

    print(f"[SUCCESS] Generacion: {generacion}")

    plt.figure(figsize=(10, 5))
    plt.plot(estad_max, label="Mejor puntuacion")
    plt.plot(estad_prom, label="Promedio", linestyle="--")
    plt.axhline(y=LONGITUD_OBJETIVO, color='r', linestyle=':', label="Meta")
    plt.title("Evolucion Algoritmo Genetico")
    plt.xlabel("Generacion")
    plt.ylabel("Fitness")
    plt.legend()
    plt.show()