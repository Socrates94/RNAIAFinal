import psycopg2

conexion = psycopg2.connect(
    user="daniel",
    password='daniel',
    host='127.0.0.1',
    port='5432',
    database="examen_redes_neuronales"
)

try:
    with conexion:
        with conexion.cursor() as cursor:
            sentencia = "SELECT platform FROM crypto_transactions LIMIT 20"
            cursor.execute(sentencia)
            registros = cursor.fetchall()
            for fila in registros:
                print(fila)
except Exception as e:
    print(f"[ERROR] {e}")
finally:
    cursor.close()
    conexion.close()