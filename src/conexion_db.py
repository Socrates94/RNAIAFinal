import psycopg2

def get_connection():
    """Retorna una conexión activa a la base de datos."""
    return psycopg2.connect(
        user="daniel",
        password="daniel",
        host="127.0.0.1",
        port="5432",
        database="examen_redes_neuronales"
    )
