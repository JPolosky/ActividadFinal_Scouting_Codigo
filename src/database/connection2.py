"""
Conector a la base de datos MySQL del proyecto (tabla players de Transfermarkt).

Los datos de conexión se toman de variables de entorno. Si no existen se usan
los del entorno local: MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB
"""
import os

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


class MySQLConnector:
    def __init__(self, host=None, port=None, user=None, password=None, database=None):
        url = URL.create(
            drivername="mysql+pymysql",
            username=user or os.getenv("MYSQL_USER", "scouting"),
            password=password or os.getenv("MYSQL_PASSWORD", "scouting123"),
            host=host or os.getenv("MYSQL_HOST", "127.0.0.1"),
            port=int(port or os.getenv("MYSQL_PORT", "3307")),
            database=database or os.getenv("MYSQL_DB", "scouting_db"),
            query={"charset": "utf8mb4"},
        )
        self.engine = create_engine(url, pool_pre_ping=True)

    def test_connection(self):
        # prueba rápida con SELECT 1
        with self.engine.connect() as conn:
            return conn.execute(text("SELECT 1")).scalar() == 1

    def extract_data(self, query):
        # ejecuta la consulta y devuelve un DataFrame
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        print(f"Datos extraídos: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df

    def close(self):
        self.engine.dispose()


if __name__ == "__main__":
    db = MySQLConnector()
    print("Conexión OK" if db.test_connection() else "Conexión fallida")
    print(db.extract_data("SELECT COUNT(*) AS jugadores FROM players"))
    db.close()
