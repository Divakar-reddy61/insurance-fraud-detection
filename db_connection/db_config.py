import mysql.connector

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Divakar@8601",
        database="insurance_db"
    )
