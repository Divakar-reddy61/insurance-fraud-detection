import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Divakar@8601",
    database="insurance_db"
)

if conn.is_connected():
    print("✅ MySQL Connected Successfully")

conn.close()
