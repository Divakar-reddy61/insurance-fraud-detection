from db_config import get_db_connection

def save_prediction(image_name, prediction, confidence):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO predictions (image_name, prediction, confidence) VALUES (%s, %s, %s)",
        (image_name, prediction, confidence)
    )

    conn.commit()
    cursor.close()
    conn.close()


def fetch_predictions():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM predictions ORDER BY timestamp DESC")
    data = cursor.fetchall()

    cursor.close()
    conn.close()
    return data
