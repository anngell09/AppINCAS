import cv2
import numpy as np
from deepface import basemodels
from database_connection import create_connection

def compare_features():
    # Inicia la captura de video
    video_capture = cv2.VideoCapture(0)

    while True:
        # Captura un frame
        ret, frame = video_capture.read()

        # Extrae las características
        embedding = basemodels.represent(img_path=frame, model_name='Facenet.py', enforce_detection=False)

        # Convierte las características a bytes
        features_bytes_webcam = embedding.tobytes()

        # Crea la conexión a la base de datos
        connection = create_connection()
        cursor = connection.cursor()

        # Obtiene las características de la base de datos
        query = "SELECT descriptors FROM mytable"
        cursor.execute(query)
        results = cursor.fetchall()

        for row in results:
            # Convierte las características de bytes a numpy array
            features_bytes_db = row[0]
            features_db = np.frombuffer(features_bytes_db, dtype=np.float32)

            # Compara las características
            if np.array_equal(features_bytes_webcam, features_db):
                print("¡Persona reconocida!")

        # Cierra la conexión
        cursor.close()
        connection.close()
