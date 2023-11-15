import cv2
import numpy as np
from deepface import basemodels
from database_connection import create_connection


def extract_features_from_image(image):
    # Carga la imagen
    image = cv2.imread("AppINCAS/my_app/static/img/referencia.jpeg")

    # Extrae las características
    embedding = basemodels.represent(img_path=image, model_name='Facenet.py', enforce_detection=False)

    # Convierte las características a bytes
    features_bytes = embedding.tobytes()

    # Crea la conexión a la base de datos
    connection = create_connection()
    cursor = connection.cursor()

    # Inserta las características en la base de datos
    query = "INSERT INTO mytable (descriptors) VALUES (%s)"
    cursor.execute(query, (features_bytes,))

    if cursor.rowcount > 0:
        print("Insercción a la base de datos exitosa")

    else:

         print("La inserción de datos no fue exitosa")

    # Cierra la conexión
    cursor.close()
    connection.close()
