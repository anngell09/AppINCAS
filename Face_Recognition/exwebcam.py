import cv2
import numpy as np
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from Conect import create_connection

def extract_features_from_webcam():
    # Inicia la captura de video
    video_capture = cv2.VideoCapture(0)

    while True:
        # Captura un frame
        ret, frame = video_capture.read()

        # Preprocesa el frame
        frame = cv2.resize(frame, (224, 224))
        frame = img_to_array(frame)
        frame = preprocess_input(frame)
        frame = np.expand_dims(frame, axis=0)

        # Carga el modelo preentrenado
        model = MobileNetV2(weights="imagenet", include_top=False)

        # Extrae las características
        features = model.predict(frame)

        # Convierte las características a bytes
        features_bytes = features.tobytes()

        return features_bytes

def compare_features():
    # Extrae las características de la webcam
    features_bytes_webcam = extract_features_from_webcam()

    # Crea la conexión a la base de datos
    connection = create_connection()
    cursor = connection.cursor()

    # Obtiene las características de la base de datos
    query = "SELECT features FROM your_table"
    cursor.execute(query)
    results = cursor.fetchall()

    recognized = False

    for row in results:
        # Convierte las características de bytes a numpy array
        features_bytes_db = row[0]
        features_db = np.frombuffer(features_bytes_db, dtype=np.float32)

        # Compara las características
        if np.array_equal(features_bytes_webcam, features_db):
            print("¡Persona reconocida!")
            recognized = True
            break

    if not recognized:
        print("Persona desconocida")

    # Cierra la conexión
    cursor.close()
    connection.close()

