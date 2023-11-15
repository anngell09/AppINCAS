import numpy as np
import cv2
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from Conect import create_connection

def extract_features(image_path = "AppINCAS/my_app/static/img/referencia.jpeg"):
    # Carga la imagen y preprocesa
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    # Carga el modelo preentrenado
    model = MobileNetV2(weights="imagenet", include_top=False)
    
    # Extrae las características
    features = model.predict(image)
    
    # Convierte las características a bytes
    features_bytes = features.tobytes()

    return features_bytes

def store_features(image_path):
    # Extrae las características
    features_bytes = extract_features(image_path)

    # Crea la conexión a la base de datos
    connection = create_connection()
    cursor = connection.cursor()

    # Inserta las características en la base de datos
    query = "INSERT INTO mytable (descriptors) VALUES (%s)"
    cursor.execute(query, (features_bytes,))

    # Cierra la conexión
    connection.commit()
    cursor.close()
    connection.close()
