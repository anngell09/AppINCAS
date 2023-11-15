import cv2
import tensorflow as tf
import numpy as np
from mtcnn import MTCNN

# Carga el modelo preentrenado de TensorFlow
model = tf.keras.applications.EfficientNetB0(include_top=False, pooling='avg')

# Inicializa el detector de rostros MTCNN
detector = MTCNN()

# Función para extraer descriptores faciales
def extract_features(face):
    face = cv2.resize(face, (224, 224))
    face = face / 255.0
    face = np.expand_dims(face, axis=0)
    features = model.predict(face)
    return features

# Función para detectar rostros y extraer descriptores faciales
def process_frame(frame):
    # Detecta rostros en el frame
    result = detector.detect_faces(frame)
    
    if result:
        # Extrae el primer rostro detectado
        bounding_box = result[0]['box']
        x, y, width, height = bounding_box
        face = frame[y:y+height, x:x+width]
        
        # Extrae los descriptores faciales del rostro
        face_features = extract_features(face)
        
        return face_features
    else:
        return None

# Carga la imagen y extrae los descriptores faciales
image = cv2.imread('AppINCAS/my_app/static/img/referencia.jpeg')
image_features = process_frame(image)

# Inicia la captura de video desde la webcam
cap = cv2.VideoCapture(0)

while True:
    # Captura un frame del video
    ret, frame = cap.read()

    # Extrae los descriptores faciales del frame
    frame_features = process_frame(frame)

    if frame_features is not None:
        # Compara los descriptores faciales de la imagen y el frame
        similarity = np.dot(image_features, frame_features.T)

        # Muestra la similitud en la ventana del video
        cv2.putText(frame, f'Similarity: {similarity}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Video', frame)

    # Rompe el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera los recursos y cierra las ventanas
cap.release()
cv2.destroyAllWindows()
