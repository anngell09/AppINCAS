from facenet_pytorch import InceptionResnetV1
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import torch

# Cargar el modelo InceptionResnetV1
encoder = InceptionResnetV1(pretrained='vggface2', classify=False).eval()

# Crear una instancia de MTCNN
mtcnn = MTCNN()

# Cargar la imagen de referencia
imagen_referencia = cv2.imread('img/IMG-20230827-WA0002.jpg')  # Reemplaza con la ubicación de la imagen de referencia

# Detección de caras en la imagen de referencia
cara_referencia = mtcnn.detect_faces(imagen_referencia)

# Asegúrate de que se haya detectado al menos una cara en la imagen de referencia
if len(cara_referencia) > 0:
    # Extraer la región de la cara en la imagen de referencia
    x1, y1, width1, height1 = cara_referencia[0]['box']
    face_referencia = imagen_referencia[y1:y1+height1, x1:x1+width1]

    # Redimensionar la cara de referencia a 160x160
    face_referencia = cv2.resize(face_referencia, (160, 160))

    # Preprocesar la imagen de referencia para el modelo InceptionResnetV1
    face_referencia = cv2.cvtColor(face_referencia, cv2.COLOR_BGR2RGB)
    face_referencia = np.transpose(face_referencia, (2, 0, 1))
    face_referencia = face_referencia / 255.0
    face_referencia_tensor = torch.tensor(face_referencia, dtype=torch.float32)
    face_referencia_tensor = face_referencia_tensor.unsqueeze(0)

    # Iniciar la cámara web
    cap = cv2.VideoCapture(0)  # Usar la cámara web (0 por defecto)

    while True:
        # Capturar un fotograma desde la cámara
        ret, frame = cap.read()

        # Detección de caras con MTCNN en el fotograma capturado
        caras = mtcnn.detect_faces(frame)

        if len(caras) > 0:
            # Extraer la región de la cara en el fotograma
            x2, y2, width2, height2 = caras[0]['box']
            face_actual = frame[y2:y2+height2, x2:x2+width2]

            # Redimensionar la cara capturada a 160x160
            face_actual = cv2.resize(face_actual, (160, 160))

            # Preprocesar la imagen capturada para el modelo InceptionResnetV1
            face_actual = cv2.cvtColor(face_actual, cv2.COLOR_BGR2RGB)
            face_actual = np.transpose(face_actual, (2, 0, 1))
            face_actual = face_actual / 255.0
            face_actual_tensor = torch.tensor(face_actual, dtype=torch.float32)
            face_actual_tensor = face_actual_tensor.unsqueeze(0)

            # Generar los embeddings de la cara de referencia y la cara capturada
            embedding_referencia = encoder(face_referencia_tensor)
            embedding_actual = encoder(face_actual_tensor)

            # Calcular la distancia euclidiana entre los embeddings
            distancia_euclidiana = torch.norm(embedding_referencia - embedding_actual)

            # Establecer un umbral de similitud
            umbral_similitud = 1.0  # Puedes ajustar este valor según tus necesidades

            # Comparar la distancia con el umbral
            if distancia_euclidiana < umbral_similitud:
                print("El rostro capturado es similar al de la imagen de referencia.")
            else:
                print("El rostro capturado es diferente al de la imagen de referencia.")

        # Mostrar el fotograma con la detección de caras
        cv2.imshow('Face Recognition', frame)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la cámara y cerrar la ventana
    cap.release()
    cv2.destroyAllWindows()
else:
    print("No se detectó una cara en la imagen de referencia.")
