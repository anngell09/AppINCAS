import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face
import torch
import torch.nn.functional as F
import numpy as np

# Inicializar el modelo MTCNN para la detección de rostros
mtcnn = MTCNN()

# Inicializar el modelo InceptionResnetV1 para la extracción de características
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Cargar la imagen de referencia desde la base de datos
referencia_image = cv2.imread("AppINCAS/my_app/static/img/referencia.jpeg")
referencia_boxes, _ = mtcnn.detect(referencia_image)
referencia_face = extract_face(referencia_image, referencia_boxes[0])
referencia_face = cv2.cvtColor(referencia_face, cv2.COLOR_RGB2BGR)
referencia_face = torch.from_numpy(np.transpose(referencia_face, (2, 0, 1))).float()
referencia_embedding = resnet(referencia_face.unsqueeze(0))

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    # Capturar un frame de la cámara
    ret, frame = cap.read()

    # Detección de rostro y extracción de características en tiempo real
    boxes, _ = mtcnn.detect(frame)
    if boxes is not None:
        for box in boxes:
            # Extracción de características del rostro de la webcam
            face_webcam = extract_face(frame, box)

            # Asegurar que la imagen esté en formato BGR
            if face_webcam.shape[2] == 3:  # Si la imagen no está en BGR
                face_webcam = cv2.cvtColor(face_webcam, cv2.COLOR_RGB2BGR)

            # Asegurar que la imagen tiene 3 canales
            if face_webcam.shape[2] != 3:
                continue  # O maneja este caso de manera adecuada según tus necesidades

            # Dibujar un rectángulo alrededor del rostro
            (x, y, w, h) = [int(i) for i in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Preprocesamiento de la imagen para que coincida con las expectativas del modelo
            face_webcam = torch.from_numpy(np.transpose(face_webcam, (2, 0, 1))).float()

            # Extracción de características con el modelo InceptionResnetV1
            embedding_webcam = resnet(face_webcam.unsqueeze(0))

            # Comparar el rostro con la imagen de referencia
            similarity = F.cosine_similarity(embedding_webcam, referencia_embedding).item()

            # Imprimir mensaje si la similitud supera el umbral
            if similarity > 0.6:
                print("¡Coincidencia!")

    # Mostrar el frame resultante
    cv2.imshow("Reconocimiento Facial", frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
