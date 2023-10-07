from facenet_pytorch import InceptionResnetV1
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import torch

# Cargar el modelo InceptionResnetV1
encoder = InceptionResnetV1(pretrained='vggface2', classify=False).eval()

# Crear una instancia de MTCNN
mtcnn = MTCNN()

# Cargar la imagen
imagen = cv2.imread('img/IMG-20230827-WA0015.jpg')  # Reemplaza con la ubicación de tu imagen
cara = mtcnn.detect_faces(imagen)

# Asegúrate de que se haya detectado al menos una cara
if len(cara) > 0:
    # Extraer la región de la cara
    x, y, width, height = cara[0]['box']
    face = imagen[y:y+height, x:x+width]
    
    # Redimensionar la cara a 160x160
    face = cv2.resize(face, (160, 160))
    
    # Preprocesar la imagen para el modelo InceptionResnetV1
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = np.transpose(face, (2, 0, 1))
    face = face / 255.0
    face_tensor = torch.tensor(face, dtype=torch.float32)
    face_tensor = face_tensor.unsqueeze(0)

    # Generar el embedding de la cara
    embedding_cara = encoder(face_tensor)
    
    # Convertir el embedding de la cara en un arreglo NumPy
    embedding_array = embedding_cara.detach().numpy()

    # Convertir el arreglo NumPy en bytes
    descriptor_bytes = embedding_array.tobytes()
    
    # Ahora tienes el descriptor facial en formato de bytes
    print("Descriptor facial en formato de bytes:", descriptor_bytes)
else:
    print("No se detectaron caras en la imagen.")
