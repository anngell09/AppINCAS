import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import mysql.connector
from PIL import Image

# Cargar el modelo preentrenado
model = models.resnet50(pretrained=True)

# Usar la capa penúltima para extraer descriptores
model = torch.nn.Sequential(*(list(model.children())[:-1]))

# Asegurarse de que el modelo esté en modo de evaluación
model.eval()

# Cargar la imagen del rostro
img = Image.open('AppINCAS/my_app/static/img/referencia.jpeg')

# Preprocesar la imagen
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0)

# Extraer descriptores faciales
with torch.no_grad():
    descriptors = model(input_batch)

# Convertir los descriptores a un tensor de numpy
descriptors_np = descriptors.numpy()

# Guardar los descriptores en un archivo
np.save('descriptors.npy', descriptors_np)

# Leer el archivo como un objeto binario
with open('descriptors.npy', 'rb') as f:
    blob_data = f.read()

# Conectar a la base de datos MySQL
cnx = mysql.connector.connect(user='root', password='',
                              host='localhost',
                              database='mydatabase')

# Crear un cursor
cursor = cnx.cursor()

# Insertar los descriptores en la base de datos
query = "INSERT INTO mytable (descriptors) VALUES (%s)"
cursor.execute(query, (blob_data,))

# Confirmar la transacción
cnx.commit()

# Cerrar la conexión
cnx.close()