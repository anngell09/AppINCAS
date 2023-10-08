from flask import Flask, render_template, request
from facenet_pytorch import InceptionResnetV1
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import torch
import os
import mysql.connector

app = Flask(__name__)

# Ruta para cargar archivos
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configurar el modelo InceptionResnetV1
encoder = InceptionResnetV1(pretrained='vggface2', classify=False).eval()

# Configurar la detección de caras con MTCNN
mtcnn = MTCNN()

def procesar_imagen(imagen_path, nie, nombres, apellidos):
    # Leer la imagen
    imagen = cv2.imread(imagen_path)
    
    # Detección de caras con MTCNN
    caras = mtcnn.detect_faces(imagen)

    # Asegurarse de que se haya detectado al menos una cara
    if len(caras) > 0:
        # Extraer la región de la primera cara detectada
        x, y, width, height = caras[0]['box']
        face = imagen[y:y+height, x:x+width]
        
        # Redimensionar la cara a 160x160
        face = cv2.resize(face, (160, 160))
        
        # Preprocesar la imagen para el modelo InceptionResnetV1
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = np.transpose(face, (2, 0, 1))
        face = face / 255.0
        
        # Convertir la cara en un tensor de PyTorch
        face_tensor = torch.tensor(face, dtype=torch.float32)
        face_tensor = face_tensor.unsqueeze(0)  # Agregar una dimensión de lote

        # Generar el embedding de la cara
        embedding_cara = encoder(face_tensor)

        # Guardar en la base de datos
        guardar_en_base_de_datos(embedding_cara, nie, nombres, apellidos)
        
        return embedding_cara
    else:
        return None

def guardar_en_base_de_datos(embedding_cara, nie, nombres, apellidos):
    # Conectar a la base de datos MySQL
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="proyectofacial"
    )
    cursor = conn.cursor()

    # Convertir el tensor de embedding a una cadena para almacenarlo en la base de datos
    embedding_str = ','.join(map(str, embedding_cara.squeeze().tolist()))

    # Insertar datos en la tabla
    cursor.execute('''
        INSERT INTO embeddings (nie, nombres, apellidos, embedding)
        VALUES (%s, %s, %s, %s)
    ''', (nie, nombres, apellidos, embedding_str))

    # Guardar cambios y cerrar la conexión
    conn.commit()
    conn.close()

@app.route('/')
def index():
    # Obtener datos de la base de datos
    datos = obtener_datos_de_base_de_datos()
    return render_template('index.html', datos=datos)

def obtener_datos_de_base_de_datos():
    # Conectar a la base de datos y obtener datos
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="proyectofacial"
    )
    cursor = conn.cursor()

    cursor.execute('''
        SELECT nie, nombres, apellidos, embedding FROM embeddings
    ''')

    datos = cursor.fetchall()

    conn.close()

    return datos

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No se proporcionó un archivo')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', message='No se seleccionó un archivo')

        nie = request.form.get('nie')
        nombres = request.form.get('nombres')
        apellidos = request.form.get('apellidos')

        if file and nie and nombres and apellidos:
            # Guardar el archivo en el sistema de archivos
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Procesar la imagen y obtener el embedding
            embedding = procesar_imagen(file_path, nie, nombres, apellidos)

            if embedding is not None:
                # El resultado es el embedding de la cara que puedes utilizar para tus necesidades
                return render_template('index.html', message='Archivo cargado exitosamente. Embedding: {}'.format(embedding))
            else:
                return render_template('index.html', message='No se detectaron caras en la imagen')

if __name__ == '__main__':
    app.run(debug=True)
