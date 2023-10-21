from flask import Flask, render_template, request, jsonify
from flask_mysqldb import MySQL

app = Flask(__name__)

# Configuración de la base de datos MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'tu_contraseña_mysql'
app.config['MYSQL_DB'] = 'tu_base_de_datos_mysql'

# Inicializa la extensión MySQL
mysql = MySQL(app)
