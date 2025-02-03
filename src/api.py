from flask import Flask, request, jsonify
import joblib
import string
import os
from flask_cors import CORS

# Cargar el modelo entrenado y el vectorizador
modelo = joblib.load("modelo_sentimientos.pkl")
vectorizador = joblib.load("vectorizador_tfidf.pkl")  # ¡Cargar el vectorizador!

# Función para limpiar el texto
def limpiar_texto(texto):
    texto = texto.lower()
    texto = texto.translate(str.maketrans("", "", string.punctuation))
    return texto

# Inicializar la aplicación Flask
app = Flask(__name__)
CORS(app)  # Habilitar CORS para permitir solicitudes desde Streamlit

@app.route("/")
def home():
    return jsonify({"mensaje": "API de Análisis de Sentimientos activa"}), 200

@app.route("/analizar", methods=["POST"])
def analizar():
    datos = request.json  # Recibir JSON con el texto
    if not datos or "texto" not in datos:
        return jsonify({"error": "Falta el parámetro 'texto'"}), 400

    texto = datos["texto"]
    texto_limpio = limpiar_texto(texto)

    # Transformar el texto con el mismo vectorizador
    texto_vectorizado = vectorizador.transform([texto_limpio])

    # Predecir el sentimiento
    sentimientos_dict = {0: "negativo", 1: "neutro", 2: "positivo"}  # Mapear números a etiquetas
    sentimiento = int(modelo.predict(texto_vectorizado)[0])  # Convertir a int
    sentimiento_texto = sentimientos_dict.get(sentimiento, "desconocido")  # Convertir a etiqueta
    return jsonify({"texto": texto, "sentimiento": sentimiento_texto})  # Devolver como texto

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Usa el puerto de Render o 5000 por defecto
    app.run(host="0.0.0.0", port=port, debug=False)  # Desactiva el modo debug
