from flask import Flask, request, jsonify
import joblib
import string

# Cargar el modelo entrenado
modelo = joblib.load("modelo_sentimientos.pkl")

# Función para limpiar el texto
def limpiar_texto(texto):
    texto = texto.lower()
    texto = texto.translate(str.maketrans("", "", string.punctuation))
    return texto

# Inicializar la aplicación Flask
app = Flask(__name__)

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
    sentimiento = modelo.predict([texto_limpio])[0]

    return jsonify({"texto": texto, "sentimiento": sentimiento})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Tomar el puerto de Render o usar 5000 por defecto
    app.run(host="0.0.0.0", port=port, debug=True)
