import joblib
import string

# Cargar el modelo entrenado
modelo = joblib.load("modelo_sentimientos.pkl")

# Función para limpiar el texto
def limpiar_texto(texto):
    texto = texto.lower()
    texto = texto.translate(str.maketrans("", "", string.punctuation))
    return texto

def analizar_sentimiento(texto):
    """ Analiza el sentimiento usando el modelo de Machine Learning """
    texto_limpio = limpiar_texto(texto)
    sentimiento = modelo.predict([texto_limpio])[0]
    return sentimiento

# Prueba
if __name__ == "__main__":
    ejemplos = [
        "Me encanta este producto, es fantástico.",
        "No me gustó la comida, estaba fría.",
        "Es un servicio normal, nada especial."
    ]
    
    for texto in ejemplos:
        print(f"Texto: {texto} → Sentimiento: {analizar_sentimiento(texto)}")
