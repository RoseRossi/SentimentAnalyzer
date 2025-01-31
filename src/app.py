import streamlit as st
import requests

# Configurar la página de Streamlit
st.set_page_config(page_title="Análisis de Sentimientos", page_icon="🎭")

# Título de la aplicación
st.title("Analizador de Sentimientos con IA")

st.write("Ingresa un texto y nuestro modelo de Machine Learning analizará su sentimiento.")

# Input del usuario
texto_usuario = st.text_area("✍ Escribe un texto para analizar:")

# URL de la API desplegada en Render
URL_API = "https://sentimentanalyzer-yvg1.onrender.com/analizar"

# Botón para analizar el sentimiento
if st.button("Analizar Sentimiento"):
    if texto_usuario:
        # Enviar texto a la API Flask
        datos = {"texto": texto_usuario}
        respuesta = requests.post(URL_API, json=datos)

        # Verificar si la API respondió correctamente
        if respuesta.status_code == 200:
            resultado = respuesta.json()
            sentimiento = resultado["sentimiento"]

            # Mostrar el resultado con colores
            if sentimiento == "positivo":
                st.success(f"💚 **Sentimiento:** {sentimiento.capitalize()} 🎉")
            elif sentimiento == "negativo":
                st.error(f"💔 **Sentimiento:** {sentimiento.capitalize()} 😞")
            else:
                st.warning(f"😐 **Sentimiento:** {sentimiento.capitalize()} 🤔")
        else:
            st.error("⚠️ Error al conectarse con la API.")
    else:
        st.warning("Por favor, ingresa un texto.")

# Pie de página
st.markdown("---")
st.write("💡 Desarrollado por Isabela | API en Render 🚀")
