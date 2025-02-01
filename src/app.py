import streamlit as st
import requests
import matplotlib.pyplot as plt

# Configurar la página de Streamlit
st.set_page_config(page_title="Análisis de Sentimientos", page_icon="🎭")

# Título de la aplicación
st.title("🔍 Analizador de Sentimientos con IA")
st.write("Ingresa un texto y nuestro modelo de Machine Learning analizará su sentimiento.")

# Input del usuario
texto_usuario = st.text_area("✍ Escribe un texto para analizar:")

# URL de la API desplegada en Render
URL_API = "https://sentimentanalyzer-yvg1.onrender.com/analizar"

# Inicializar historial si no existe
if "historial" not in st.session_state:
    st.session_state["historial"] = []

# Botón para analizar el sentimiento
if st.button("🔍 Analizar Sentimiento"):
    if texto_usuario:
        # Enviar texto a la API Flask
        datos = {"texto": texto_usuario}
        respuesta = requests.post(URL_API, json=datos)

        # Verificar si la API respondió correctamente
        if respuesta.status_code == 200:
            resultado = respuesta.json()
            sentimiento = resultado["sentimiento"]

            # Guardar en el historial (máximo 10 elementos)
            st.session_state["historial"].append({"texto": texto_usuario, "sentimiento": sentimiento})
            if len(st.session_state["historial"]) > 10:
                st.session_state["historial"].pop(0)

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

# Sidebar: Historial de análisis
st.sidebar.title("📜 Historial de Análisis")
for item in st.session_state["historial"]:
    st.sidebar.write(f"🔹 {item['texto']} → **{item['sentimiento'].capitalize()}**")

# Sidebar: Gráfico de distribución de sentimientos
st.sidebar.title("📊 Distribución de Sentimientos")

# Contar cantidad de cada sentimiento
contadores = {"positivo": 0, "negativo": 0, "neutro": 0}
for item in st.session_state["historial"]:
    contadores[item["sentimiento"]] += 1

# Si hay datos en el historial, generamos el gráfico
if sum(contadores.values()) > 0:
    fig, ax = plt.subplots()
    ax.bar(contadores.keys(), contadores.values(), color=["green", "red", "gray"])
    ax.set_title("Distribución de Sentimientos")
    ax.set_ylabel("Cantidad")
    
    # Mostrar gráfico en Streamlit
    st.sidebar.pyplot(fig)

# Pie de página
st.markdown("---")
st.write("💡 Desarrollado por Isabela | API en Render 🚀")
