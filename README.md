# Análisis de Sentimientos con Machine Learning, Flask y Streamlit

## Descripción
Este proyecto analiza el sentimiento de un texto (**positivo, negativo o neutro**) utilizando **Machine Learning**, una **API en Flask** y una **interfaz en Streamlit**.

El modelo de Machine Learning fue entrenado con **scikit-learn** y usa **TfidfVectorizer** para procesar el texto. Luego, se despliega una API en **Flask** y una interfaz visual en **Streamlit**, ambas alojadas en Render.

---
## Tecnologías Utilizadas
- **Python 3.9+**  
- **Flask** (API REST)  
- **Streamlit** (Interfaz Gráfica)  
- **scikit-learn** (Machine Learning)  
- **pandas** (Manejo de datos)  
- **nltk** (Procesamiento de texto)  
- **joblib** (Guardar/cargar el modelo)  
- **Render** (Despliegue en la nube)  


---
## Instalación y Uso
### 1️⃣ **Clonar el repositorio**
```sh
git clone https://github.com/RoseRossi/SentimentAnalyzer.git
cd SentimentAnalyzer
```
### 2️⃣ **Crear y activar el entorno virtual**
```sh
python -m venv venv
# Activar el entorno
# En Windows:
venv\Scripts\activate
# En Mac/Linux:
source venv/bin/activate
```
### 3️⃣ **Instalar las dependencias**
```sh
pip install -r requirements.txt
```
### 4️⃣ **Entrenar el modelo (opcional)**
```sh
python src/train_model.py
```
### 5️⃣ **Ejecutar la API Flask**
```sh
python src/api.py
```
La API estará disponible en: `http://127.0.0.1:5000/`

### 6️⃣ **Ejecutar la interfaz en Streamlit**
```sh
streamlit run src/app.py
```
La interfaz estará disponible en: `http://localhost:8501/`

---
## Despliegue en la Nube
### **API Flask en Render**
La API está desplegada en Render y se puede acceder desde:
```
https://sentimentanalyzer-yvg1.onrender.com/analizar
```
Ejemplo de petición con `curl`:
```sh
curl -X POST https://sentimentanalyzer-yvg1.onrender.com/analizar \
-H "Content-Type: application/json" \
-d '{"texto": "Me encanta este producto, es increíble"}'
```

### **Interfaz en Streamlit**
La interfaz está desplegada en:
```
https://sentimentanalyzer-uistreamlit-run-src.onrender.com
```
Aquí puedes escribir cualquier texto y ver su análisis de sentimiento en tiempo real.


