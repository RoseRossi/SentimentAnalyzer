import pandas as pd
import string
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE  # Nueva librería para balanceo

# Descargar stopwords y lematizador si es la primera vez
nltk.download("stopwords")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("spanish"))

# Función para limpiar el texto
def limpiar_texto(texto):
    texto = texto.lower()  # Convertir a minúsculas
    texto = texto.translate(str.maketrans("", "", string.punctuation))  # Eliminar puntuación
    palabras = texto.split()
    palabras = [lemmatizer.lemmatize(palabra) for palabra in palabras if palabra not in stop_words]
    return " ".join(palabras)

# Cargar el dataset
df = pd.read_csv("data/reviews.csv")

print("Primeras filas del dataset:")
print(df.head())

print("Distribución de clases:")
print(df["sentiment"].value_counts())

# Limpiar texto
df["review"] = df["review"].apply(limpiar_texto)

# Codificar etiquetas de sentimiento
label_encoder = LabelEncoder()
df["sentiment"] = label_encoder.fit_transform(df["sentiment"])

# Separar datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df["review"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"])

# Convertir a formato numérico para SMOTE
vectorizador = TfidfVectorizer(ngram_range=(1,1), max_features=10000)
X_train_tfidf = vectorizador.fit_transform(X_train)
X_test_tfidf = vectorizador.transform(X_test)

# Aplicar SMOTE
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_train_tfidf, y_train = smote.fit_resample(X_train_tfidf, y_train)

# Crear modelo de Machine Learning con RandomForestClassifier
modelo = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=5, min_samples_leaf=2, random_state=42)

# Validación cruzada para mejor evaluación
try:
    scores = cross_val_score(modelo, X_train_tfidf, y_train, cv=5)
    print(f"Precisión promedio en validación cruzada: {scores.mean():.2f}")
except Exception as e:
    print(f"Error en validación cruzada: {e}")

# Entrenar el modelo
modelo.fit(X_train_tfidf, y_train)

# Revisar qué tan bien aprende el modelo en entrenamiento
train_accuracy = modelo.score(X_train_tfidf, y_train)
test_accuracy = modelo.score(X_test_tfidf, y_test)

print(f"Precisión en entrenamiento: {train_accuracy:.2f}")
print(f"Precisión en prueba: {test_accuracy:.2f}")

# Evaluar el modelo en los datos de prueba
y_pred = modelo.predict(X_test_tfidf)
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, zero_division=1))

# Probar el modelo con datos de prueba
print("\nEjemplo de predicciones en datos de prueba:")
for texto, real, pred in zip(X_test[:10], y_test[:10], y_pred[:10]):
    print(f"Texto: {texto} | Real: {label_encoder.inverse_transform([real])[0]} | Predicción: {label_encoder.inverse_transform([pred])[0]}")

# Después de entrenar y ajustar el vectorizador
joblib.dump(vectorizador, "vectorizador_tfidf.pkl")  # Guarda el vectorizador
joblib.dump(modelo, "modelo_sentimientos.pkl")  # Guarda el modelo
print("Modelo y vectorizador guardados correctamente.")