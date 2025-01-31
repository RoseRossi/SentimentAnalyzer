import pandas as pd
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Función para limpiar el texto
def limpiar_texto(texto):
    texto = texto.lower()
    texto = texto.translate(str.maketrans("", "", string.punctuation))
    return texto

# Cargar el dataset
df = pd.read_csv("data/reviews.csv")

print("Primeras filas del dataset:")
print(df.head())

print("Distribución de clases:")
print(df["sentiment"].value_counts())


# Limpiar texto
df["review"] = df["review"].apply(limpiar_texto)

# Separar datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df["review"], df["sentiment"], test_size=0.2, random_state=42)

# Crear modelo de Machine Learning
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Usamos una lista personalizada de stop words en español
stop_words_es = ["el", "la", "los", "las", "de", "y", "a", "en", "que", "con", "por", "para", "es", "un", "una", "se"]

modelo = Pipeline([
    ("vectorizador", TfidfVectorizer(ngram_range=(1,2), stop_words=stop_words_es, max_features=1000)),
    ("clasificador", LogisticRegression(max_iter=500, solver='lbfgs', C=1.5))
])


# Entrenar el modelo
modelo.fit(X_train, y_train)

# Revisar qué tan bien aprende el modelo en entrenamiento
train_accuracy = modelo.score(X_train, y_train)
test_accuracy = modelo.score(X_test, y_test)

print(f"Precisión en entrenamiento: {train_accuracy:.2f}")
print(f"Precisión en prueba: {test_accuracy:.2f}")


# Probar el modelo con datos de prueba
predicciones = modelo.predict(X_test)
print("\nEjemplo de predicciones en datos de prueba:")
for texto, real, pred in zip(X_test[:10], y_test[:10], predicciones[:10]):
    print(f"Texto: {texto} | Real: {real} | Predicción: {pred}")


# Guardar el modelo entrenado
joblib.dump(modelo, "modelo_sentimientos.pkl")
print("Modelo guardado correctamente.")
