import streamlit as st
import joblib
import re
import spacy
from textblob import TextBlob

# Cargar modelos de spaCy optimizados
nlp_es = spacy.load("es_core_news_sm", disable=["parser", "ner"])
nlp_en = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Heurística simple de idioma
def detectar_idioma_simple(texto):
    texto = texto.lower()
    if any(p in texto for p in [" el ", " la ", " los ", " una ", " esto ", " que "]):
        return "es"
    return "en"

# Función de limpieza + lematización
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+|https\S+", '', texto)
    texto = re.sub(r'\@[\w]*', '', texto)
    texto = re.sub(r'\#', '', texto)
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\d+', '', texto)
    texto = texto.strip()

    if not texto or len(texto.split()) < 1:
        return ""

    idioma = detectar_idioma_simple(texto)
    nlp = nlp_es if idioma == "es" else nlp_en
    doc = nlp(texto)
    return " ".join([token.lemma_ for token in doc])

# Cargar modelo y vectorizador
try:
    model = joblib.load("models/modelo.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
except Exception as e:
    st.error(f"❌ Error al cargar el modelo/vectorizador: {e}")
    st.stop()

# Función de predicción
def predecir(texto):
    limpio = limpiar_texto(texto)
    if limpio == "":
        return None, None, None
    vect = vectorizer.transform([limpio])
    pred = model.predict(vect)[0]
    prob = model.predict_proba(vect)[0]
    sentimiento = TextBlob(limpio).sentiment.polarity
    return pred, prob, sentimiento

# Interfaz Streamlit
st.title("📰 Clasificador de Noticias Falsas")

entrada = st.text_input("✏️ Escribe un titular:")

if entrada:
    etiqueta, prob, sentimiento = predecir(entrada)

    if etiqueta is None:
        st.warning("⚠️ El titular no contiene información suficiente tras el procesamiento.")
    else:
        st.markdown(f"📢 **Resultado:** {'✅ Real' if etiqueta == 1 else '❌ Falsa'}")
        st.markdown(f"📊 **Confianza del modelo:** {max(prob) * 100:.2f}%")

        if sentimiento > 0.1:
            st.markdown("💬 **Sentimiento:** Positivo 😊")
        elif sentimiento < -0.1:
            st.markdown("💬 **Sentimiento:** Negativo 😠")
        else:
            st.markdown("💬 **Sentimiento:** Neutro 😐")

            
            

