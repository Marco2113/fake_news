# 🧠 Clasificador de Noticias Reales vs Falsas

Este proyecto aplica técnicas de **Procesamiento de Lenguaje Natural (NLP)** para construir un modelo de clasificación binaria que determine si un titular de noticia es **real o falso**. Además, se incluye una aplicación web desarrollada en **Streamlit** que permite realizar predicciones en tiempo real.

---

## 📁 Estructura del Proyecto

```
📦 proyecto/
├── data/
│   ├── training_data.csv
│   └── testing_data.csv
├── models/
│   ├── modelo.pkl
│   └── vectorizer.pkl
├── output/
│   └── predicciones.csv
├── modelopredictivo_con_rf_y_matriz.ipynb
├── app.py
├── requirements.txt
└── README.md
```

---

## 🧠 Características del modelo

- Preprocesamiento con **spaCy**: limpieza, normalización y lematización.
- Soporte **multilingüe** (español e inglés) con heurística de idioma optimizada.
- Representación del texto mediante **TF-IDF**.
- Clasificación binaria con:
  - `LogisticRegression`
  - `RandomForestClassifier` (mejor evaluado)
- Matriz de confusión incluida en el notebook.
- Aplicación con Streamlit para predicción y análisis de sentimiento.

---

## ▶️ Cómo ejecutar

### 1. Clona el repositorio y entra en la carpeta:

```bash
git clone https://github.com/Marco2113/fake_news
cd fake_news
```

### 2. Instala las dependencias:

```bash
pip install -r requirements.txt
```

### 3. Descarga los modelos de lenguaje de spaCy:

```bash
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm
```

### 4. Ejecuta la aplicación web:

```bash
streamlit run app.py
```

---

## 📊 Dataset

- `training_data.csv`: contiene titulares con etiquetas (0 = falso, 1 = real).
- `testing_data.csv`: contiene titulares sin etiqueta (etiqueta = 2).
- Ambos archivos deben estar en la carpeta `data/`.

---

## 📝 Autor

- **Nombre**: Marco Adrian
- **Curso**: Master en Data Science e IA
- **Fecha**: Mayo, 2025

---

## ✅ Créditos

- [spaCy](https://spacy.io/)
- [scikit-learn](https://scikit-learn.org/)
- [Streamlit](https://streamlit.io/)
- [TextBlob](https://textblob.readthedocs.io/)
