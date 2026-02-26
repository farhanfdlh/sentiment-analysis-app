# 📊 Sentiment Analysis App — SVM + SMOTE

A web-based Sentiment Analysis application built using **Support Vector Machine (SVM)** with **SMOTE** for handling class imbalance, and deployed using **Streamlit**.

This application allows users to upload review data (CSV/XLSX), automatically preprocess Indonesian text, and generate sentiment predictions along with visualization and model evaluation.

---

## 🚀 Live Demo

🔗 Deployed via Streamlit Cloud  
https://ai-chatbot-sentiment-analysis-app.streamlit.app/

---

## 🧠 Project Overview

This project implements:

- Text preprocessing (cleaning, case folding, normalization, stopword removal, stemming)
- TF-IDF feature extraction
- SVM classification model
- SMOTE for handling imbalanced dataset
- Lexicon-based sentiment comparison
- Interactive dashboard visualization

The application compares model predictions with lexicon-based sentiment scoring to measure performance consistency.

---

## ⚙️ Tech Stack

- Python
- Streamlit
- Scikit-learn
- Imbalanced-learn (SMOTE)
- TF-IDF Vectorizer
- Sastrawi (Indonesian Stemmer)
- NLTK (Stopwords)
- Plotly
- Pandas
- NumPy

---

## 🔎 Features

### 1️⃣ Data Upload
- Supports `.csv` and `.xlsx`
- Requires column: `ulasan`

### 2️⃣ Text Preprocessing
- Emoji removal  
- Special character cleaning  
- Case folding  
- Word normalization  
- Stopword removal  
- Indonesian stemming (Sastrawi)  

### 3️⃣ Sentiment Prediction
- TF-IDF vectorization  
- SVM classification  
- Output: Positif / Negatif  

### 4️⃣ Lexicon Comparison
- Sentiment scoring using positive & negative lexicon  
- Model vs Lexicon accuracy comparison  

### 5️⃣ Visualization
- Interactive Pie Chart  
- Sentiment distribution metrics  
- Filterable prediction table  
- CSV export  

---

## 📊 Model Information

- Algorithm: Support Vector Machine (SVM)
- Imbalance Handling: SMOTE
- Feature Extraction: TF-IDF
- Language: Indonesian
- The model was trained using Indonesian user reviews of the DeepSeek AI chatbot from Google Play Store.
