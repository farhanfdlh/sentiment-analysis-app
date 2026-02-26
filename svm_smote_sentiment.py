import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import joblib
import plotly.express as px
from typing import Tuple, List, Dict, Any

# --- NLP Dependencies ---
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("indonesian"))

# Stemmer
_factory = StemmerFactory()
_stemmer = _factory.create_stemmer()

# ======================= PREPROCESSING =========================

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    emoji_pattern = re.compile(
        "["  
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub("", text)
    text = re.sub(r"[^0-9A-Za-z\s]", "", text)
    text = re.sub(r"\d+", "", text)
    return " ".join(text.split())


def to_lower(text: str) -> str:
    return text.lower() if isinstance(text, str) else ""


def normalize_words(text: str, dictionary_nonstandard: Dict[str, str]):
    if not isinstance(text, str):
        return "", [], [], []
    tokens = text.split()
    processed = [dictionary_nonstandard.get(token, token) for token in tokens]
    return " ".join(processed), [], [], []


def word_tokenizer(text: str) -> List[str]:
    return text.strip().split() if isinstance(text, str) else []


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [w for w in tokens if w not in STOPWORDS]


def apply_stemming(token_list: List[str]) -> List[str]:
    return [_stemmer.stem(w) for w in token_list]


def preprocess_full_text(raw_text: Any, dictionary_nonstandard: Dict[str, str]) -> str:
    if not isinstance(raw_text, str):
        return ""
    text = clean_text(raw_text)
    text = to_lower(text)
    text, _, _, _ = normalize_words(text, dictionary_nonstandard)
    tokens = word_tokenizer(text)
    tokens = remove_stopwords(tokens)
    tokens = apply_stemming(tokens)
    return " ".join(tokens)


# ======================= LEXICON =========================

def load_lexicon():
    pos_df = pd.read_csv("positive_lex.tsv", sep="\t")
    neg_df = pd.read_csv("negative_lex.tsv", sep="\t")

    pos_df.columns = ["word", "weight"]
    neg_df.columns = ["word", "weight"]

    merged = pd.concat([pos_df, neg_df])
    merged["weight"] = pd.to_numeric(merged["weight"], errors="coerce").fillna(0)

    return dict(zip(merged["word"], merged["weight"]))


def compute_sentiment_from_lexicon(text: str, lexicon: Dict[str, float]):
    if not isinstance(text, str):
        return 0.0, "Positif"

    tokens = text.split()
    score = sum(lexicon.get(tok, 0.0) for tok in tokens)

    # Tidak ada netral: positif >= 0, negatif < 0
    label = "Positif" if score >= 0 else "Negatif"
    return score, label


# ======================= MODEL LABEL NORMALIZATION =========================

def normalize_model_label(pred):
    if isinstance(pred, (int, float)):
        return "Positif" if pred == 1 else "Negatif"
    s = str(pred).strip().lower()
    if s in {"positif", "positive", "1"}:
        return "Positif"
    return "Negatif"


# ======================= STREAMLIT UI =========================
st.set_page_config(page_title="Analisis Sentimen", layout="wide")
st.title("Analisis Sentimen — Model SVM + SMOTE")

st.markdown("""
### Panduan
- Upload **CSV atau XLSX** berisi kolom `ulasan`.
- Model, lexicon, dan kamus kata baku otomatis dimuat dari folder yang sama dengan svm_smote_sentiment.py.
""")

# ======================= FILE UPLOAD =========================

data_file = st.file_uploader("Upload Data Ulasan (CSV/XLSX)", type=["csv", "xlsx"])
run_button = st.button("Analisis Sentimen")

# Jika user hanya ingin filter tabel tanpa klik analisis ulang
if "hasil_df" in st.session_state and not run_button:
    df = st.session_state["hasil_df"]
else:
    df = None


# ======================= EXECUTION =========================

if run_button:

    if data_file is None:
        st.error("Silakan upload data ulasan.")
        st.stop()

    try:
        df = pd.read_csv(data_file) if data_file.name.endswith(".csv") else pd.read_excel(data_file)
    except:
        st.error("Format file tidak valid atau rusak.")
        st.stop()

    if "ulasan" not in df.columns:
        st.error("File harus memiliki kolom **ulasan**.")
        st.stop()

    df["ulasan"] = df["ulasan"].astype(str)

    # Load lexicon
    LEXICON = load_lexicon()

    # Load model & vectorizer
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    model = joblib.load("svm_smote_model.joblib")

    # Load kamus kata baku
    kamus_nonbaku = {}
    try:
        df_kamus = pd.read_excel("kamuskatabaku.xlsx")
        kamus_nonbaku = dict(zip(df_kamus["tidak_baku"], df_kamus["kata_baku"]))
    except:
        pass

    # Preprocessing
    df["preprocessed"] = df["ulasan"].apply(lambda x: preprocess_full_text(x, kamus_nonbaku))

    # Lexicon sentiment
    lex = df["preprocessed"].apply(lambda t: compute_sentiment_from_lexicon(t, LEXICON))
    df["lexicon_score"] = lex.apply(lambda x: x[0])
    df["lexicon_label"] = lex.apply(lambda x: x[1])

    # Model Predictions
    X = vectorizer.transform(df["preprocessed"])
    preds = model.predict(X)
    df["prediksi_model"] = [normalize_model_label(p) for p in preds]

    # Accuracy comparison
    accuracy = (df["prediksi_model"] == df["lexicon_label"]).mean()

    # Save to session_state (FIX FILTER TROUBLE)
    st.session_state["hasil_df"] = df
    st.session_state["accuracy"] = accuracy


# ======================= DISPLAY RESULTS =========================

if df is not None:

    # ================= PIE CHART (Plotly) =================
    st.subheader("Distribusi Sentimen (Prediksi Model)")

    counts = df["prediksi_model"].value_counts()
    chart_df = pd.DataFrame({"sentimen": counts.index, "jumlah": counts.values})

    fig = px.pie(chart_df, names="sentimen", values="jumlah", hole=0.35)
    st.plotly_chart(fig, use_container_width=True)

    # ================= RINGKASAN HASIL =================
    st.subheader("Ringkasan")

    df = st.session_state.get("hasil_df", None)
    acc = st.session_state.get("accuracy", 0)

    if df is not None:
        total = len(df)

        # Hitung distribusi
        pos_count = (df["prediksi_model"] == "Positif").sum()
        neg_count = (df["prediksi_model"] == "Negatif").sum()

        pos_pct = (pos_count / total * 100) if total > 0 else 0
        neg_pct = (neg_count / total * 100) if total > 0 else 0

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Ulasan", total)

        with col2:
            st.metric("Sentimen Positif",
                    f"{pos_count}  ({pos_pct:.1f}%)")

        with col3:
            st.metric("Sentimen Negatif",
                    f"{neg_count}  ({neg_pct:.1f}%)")

        with col4:
            st.metric("Akurasi Model vs Lexicon", f"{acc:.4f}")

    # ================= FILTER & TABLE =================
    st.subheader("Tabel Hasil Prediksi")

    filter_choice = st.selectbox(
        "Filter Sentimen",
        ["Semua", "Positif", "Negatif"]
    )

    if filter_choice == "Semua":
        show_df = df
    else:
        show_df = df[df["prediksi_model"] == filter_choice]

    st.dataframe(show_df[[
        "ulasan", "preprocessed", "prediksi_model",
        "lexicon_score", "lexicon_label"
    ]], use_container_width=True)

    # Download CSV
    csv_buf = io.BytesIO()
    show_df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)
    st.download_button("Unduh CSV", csv_buf, "hasil_sentimen.csv", "text/csv")

