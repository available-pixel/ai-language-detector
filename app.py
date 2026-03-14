# app.py
# Streamlit AI Language Detector (234 languages)

import streamlit as st
import joblib
import pandas as pd
import json
from language_full_map import LANGUAGE_FULL_MAP

# -----------------------------
# Custom CSS for background
# -----------------------------
st.markdown(
    """
    <style>
    /* Gradient background for the whole page */
    .stApp {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        background-attachment: fixed;
    }

    /* Optional: Make text more readable */
    .stTextArea textarea, .stTextArea label, .stMarkdown, .stSubheader {
        color: #333333;
        font-weight: 500;
    }

    /* Optional: Card background for info boxes and tables */
    .stInfo, .stTable, .stDataFrame {
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Load model
# -----------------------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

with open("language_map.json", "r") as f:
    language_map = json.load(f)

# -----------------------------
# Streamlit page
# -----------------------------
st.set_page_config(
    page_title="AI Language Detector",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 AI Language Detector")

# -----------------------------
# Columns Layout
# -----------------------------
col1, col2 = st.columns([2,1])

# =============================
# LEFT COLUMN
# =============================
with col1:

    st.warning(
        "⚠️ **Best accuracy with medium or long sentences (25+ characters).**"
    )

    st.write(f"Supports **{len(language_map)} languages**.")
    
    # -----------------------------
    # Example supported languages
    # -----------------------------
    st.markdown("**Example supported languages:** English, French, Spanish, German, Arabic, Chinese, Japanese, Portuguese, Italian, Russian, Hindi, Bengali, Urdu, Turkish, Persian, Korean, Vietnamese, Thai, Indonesian, Malay, Swahili, Yoruba, Zulu, Hausa, Igbo, Amharic, Somali, Oromo, Dutch, Danish, Swedish, Norwegian, Finnish, Icelandic, Polish, Czech, Slovak, Hungarian, Romanian, Bulgarian, Greek, Hebrew, Ukrainian, Serbian, Croatian, Slovenian, Latvian, Lithuanian, Estonian, Catalan, Basque and 180+ more.")
    
    # -----------------------------
    # Instruction for user
    # -----------------------------
    st.info(
        "💡 How to use the text box:\n"
        "- Type or paste your sentence, then click outside the box.\n"
        "- The AI will automatically detect the language. Short sentences may be less accurate."
    )   

    # -----------------------------
    # Text input
    # -----------------------------
    text_input = st.text_area(
        "🌍 Enter your sentence here:",
        placeholder="🖊️ Type your sentence here...",
        height=150
    ) 

    MIN_CHARS = 25

    if text_input.strip():

        if len(text_input) < MIN_CHARS:

            st.error(
                f"Sentence too short ({len(text_input)} characters).\n"
                f"Please enter **at least {MIN_CHARS} characters**."
            )

        else:

            X_input = vectorizer.transform([text_input])
            probs = model.predict_proba(X_input)[0]

            # Top 3 predictions
            top3_idx = probs.argsort()[::-1][:3]
            top3_probs = probs[top3_idx]

            top3_iso = [language_map[str(i)] for i in top3_idx]
            top3_names = [LANGUAGE_FULL_MAP[iso] for iso in top3_iso]

            # ----------------
            # Result
            # ----------------
            st.subheader("Detected Language")
            st.success(top3_names[0])

            st.subheader("Top Predictions")

            table_data = pd.DataFrame({
                "Language": top3_names,
                "Confidence": [f"{p*100:.2f}%" for p in top3_probs]
            })

            st.table(table_data)

            chart_data = pd.DataFrame({
                "Language": top3_names,
                "Confidence": top3_probs
            }).set_index("Language")

            st.bar_chart(chart_data)

            st.caption("Confidence shows how sure the AI is.")

# =============================
# RIGHT COLUMN
# =============================
with col2:

    st.subheader("📋 Example Sentences")

    examples = [
"🇬🇧 Hello, I am learning artificial intelligence and I enjoy programming in Python.",
"🇫🇷 Bonjour, je suis en train d'apprendre l'intelligence artificielle et la programmation.",
"🇪🇸 Hola, estoy aprendiendo inteligencia artificial y me gusta programar en Python.",
"🇩🇪 Das ist ein Beispieltext auf Deutsch, um die Sprache korrekt zu erkennen.",
"🇮🇹 Sto imparando l'intelligenza artificiale e mi piace programmare in Python.",
"🇵🇹 Estou aprendendo inteligência artificial e gosto de programar em Python.",
"🇳🇱 Ik leer kunstmatige intelligentie en programmeer graag in Python.",
"🇷🇺 Я изучаю искусственный интеллект и люблю программировать на Python.",
"🇨🇳 我正在学习人工智能，并且喜欢用 Python 编程。"
]

    for s in examples:
        st.code(s)