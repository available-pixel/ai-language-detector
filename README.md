# 🌍 AI Language Detector

**Detect 234 languages in any text** using a Naive Bayes AI model. This Streamlit web app identifies languages, shows confidence scores, and works best with medium-to-long sentences.

🔗 **Try it online:** [AI Language Detector](https://ai-language-detector-fadil-ade.streamlit.app/)

---

## Features

- ✅ Detects **234 languages**  
- ✅ Shows **full language names** (no codes)  
- ✅ Top 3 predictions with confidence scores  
- ✅ Bar chart visualization for confidence  
- ✅ Column layout: input on left, example sentences on right  
- ✅ Gradient background and clear instructions  
- ✅ Auto-downloads model & vectorizer if missing  

---

## How to Use

1. Type or paste your sentence in the input box.  
2. Click outside the text box or press Enter.  
3. The AI automatically detects the language.  
4. For short sentences (<25 characters), a warning appears — detection may be inaccurate.  

### Example Sentences

🇬🇧 Hello, I am learning artificial intelligence and I enjoy programming in Python.
🇫🇷 Bonjour, je suis en train d'apprendre l'intelligence artificielle et la programmation.
🇪🇸 Hola, estoy aprendiendo inteligencia artificial y me gusta programar en Python.
🇩🇪 Das ist ein Beispieltext auf Deutsch, um die Sprache korrekt zu erkennen.
🇮🇹 Sto imparando l'intelligenza artificiale e mi piace programmare in Python.
🇵🇹 Estou aprendendo inteligência artificial e gosto de programar em Python.
🇳🇱 Ik leer kunstmatige intelligentie en programmeer graag in Python.
🇷🇺 Я изучаю искусственный интеллект и люблю программировать на Python.
🇨🇳 我正在学习人工智能，并且喜欢用 Python 编程。

**Installation (for local use)**

# Clone repository
- git clone https://github.com/available-pixel/ai-language-detector.git
- cd ai-language-detector

# Create and activate virtual environment
- python -m venv venv
- venv\Scripts\activate   # Windows
- source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py

**Requirements**

- Python 3.11 (recommended)
- Streamlit
- scikit-learn
- pandas
- gdown

**About the Model**

Trained on 234 languages using Naive Bayes
Character-level TF-IDF vectorization (1–4 grams)
Handles medium-to-long text best
