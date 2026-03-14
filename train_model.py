# train_model.py
# Train AI Language Detector on 234 languages using Naive Bayes with char+word TF-IDF

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import json
from language_index_map import INDEX_TO_ISO

# -----------------------------
# 1. Load full dataset
# -----------------------------
data = pd.read_csv("data/dataset.csv")
print("Full dataset loaded")
print(data.head())

# -----------------------------
# 2. Use all languages
# -----------------------------
data = data.copy()
print(f"Number of sentences: {len(data)}")
print(f"Number of unique languages: {data['language'].nunique()}")

# -----------------------------
# 3. Map numeric language codes -> consecutive labels
# -----------------------------
unique_labels = sorted(data["language"].unique())
label_mapping = {num: idx for idx, num in enumerate(unique_labels)}
data["label"] = data["language"].map(label_mapping)

X = data["text"]
y = data["label"]

# -----------------------------
# 3b. Save mapping for app.py
# -----------------------------
inverse_label_mapping = {v: k for k, v in label_mapping.items()}

model_label_to_iso = {
    label: INDEX_TO_ISO[str(num_code)]
    for label, num_code in inverse_label_mapping.items()
}

with open("language_map.json", "w") as f:
    json.dump(model_label_to_iso, f)
print("Language map (model label -> ISO code) saved")

# -----------------------------
# 4. Vectorize text (char + word TF-IDF)
# -----------------------------
char_vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(1, 4),
    lowercase=True,
    max_features=100000
)

word_vectorizer = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 2),
    lowercase=True,
    max_features=50000
)

vectorizer = FeatureUnion([
    ('char', char_vectorizer),
    ('word', word_vectorizer)
])

X_vectorized = vectorizer.fit_transform(X)
print("Text vectorization (char + word) complete")

# -----------------------------
# 5. Split dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 6. Train Naive Bayes model
# -----------------------------
model = MultinomialNB(alpha=1.5)  # increased smoothing for short text
model.fit(X_train, y_train)
print("Naive Bayes model trained successfully")

# -----------------------------
# 7. Test model
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# -----------------------------
# 8. Save model and vectorizer
# -----------------------------
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("Model and vectorizer saved")