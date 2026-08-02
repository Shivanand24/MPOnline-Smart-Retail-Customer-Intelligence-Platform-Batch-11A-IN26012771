"""
Model Training and Serialization Script (train_models.py)
Trains and serializes all ML & NLP models required by the Smart Retail Platform:
- Sentiment Analysis (TF-IDF + Logistic Regression)
- Chatbot Intent Classification (TF-IDF + Multinomial NB)
- Product Image Classifier (Visual Features + Random Forest)
- Face Database Encodings Initialization
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

MODEL_DIR = "app/models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_sentiment_model():
    print("--> Training Sentiment Analysis Model...")
    df = pd.read_csv("data/reviews.csv")
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
    X = vectorizer.fit_transform(df['review_text'])
    y = df['sentiment']
    
    model = LogisticRegression(C=1.0, max_iter=200, random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, os.path.join(MODEL_DIR, "sentiment_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "sentiment_vectorizer.pkl"))
    print(f"    Sentiment model saved to {MODEL_DIR}/sentiment_model.pkl (Accuracy: {model.score(X, y):.2%})")

def train_chatbot_model():
    print("--> Training FAQ Chatbot Intent Model...")
    with open("data/intents.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    patterns = []
    labels = []
    
    for intent in data["intents"]:
        tag = intent["tag"]
        for p in intent["patterns"]:
            patterns.append(p)
            labels.append(tag)
            
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(patterns)
    
    model = MultinomialNB(alpha=0.1)
    model.fit(X, labels)
    
    joblib.dump(model, os.path.join(MODEL_DIR, "chatbot_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "chatbot_vectorizer.pkl"))
    print(f"    Chatbot intent model saved to {MODEL_DIR}/chatbot_model.pkl (Training patterns: {len(patterns)})")

def train_product_classifier():
    print("--> Training Product Image Category Classifier...")
    # Generate synthetic visual features for 5 retail categories: Clothing, Shoes, Electronics, Bags, Groceries
    categories = ["Clothing", "Shoes", "Electronics", "Bags", "Groceries"]
    X_samples = []
    y_samples = []

    np.random.seed(42)
    for cat_idx, cat in enumerate(categories):
        for _ in range(50):
            # Create synthetic feature vector (48 color hist + 64 spatial texture = 112 features)
            # Add distinct cluster means per category for accurate classification demo
            base = np.zeros(112)
            base[:48] = np.random.dirichlet(np.ones(48) * (cat_idx + 1))
            base[48:] = np.random.normal(loc=(cat_idx * 0.2), scale=0.05, size=64)
            X_samples.append(base)
            y_samples.append(cat)

    X_train = np.array(X_samples)
    y_train = np.array(y_samples)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    joblib.dump(clf, os.path.join(MODEL_DIR, "product_classifier.pkl"))
    print(f"    Product classifier saved to {MODEL_DIR}/product_classifier.pkl (Classes: {categories})")

def init_face_database():
    print("--> Initializing Face Encodings Database...")
    db_path = os.path.join(MODEL_DIR, "face_db.pkl")
    
    np.random.seed(101)
    emb1 = np.random.randn(128).astype(np.float32)
    emb1 /= np.linalg.norm(emb1)
    
    emb2 = np.random.randn(128).astype(np.float32)
    emb2 /= np.linalg.norm(emb2)

    face_db = {
        "CUST-1001": {
            "name": "Sarah Jenkins (Gold VIP)",
            "embedding": emb1,
            "visits": 15,
            "last_seen": "2026-08-02T10:30:00"
        },
        "CUST-1002": {
            "name": "Michael Chang (Platinum VIP)",
            "embedding": emb2,
            "visits": 32,
            "last_seen": "2026-08-01T16:45:00"
        }
    }
    joblib.dump(face_db, db_path)
    print(f"    Face Database saved to {db_path} with initial VIP members.")

if __name__ == "__main__":
    print("==================================================")
    print("   TRAINING SMART RETAIL AI PLATFORM MODELS      ")
    print("==================================================")
    train_sentiment_model()
    train_chatbot_model()
    train_product_classifier()
    init_face_database()
    print("==================================================")
    print("   ALL MODELS TRAINED AND SERIALIZED SUCCESSFULLY!")
    print("==================================================")
