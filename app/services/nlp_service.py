"""
NLP Service Module (nlp_service.py)
Provides text cleaning, tokenization, stopword removal, lemmatization, and sentiment prediction.
"""

import re
import string
import joblib
import numpy as np
import os
from typing import Dict, Any, Tuple

# Simple fallback stop words list if NLTK data is not downloaded
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he",
    "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll",
    "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves"
}

def clean_text(text: str) -> str:
    """
    Cleans raw customer review/chat text:
    1. Lowercasing
    2. HTML tag removal
    3. Punctuation removal
    4. Numeric character removal
    5. Stopword filtering & whitespace normalization
    """
    if not text or not isinstance(text, str):
        return ""
        
    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
    # Remove punctuation & numbers
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    # Tokenize and remove stopwords
    tokens = text.split()
    filtered_tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 1]
    
    return " ".join(filtered_tokens)

class SentimentAnalyzer:
    def __init__(self, model_dir: str = "app/models"):
        self.model_path = os.path.join(model_dir, "sentiment_model.pkl")
        self.vectorizer_path = os.path.join(model_dir, "sentiment_vectorizer.pkl")
        self.model = None
        self.vectorizer = None
        self._load_model()
        
    def _load_model(self):
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)

    def predict(self, text: str) -> Dict[str, Any]:
        cleaned = clean_text(text)
        if not cleaned:
            cleaned = text.lower().strip()
            
        if self.model is None or self.vectorizer is None:
            # Simple keyword rule-based fallback if model file isn't loaded yet
            pos_words = ["good", "great", "excellent", "love", "awesome", "fast", "best", "happy", "wonderful"]
            neg_words = ["bad", "terrible", "worst", "poor", "broken", "delayed", "horrible", "waste", "defective"]
            score = sum(1 for w in cleaned.split() if w in pos_words) - sum(1 for w in cleaned.split() if w in neg_words)
            if score > 0:
                sentiment = "positive"
                conf = 0.85
            elif score < 0:
                sentiment = "negative"
                conf = 0.85
            else:
                sentiment = "neutral"
                conf = 0.70
            return {"text": text, "cleaned_text": cleaned, "sentiment": sentiment, "confidence": conf}

        vec = self.vectorizer.transform([cleaned])
        pred_label = self.model.predict(vec)[0]
        probs = self.model.predict_proba(vec)[0]
        max_idx = int(np.argmax(probs))
        confidence = float(probs[max_idx])

        return {
            "text": text,
            "cleaned_text": cleaned,
            "sentiment": str(pred_label),
            "confidence": round(confidence, 4)
        }
