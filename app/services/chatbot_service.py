"""
Chatbot Service Module (chatbot_service.py)
Implements a Hybrid Retail FAQ Chatbot using:
1. Exact pattern rule matching
2. ML TF-IDF + Classifier intent matching fallback
3. Contextual response generation
"""

import json
import os
import joblib
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from app.services.nlp_service import clean_text

class RetailChatbot:
    def __init__(self, data_path: str = "data/intents.json", model_dir: str = "app/models"):
        self.intents_file = data_path
        self.model_path = os.path.join(model_dir, "chatbot_model.pkl")
        self.vectorizer_path = os.path.join(model_dir, "chatbot_vectorizer.pkl")
        
        self.intents_data = self._load_intents()
        self.model = None
        self.vectorizer = None
        self._load_ml_model()

    def _load_intents(self) -> Dict[str, Any]:
        if os.path.exists(self.intents_file):
            with open(self.intents_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"intents": []}

    def _load_ml_model(self):
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)

    def _rule_based_match(self, user_msg: str) -> Tuple[Optional[str], Optional[str]]:
        msg_clean = user_msg.lower().strip()
        for intent in self.intents_data.get("intents", []):
            for pattern in intent.get("patterns", []):
                if pattern.lower() in msg_clean or msg_clean in pattern.lower():
                    response = np.random.choice(intent["responses"])
                    return intent["tag"], response
        return None, None

    def get_response(self, user_msg: str) -> Dict[str, Any]:
        if not user_msg or not user_msg.strip():
            return {
                "message": user_msg,
                "intent": "unknown",
                "response": "Please enter a question or message so I can assist you!",
                "confidence": 0.0,
                "match_type": "default"
            }

        # Step 1: Rule-based exact pattern matching
        rule_tag, rule_resp = self._rule_based_match(user_msg)
        if rule_tag and rule_resp:
            return {
                "message": user_msg,
                "intent": rule_tag,
                "response": str(rule_resp),
                "confidence": 0.98,
                "match_type": "rule-based"
            }

        # Step 2: ML intent classifier fallback
        cleaned = clean_text(user_msg)
        if self.model is not None and self.vectorizer is not None and len(cleaned) > 0:
            vec = self.vectorizer.transform([cleaned])
            probs = self.model.predict_proba(vec)[0]
            max_idx = int(np.argmax(probs))
            conf = float(probs[max_idx])
            predicted_tag = self.model.classes_[max_idx]

            if conf >= 0.35:
                # Find response for matched intent
                for intent in self.intents_data.get("intents", []):
                    if intent["tag"] == predicted_tag:
                        resp = np.random.choice(intent["responses"])
                        return {
                            "message": user_msg,
                            "intent": predicted_tag,
                            "response": str(resp),
                            "confidence": round(conf, 4),
                            "match_type": "ml-classifier"
                        }

        # Step 3: Default fallback
        fallback_resp = "I'm sorry, I didn't quite understand that. You can ask me about order tracking, returns, store hours, shipping, or speak to a live representative!"
        return {
            "message": user_msg,
            "intent": "fallback",
            "response": fallback_resp,
            "confidence": 0.20,
            "match_type": "fallback"
        }
