"""
Unified ML Pipeline Module (pipeline.py)
Loads all trained Computer Vision, NLP, and Chatbot models once at system startup.
Provides clean unified interface for all services.
"""

from app.services.cv_service import ProductClassifierService, FaceRecognitionService
from app.services.nlp_service import SentimentAnalyzer
from app.services.chatbot_service import RetailChatbot

class UnifiedMLPipeline:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(UnifiedMLPipeline, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        print("[UnifiedMLPipeline] Initializing and loading all ML & CV models into memory...")
        self.product_classifier = ProductClassifierService()
        self.face_recognizer = FaceRecognitionService()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.chatbot = RetailChatbot()
        print("[UnifiedMLPipeline] All models successfully loaded!")

    def get_stats(self):
        # Aggregate dashboard stats across models
        total_customers = len(self.face_recognizer.face_database)
        total_visits = sum(c["visits"] for c in self.face_recognizer.face_database.values())
        return {
            "total_vip_customers": total_customers,
            "total_logged_visits": total_visits,
            "model_status": {
                "product_classifier": "Active",
                "face_recognition": "Active",
                "sentiment_analyzer": "Active",
                "faq_chatbot": "Active"
            }
        }

# Global Pipeline Singleton
pipeline = UnifiedMLPipeline()
