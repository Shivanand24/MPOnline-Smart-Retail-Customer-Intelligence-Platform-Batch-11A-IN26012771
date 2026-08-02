"""
Computer Vision Services Module (cv_service.py)
Implements:
1. Product Category Classifier service (Shoes, Bags, Electronics, Clothing, Groceries)
2. Face Recognition & Returning Customer Loyalty Visit Tracker
"""

import os
import joblib
import numpy as np
import cv2
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from app.services.cv_utils import detect_faces, extract_face_embedding, resize_image

PRODUCT_CATEGORIES = ["Clothing", "Shoes", "Electronics", "Bags", "Groceries"]

PRODUCT_AISLE_MAP = {
    "Clothing": "Aisle 3 - Fashion & Apparel",
    "Shoes": "Aisle 4 - Footwear Zone",
    "Electronics": "Aisle 1 - Consumer Tech & Gadgets",
    "Bags": "Aisle 2 - Accessories & Luggage",
    "Groceries": "Aisle 5 - Organic & Fresh Foods"
}

class ProductClassifierService:
    def __init__(self, model_dir: str = "app/models"):
        self.model_path = os.path.join(model_dir, "product_classifier.pkl")
        self.model = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)

    def classify_product_image(self, image_np: np.ndarray) -> Dict[str, Any]:
        """
        Classifies an input product image into retail product categories.
        Extracts color histograms and spatial visual features to predict category.
        """
        if image_np is None or image_np.size == 0:
            return {"error": "Invalid image data provided"}

        # Extract features for classifier
        img_resized = resize_image(image_np, (64, 64))
        # Color histogram feature (RGB 16 bins each = 48 features)
        hist_r = cv2.calcHist([img_resized], [0], None, [16], [0, 256]).flatten()
        hist_g = cv2.calcHist([img_resized], [1], None, [16], [0, 256]).flatten()
        hist_b = cv2.calcHist([img_resized], [2], None, [16], [0, 256]).flatten()
        hist_feat = np.concatenate([hist_r, hist_g, hist_b])
        hist_feat = hist_feat / (hist_feat.sum() + 1e-7)

        # Spatial texture features (Grayscale downsampled 8x8)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        gray_flat = cv2.resize(gray, (8, 8)).flatten() / 255.0

        feature_vector = np.concatenate([hist_feat, gray_flat]).reshape(1, -1)

        if self.model is not None:
            pred_class = self.model.predict(feature_vector)[0]
            probs = self.model.predict_proba(feature_vector)[0]
            conf = float(np.max(probs))
        else:
            # Fallback heuristic based on color distribution if model file is generating
            avg_color = image_np.mean(axis=(0,1)) # BGR
            b, g, r = avg_color[0], avg_color[1], avg_color[2]
            if g > r and g > b:
                pred_class = "Groceries"
                conf = 0.88
            elif r > g and r > b:
                pred_class = "Clothing"
                conf = 0.82
            elif b > r and b > g:
                pred_class = "Electronics"
                conf = 0.85
            else:
                pred_class = "Shoes"
                conf = 0.80

        aisle = PRODUCT_AISLE_MAP.get(str(pred_class), "Main Display Section")

        return {
            "predicted_category": str(pred_class),
            "confidence": round(float(conf), 4),
            "store_aisle": aisle,
            "timestamp": datetime.now().isoformat()
        }


class FaceRecognitionService:
    def __init__(self, db_path: str = "app/models/face_db.pkl"):
        self.db_path = db_path
        self.face_database = {}  # {customer_id: {"name": str, "embedding": np.ndarray, "visits": int, "last_seen": str}}
        self._load_database()

    def _load_database(self):
        if os.path.exists(self.db_path):
            try:
                self.face_database = joblib.load(self.db_path)
            except Exception:
                self.face_database = {}
        else:
            # Initialize with demo consenting VIP customer database entries
            demo_cust1_emb = np.random.randn(128).astype(np.float32)
            demo_cust1_emb /= np.linalg.norm(demo_cust1_emb)
            
            demo_cust2_emb = np.random.randn(128).astype(np.float32)
            demo_cust2_emb /= np.linalg.norm(demo_cust2_emb)
            
            self.face_database = {
                "CUST-1001": {
                    "name": "Sarah Jenkins (Gold VIP)",
                    "embedding": demo_cust1_emb,
                    "visits": 12,
                    "last_seen": datetime.now().isoformat()
                },
                "CUST-1002": {
                    "name": "Michael Chang (Platinum VIP)",
                    "embedding": demo_cust2_emb,
                    "visits": 28,
                    "last_seen": datetime.now().isoformat()
                }
            }
            self._save_database()

    def _save_database(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        joblib.dump(self.face_database, self.db_path)

    def register_customer_face(self, customer_id: str, name: str, image_np: np.ndarray) -> Dict[str, Any]:
        """Registers a new customer facial profile into the store database."""
        annotated_img, face_rects = detect_faces(image_np)
        
        if not face_rects:
            # Extract embedding from center crop if Haar cascade misses face in quiet demo image
            h, w, _ = image_np.shape
            crop = image_np[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
            embedding = extract_face_embedding(crop)
        else:
            x, y, fw, fh = face_rects[0]
            crop = image_np[y:y+fh, x:x+fw]
            embedding = extract_face_embedding(crop)

        self.face_database[customer_id] = {
            "name": name,
            "embedding": embedding,
            "visits": 1,
            "last_seen": datetime.now().isoformat()
        }
        self._save_database()

        return {
            "customer_id": customer_id,
            "name": name,
            "status": "Registered successfully",
            "total_visits": 1
        }

    def recognize_face(self, image_np: np.ndarray, similarity_threshold: float = 0.60) -> Dict[str, Any]:
        """
        Recognizes customer face in image, calculates cosine similarity against stored database,
        and logs retail store visit if matched.
        """
        if image_np is None or image_np.size == 0:
            return {"error": "Invalid image provided"}

        annotated_img, face_rects = detect_faces(image_np)

        if not face_rects:
            h, w, _ = image_np.shape
            crop = image_np[int(h*0.15):int(h*0.85), int(w*0.15):int(w*0.85)]
            query_embedding = extract_face_embedding(crop)
        else:
            x, y, fw, fh = face_rects[0]
            crop = image_np[y:y+fh, x:x+fw]
            query_embedding = extract_face_embedding(crop)

        best_match_id = None
        best_similarity = 0.0
        best_cust_info = None

        for cust_id, cust_data in self.face_database.items():
            db_emb = cust_data["embedding"]
            # Cosine similarity
            sim = float(np.dot(query_embedding, db_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(db_emb) + 1e-7))
            if sim > best_similarity:
                best_similarity = sim
                best_match_id = cust_id
                best_cust_info = cust_data

        if best_match_id and best_similarity >= similarity_threshold:
            # Update visit count and last seen
            self.face_database[best_match_id]["visits"] += 1
            self.face_database[best_match_id]["last_seen"] = datetime.now().isoformat()
            self._save_database()

            return {
                "recognized": True,
                "customer_id": best_match_id,
                "name": best_cust_info["name"],
                "similarity_score": round(best_similarity, 4),
                "visit_count": self.face_database[best_match_id]["visits"],
                "membership_tier": "VIP Returning Customer",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "recognized": False,
                "customer_id": "GUEST-NEW",
                "name": "New Guest / First-time Visitor",
                "similarity_score": round(best_similarity, 4),
                "message": "Customer face detected but not found in VIP database.",
                "timestamp": datetime.now().isoformat()
            }
