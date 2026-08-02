"""
Automated REST API & ML Model Test Suite (test_endpoints.py)
Tests all system endpoints using FastAPI TestClient:
- Health check & dashboard stats
- Face recognition & registration
- Product image classification
- Sentiment analysis
- Retail FAQ Chatbot intent matcher
"""

import os
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def create_dummy_image_bytes(color=(100, 150, 200), width=100, height=100) -> bytes:
    """Creates a sample dummy JPEG image buffer for testing file uploads."""
    img = np.full((height, width, 3), color, dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "documentation" in data

def test_dashboard_stats_endpoint():
    response = client.get("/dashboard/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_vip_customers" in data
    assert "total_logged_visits" in data
    assert data["model_status"]["face_recognition"] == "Active"

def test_analyze_sentiment_positive():
    response = client.post("/analyze-sentiment", json={"text": "The product quality is exceptional! Fast delivery."})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ["positive", "neutral", "negative"]
    assert "confidence" in data

def test_analyze_sentiment_empty():
    response = client.post("/analyze-sentiment", json={"text": "   "})
    assert response.status_code == 400

def test_chatbot_intent_matching():
    response = client.post("/chatbot", json={"message": "What are your store hours?"})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] in ["store_hours", "greeting", "thanks", "fallback"]
    assert len(data["response"]) > 0

def test_classify_product_endpoint():
    img_bytes = create_dummy_image_bytes(color=(0, 200, 0)) # Greenish for groceries test
    response = client.post(
        "/classify-product",
        files={"file": ("product.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_category" in data
    assert "store_aisle" in data
    assert "confidence" in data

def test_face_recognition_guest():
    img_bytes = create_dummy_image_bytes(color=(128, 128, 128))
    response = client.post(
        "/recognize-face",
        files={"file": ("face.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "recognized" in data
    assert "customer_id" in data

def test_face_registration():
    img_bytes = create_dummy_image_bytes(color=(220, 180, 140))
    response = client.post(
        "/register-face",
        data={"customer_id": "TEST-CUST-999", "name": "Test User"},
        files={"file": ("user.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "TEST-CUST-999"
    assert data["status"] == "Registered successfully"
