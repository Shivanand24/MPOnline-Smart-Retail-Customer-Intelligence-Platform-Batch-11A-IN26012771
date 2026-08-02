"""
Pydantic Schemas Module (schemas.py)
Defines request and response data models for input validation and automatic Swagger API docs generation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# --- Face Recognition Schemas ---
class FaceRecognitionResponse(BaseModel):
    recognized: bool = Field(..., description="Whether the face was matched to a stored VIP customer profile")
    customer_id: str = Field(..., description="Unique customer identifier or GUEST ID")
    name: str = Field(..., description="Customer full name or guest label")
    similarity_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    visit_count: Optional[int] = Field(None, description="Total logged visits for returning customer")
    membership_tier: Optional[str] = Field(None, description="VIP Membership Tier status")
    timestamp: str = Field(..., description="ISO 8601 timestamp of face detection")

class FaceRegisterRequest(BaseModel):
    customer_id: str = Field(..., example="CUST-2001", description="New customer ID")
    name: str = Field(..., example="Alex Morgan", description="Full name of customer")

class FaceRegisterResponse(BaseModel):
    customer_id: str
    name: str
    status: str
    total_visits: int

# --- Product Classifier Schemas ---
class ProductClassificationResponse(BaseModel):
    predicted_category: str = Field(..., example="Electronics", description="Predicted retail category")
    confidence: float = Field(..., example=0.92, description="Model classification confidence score")
    store_aisle: str = Field(..., example="Aisle 1 - Consumer Tech & Gadgets", description="Recommended retail store aisle")
    timestamp: str = Field(..., description="ISO 8601 prediction timestamp")

# --- Sentiment Analysis Schemas ---
class SentimentRequest(BaseModel):
    text: str = Field(..., example="The product quality is exceptional! Fast delivery.", description="Raw customer review or message text")

class SentimentResponse(BaseModel):
    text: str
    cleaned_text: str
    sentiment: str = Field(..., example="positive", description="Sentiment classification: positive, neutral, or negative")
    confidence: float = Field(..., example=0.9412, description="Model prediction probability score")

# --- Chatbot Schemas ---
class ChatbotRequest(BaseModel):
    message: str = Field(..., example="What is your return policy?", description="User chat query or FAQ question")

class ChatbotResponse(BaseModel):
    message: str
    intent: str = Field(..., example="return_policy", description="Matched intent tag")
    response: str = Field(..., description="AI Chatbot bot response message")
    confidence: float = Field(..., example=0.98, description="Intent matching confidence score")
    match_type: str = Field(..., example="rule-based", description="Type of match: rule-based, ml-classifier, or fallback")

# --- Dashboard Stats Schema ---
class DashboardStatsResponse(BaseModel):
    total_vip_customers: int
    total_logged_visits: int
    model_status: Dict[str, str]
    system_version: str = "v1.0.0"
