"""
NLP Router Module (nlp.py)
Exposes endpoint for:
- POST /analyze-sentiment (Analyze customer feedback / review sentiment)
"""

from fastapi import APIRouter, HTTPException, status
from app.schemas import SentimentRequest, SentimentResponse
from app.services.pipeline import pipeline

router = APIRouter(prefix="", tags=["NLP Module"])

@router.post("/analyze-sentiment", response_model=SentimentResponse, summary="Analyze Customer Feedback Sentiment")
async def analyze_sentiment_endpoint(payload: SentimentRequest):
    """
    Cleans raw customer review text and classifies sentiment as positive, neutral, or negative using a TF-IDF + Logistic Regression model.
    """
    if not payload.text or not payload.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text field cannot be empty."
        )
    result = pipeline.sentiment_analyzer.predict(payload.text)
    return result
