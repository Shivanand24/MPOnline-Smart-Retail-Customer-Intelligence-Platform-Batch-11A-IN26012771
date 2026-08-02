"""
Chatbot Router Module (chatbot.py)
Exposes endpoint for:
- POST /chatbot (Interactive Retail Customer Service FAQ Chatbot)
"""

from fastapi import APIRouter, HTTPException, status
from app.schemas import ChatbotRequest, ChatbotResponse
from app.services.pipeline import pipeline

router = APIRouter(prefix="", tags=["Chatbot Module"])

@router.post("/chatbot", response_model=ChatbotResponse, summary="Retail AI Customer Support Assistant")
async def chatbot_endpoint(payload: ChatbotRequest):
    """
    Receives customer questions (e.g. order status, returns, shipping, store hours) and returns an AI bot response using hybrid intent matching.
    """
    if not payload.message or not payload.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message field cannot be empty."
        )
    result = pipeline.chatbot.get_response(payload.message)
    return result
