"""
FastAPI Entrypoint (main.py)
AI-Powered Smart Retail & Customer Intelligence Platform API Gateway
Exposes computer vision, sentiment analysis, FAQ chatbot, and retail dashboard analytics endpoints.
"""

from fastapi import FastAPI, Header, HTTPException, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routers import vision, nlp, chatbot
from app.schemas import DashboardStatsResponse
from app.services.pipeline import pipeline

app = FastAPI(
    title="Smart Retail & Customer Intelligence Platform API",
    description="""
    Production-grade AI microservices platform integrating Computer Vision, NLP, and Chatbot analytics:
    - 📸 **Face Recognition & Loyalty Visit Tracking**: Detect returning VIP customers & log store visits.
    - 🛍️ **Product Image Classifier**: Classify retail items & map to physical store aisles.
    - 💬 **Customer Review Sentiment Analyzer**: Preprocess customer text & predict sentiment.
    - 🤖 **Retail FAQ Chatbot**: Rule-based + ML hybrid intent assistant.
    - 📊 **Executive Analytics API**: Aggregated business metrics.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for web dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional API Key Authentication Helper
API_KEY_SECRET = "smart-retail-api-key-2026"

def verify_api_key_optional(x_api_key: str = Header(None)):
    """Optional API Key header verification. Passes through if not set for easy testing."""
    if x_api_key is not None and x_api_key != API_KEY_SECRET:
        raise HTTPException(status_code=401, detail="Invalid API Key provided in header")
    return x_api_key

# Register routers
app.include_router(vision.router, dependencies=[Depends(verify_api_key_optional)])
app.include_router(nlp.router, dependencies=[Depends(verify_api_key_optional)])
app.include_router(chatbot.router, dependencies=[Depends(verify_api_key_optional)])

@app.get("/", summary="Root API Health Check & System Status")
async def root():
    return {
        "status": "online",
        "system": "AI-Powered Smart Retail & Customer Intelligence Platform",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": [
            "POST /recognize-face",
            "POST /register-face",
            "POST /classify-product",
            "POST /analyze-sentiment",
            "POST /chatbot",
            "GET /dashboard/stats"
        ]
    }

@app.get("/dashboard/stats", response_model=DashboardStatsResponse, summary="Executive Dashboard Retail Analytics")
async def get_dashboard_stats():
    """
    Returns aggregated metrics on VIP customer visits, system model health status, and store activity.
    """
    stats = pipeline.get_stats()
    return stats
