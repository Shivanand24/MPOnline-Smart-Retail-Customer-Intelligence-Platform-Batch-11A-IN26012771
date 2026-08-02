"""
Vision API Router (vision.py)
Exposes endpoints for:
- POST /recognize-face (Upload image -> returning customer ID & visit log)
- POST /register-face (Register new customer face profile)
- POST /classify-product (Upload product image -> category & store aisle location)
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from typing import Optional

from app.schemas import FaceRecognitionResponse, FaceRegisterResponse, ProductClassificationResponse
from app.services.pipeline import pipeline

router = APIRouter(prefix="", tags=["Computer Vision Module"])

def read_image_from_upload(file: UploadFile) -> np.ndarray:
    try:
        contents = file.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image.")
        return img
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image upload: {str(e)}"
        )

@router.post("/recognize-face", response_model=FaceRecognitionResponse, summary="Recognize Customer Face & Log Visit")
async def recognize_face_endpoint(file: UploadFile = File(..., description="Customer face image file (JPEG/PNG)")):
    """
    Uploads a webcam snapshot or camera feed image frame.
    Detects faces, generates feature embeddings, performs cosine similarity matching against registered VIP customer profiles, and logs visit timestamp.
    """
    image_np = read_image_from_upload(file)
    result = pipeline.face_recognizer.recognize_face(image_np)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@router.post("/register-face", response_model=FaceRegisterResponse, summary="Register New Customer Face Profile")
async def register_face_endpoint(
    customer_id: str = Form(..., description="Unique Customer Identifier, e.g. CUST-2001"),
    name: str = Form(..., description="Customer Full Name"),
    file: UploadFile = File(..., description="Customer facial photo")
):
    """
    Registers a new customer's facial profile into the store database for automated VIP loyalty recognition.
    """
    image_np = read_image_from_upload(file)
    result = pipeline.face_recognizer.register_customer_face(customer_id, name, image_np)
    return result

@router.post("/classify-product", response_model=ProductClassificationResponse, summary="Classify Product Image Category")
async def classify_product_endpoint(file: UploadFile = File(..., description="Product image file (JPEG/PNG)")):
    """
    Uploads a retail product image and predicts its category (Clothing, Shoes, Electronics, Bags, Groceries) along with store aisle placement recommendations.
    """
    image_np = read_image_from_upload(file)
    result = pipeline.product_classifier.classify_product_image(image_np)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
