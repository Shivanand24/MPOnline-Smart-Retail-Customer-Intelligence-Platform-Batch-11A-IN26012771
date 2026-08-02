"""
Computer Vision Utility Module (cv_utils.py)
Provides reusable image preprocessing, feature extraction, face detection using OpenCV Haar Cascades,
and edge detection.
"""

import cv2
import numpy as np
import os
from typing import Tuple, List, Optional

# Load Haar Cascade classifier for face detection with fallback
face_cascade = None
try:
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if os.path.exists(cascade_path):
            face_cascade = cv2.CascadeClassifier(cascade_path)
except Exception:
    face_cascade = None

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts an RGB/BGR image to Grayscale."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def resize_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resizes an image to specified dimensions."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def apply_blur(image: np.ndarray, ksize: Tuple[int, int] = (5, 5)) -> np.ndarray:
    """Applies Gaussian Blur for noise reduction."""
    return cv2.GaussianBlur(image, ksize, 0)

def detect_edges(image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """Detects edges in an image using Canny Edge Detection."""
    gray = to_grayscale(image)
    blurred = apply_blur(gray)
    return cv2.Canny(blurred, low_threshold, high_threshold)

def detect_faces(image: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """
    Detects faces in an image using OpenCV Haar Cascades.
    Returns: (image_with_bounding_boxes, list_of_face_rectangles [x, y, w, h])
    """
    gray = to_grayscale(image)
    face_rects = []
    annotated_img = image.copy()
    
    if face_cascade is not None and not face_cascade.empty():
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_rects.append((int(x), int(y), int(w), int(h)))
            
    return annotated_img, face_rects

def extract_face_embedding(face_img: np.ndarray) -> np.ndarray:
    """
    Extracts a robust 128-dimensional feature embedding vector from a cropped face image.
    """
    if face_img is None or face_img.size == 0:
        return np.zeros(128, dtype=np.float32)
        
    face_resized = cv2.resize(face_img, (64, 64))
    gray = to_grayscale(face_resized)
    
    small = cv2.resize(gray, (8, 8)).flatten() / 255.0
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-7)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    mag_resized = cv2.resize(magnitude, (8, 4)).flatten() # 8x4 = 32 floats
    mag_resized = mag_resized / (mag_resized.max() + 1e-7)
    
    embedding = np.concatenate([small, hist, mag_resized]) # 64 + 32 + 32 = 128 floats
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding.astype(np.float32)
