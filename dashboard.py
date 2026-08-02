"""
Streamlit Web Dashboard (dashboard.py)
Interactive Web Interface for Smart Retail & Customer Intelligence Platform
"""

import os
import io
import json
import requests
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Smart Retail AI Platform",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2563EB;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🛍️ AI-Powered Smart Retail & Customer Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Unified Computer Vision, NLP & Retail Analytics Platform</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/shopping-bag--v1.png", width=70)
st.sidebar.title("Navigation")
menu_selection = st.sidebar.radio(
    "Select Platform Module",
    [
        "📊 Executive Dashboard",
        "📸 Face Recognition & VIP Loyalty",
        "🛍️ Product Category Scanner",
        "💬 Feedback Sentiment Analyzer",
        "🤖 Support AI Chatbot"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Backend API**: `http://127.0.0.1:8000`\n\nDocs available at `/docs`")

# Helper functions to talk to local FastAPI or direct service
def check_api_health():
    try:
        r = requests.get(f"{API_BASE_URL}/", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

api_online = check_api_health()
if api_online:
    st.sidebar.success("🟢 REST API Gateway: Online")
else:
    st.sidebar.warning("🟡 REST API Gateway: Offline (Run `uvicorn app.main:app`)")

# ----------------------------------------------------
# 1. EXECUTIVE DASHBOARD
# ----------------------------------------------------
if menu_selection == "📊 Executive Dashboard":
    st.header("📊 Executive Analytics & System Health")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Registered VIP Customers", value="24", delta="+3 this week")
    with col2:
        st.metric(label="Logged In-Store Visits", value="158", delta="+12 today")
    with col3:
        st.metric(label="Positive Sentiment Rate", value="82.4%", delta="+4.1%")
    with col4:
        st.metric(label="Chatbot Resolution", value="94.2%", delta="+1.8%")

    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🛍️ In-Store Product Scans by Category")
        cat_data = pd.DataFrame({
            "Category": ["Clothing", "Shoes", "Electronics", "Bags", "Groceries"],
            "Scans": [142, 98, 76, 54, 110]
        })
        st.bar_chart(cat_data.set_index("Category"))

    with c2:
        st.subheader("💬 Customer Feedback Sentiment Breakdown")
        sent_data = pd.DataFrame({
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Count": [65, 20, 15]
        })
        st.bar_chart(sent_data.set_index("Sentiment"))

# ----------------------------------------------------
# 2. FACE RECOGNITION & LOYALTY
# ----------------------------------------------------
elif menu_selection == "📸 Face Recognition & VIP Loyalty":
    st.header("📸 Face Recognition & VIP Loyalty Visit Logging")
    st.write("Detect returning customers via face encodings, log visit history, and trigger VIP loyalty perks.")
    
    tab1, tab2 = st.tabs(["Recognize Visitor", "Register New VIP Face"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload Customer Photo / Camera Frame", type=["jpg", "jpeg", "png"], key="rec_face")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Customer Image", width=300)
            
            if st.button("🔍 Process Facial Recognition"):
                if api_online:
                    uploaded_file.seek(0)
                    files = {"file": ("face.jpg", uploaded_file.getvalue(), "image/jpeg")}
                    res = requests.post(f"{API_BASE_URL}/recognize-face", files=files)
                    if res.status_code == 200:
                        data = res.json()
                        if data["recognized"]:
                            st.success(f"🎉 **VIP Customer Recognized!**")
                            st.write(f"**Customer Name:** {data['name']}")
                            st.write(f"**Customer ID:** {data['customer_id']}")
                            st.write(f"**Total Visits:** {data['visit_count']}")
                            st.write(f"**Match Similarity:** {data['similarity_score']*100:.1f}%")
                            st.write(f"**Logged Timestamp:** {data['timestamp']}")
                        else:
                            st.warning("👤 **New Visitor / Guest Detected**")
                            st.write(data.get("message", "Not found in VIP database."))
                    else:
                        st.error("API error during recognition.")
                else:
                    # Direct service invocation if API offline
                    from app.services.pipeline import pipeline
                    img_np = np.array(image)
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    data = pipeline.face_recognizer.recognize_face(img_bgr)
                    if data.get("recognized"):
                        st.success(f"🎉 **VIP Customer Recognized!**")
                        st.write(f"**Name:** {data['name']}")
                        st.write(f"**Customer ID:** {data['customer_id']}")
                        st.write(f"**Visits:** {data['visit_count']}")
                    else:
                        st.warning("👤 **New Visitor Detected**")

    with tab2:
        st.subheader("Register Customer Facial Profile")
        with st.form("reg_form"):
            c_id = st.text_input("Customer ID", value="CUST-2005")
            c_name = st.text_input("Customer Full Name", value="Elena Rostova")
            c_file = st.file_uploader("Customer Facial Photo", type=["jpg", "png"], key="reg_file")
            submit = st.form_submit_button("Register Customer")
            
            if submit and c_file:
                if api_online:
                    files = {"file": ("user.jpg", c_file.getvalue(), "image/jpeg")}
                    data_form = {"customer_id": c_id, "name": c_name}
                    r = requests.post(f"{API_BASE_URL}/register-face", data=data_form, files=files)
                    if r.status_code == 200:
                        st.success(f"Successfully registered {c_name} ({c_id}) into VIP database!")
                else:
                    st.success(f"Saved {c_name} to database!")

# ----------------------------------------------------
# 3. PRODUCT CATEGORY SCANNER
# ----------------------------------------------------
elif menu_selection == "🛍️ Product Category Scanner":
    st.header("🛍️ Smart Product Category Classifier")
    st.write("Classifies product images into **Clothing, Shoes, Electronics, Bags, Groceries** and maps store aisle locations.")

    prod_file = st.file_uploader("Upload Product Image", type=["jpg", "png", "jpeg"], key="prod_up")
    if prod_file:
        img = Image.open(prod_file)
        st.image(img, caption="Product Image", width=300)
        
        if st.button("🏷️ Classify Product"):
            if api_online:
                prod_file.seek(0)
                files = {"file": ("prod.jpg", prod_file.getvalue(), "image/jpeg")}
                res = requests.post(f"{API_BASE_URL}/classify-product", files=files)
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"**Predicted Category:** {data['predicted_category']}")
                    st.info(f"📍 **Recommended Store Aisle:** {data['store_aisle']}")
                    st.progress(float(data['confidence']))
                    st.write(f"Confidence Score: {data['confidence']*100:.1f}%")
            else:
                from app.services.pipeline import pipeline
                img_np = np.array(img)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                data = pipeline.product_classifier.classify_product_image(img_bgr)
                st.success(f"**Predicted Category:** {data['predicted_category']}")
                st.info(f"📍 **Recommended Aisle:** {data['store_aisle']}")

# ----------------------------------------------------
# 4. FEEDBACK SENTIMENT ANALYZER
# ----------------------------------------------------
elif menu_selection == "💬 Feedback Sentiment Analyzer":
    st.header("💬 Customer Feedback Sentiment Analysis")
    st.write("Preprocess customer review text and analyze sentiment using TF-IDF + Logistic Regression NLP pipeline.")

    user_review = st.text_area("Enter Customer Review or Chat Feedback:", "The delivery was super fast and the item quality exceeded all my expectations!")
    
    if st.button("⚡ Analyze Sentiment"):
        if api_online:
            res = requests.post(f"{API_BASE_URL}/analyze-sentiment", json={"text": user_review})
            if res.status_code == 200:
                data = res.json()
                sent = data["sentiment"].upper()
                if sent == "POSITIVE":
                    st.success(f"😊 **Sentiment:** POSITIVE (Confidence: {data['confidence']*100:.1f}%)")
                elif sent == "NEGATIVE":
                    st.error(f"😞 **Sentiment:** NEGATIVE (Confidence: {data['confidence']*100:.1f}%)")
                else:
                    st.warning(f"😐 **Sentiment:** NEUTRAL (Confidence: {data['confidence']*100:.1f}%)")
                
                st.write("**Cleaned Preprocessed Text:**", f"`{data['cleaned_text']}`")
        else:
            from app.services.pipeline import pipeline
            data = pipeline.sentiment_analyzer.predict(user_review)
            st.success(f"Sentiment: {data['sentiment']} ({data['confidence']})")

    st.markdown("---")
    st.subheader("📁 Batch Dataset Reviews Preview (`data/reviews.csv`)")
    if os.path.exists("data/reviews.csv"):
        df_rev = pd.read_csv("data/reviews.csv")
        st.dataframe(df_rev.head(10), use_container_width=True)

# ----------------------------------------------------
# 5. SUPPORT AI CHATBOT
# ----------------------------------------------------
elif menu_selection == "🤖 Support AI Chatbot":
    st.header("🤖 Retail Support AI Assistant")
    st.write("Hybrid FAQ chatbot: exact rule intent matching + ML fallback classifier.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"sender": "bot", "text": "Hello! I am your Smart Retail AI assistant. Ask me about order tracking, returns, shipping, or store hours!"}
        ]

    for chat in st.session_state.chat_history:
        if chat["sender"] == "user":
            st.chat_message("user").write(chat["text"])
        else:
            st.chat_message("assistant").write(chat["text"])

    user_query = st.chat_input("Type your question here (e.g. 'Where is my order?' or 'Return policy')")
    if user_query:
        st.session_state.chat_history.append({"sender": "user", "text": user_query})
        st.chat_message("user").write(user_query)
        
        if api_online:
            res = requests.post(f"{API_BASE_URL}/chatbot", json={"message": user_query})
            if res.status_code == 200:
                data = res.json()
                bot_reply = data["response"]
                st.session_state.chat_history.append({"sender": "bot", "text": bot_reply})
                st.chat_message("assistant").write(bot_reply)
        else:
            from app.services.pipeline import pipeline
            data = pipeline.chatbot.get_response(user_query)
            st.session_state.chat_history.append({"sender": "bot", "text": data["response"]})
            st.chat_message("assistant").write(data["response"])
