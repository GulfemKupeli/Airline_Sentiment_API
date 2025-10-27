# app.py - Streamlit application for BERT-based Sentiment Analysis
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import time
import re

MODEL_NAME = "theFeastofCrows/airline-sentiment-roberta" 
LABEL_NAMES = ['negative', 'neutral', 'positive'] 

@st.cache_resource
def load_model():
    """Loads the fine-tuned BERT model and tokenizer from the Hugging Face Hub."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
        model.to(device)
        model.eval() 
        st.success(f"Model successfully loaded on {device}.")
        return tokenizer, model, device
    except Exception as e:
        st.error(f"CRITICAL ERROR: Could not load model {MODEL_NAME}. Check Hugging Face repo name and files. Error: {e}")
        return None, None, None

# --- PREDICTION FUNCTION ---
def predict_sentiment(text_input, tokenizer, model, device):
    """Processes input text and returns the sentiment prediction."""
    if model is None:
        return "MODEL_NOT_LOADED"
    
    text_input = re.sub(r'http\S+|www\S+|https\S+', '', text_input)
    
    inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    prediction_index = torch.argmax(logits, dim=-1).item() 

    return LABEL_NAMES[prediction_index]

# --- STREAMLIT USER INTERFACE (UI) ---

st.title("✈️ Fine-Tuned BERT Airline Sentiment Analyzer")
st.markdown("Use a fine-tuned RoBERTa model to predict the emotional tone of airline-related tweets.")

tokenizer, model, device = load_model()

if model:
    tweet_input = st.text_area("Enter Tweet for Analysis:", 
                               "My flight was delayed by 6 hours. Customer service was horrible and I missed my connection.")
    
    if st.button("Predict Sentiment"):
        with st.spinner('Model is predicting...'):
            start_time = time.time()
            
            sentiment = predict_sentiment(tweet_input, tokenizer, model, device)
            
            end_time = time.time()
            
            st.subheader("Prediction Result:")
            
            if sentiment == 'negative':
                st.error(f"🔴 NEGATIVE (Inference time: {end_time - start_time:.2f} s)")
            elif sentiment == 'positive':
                st.success(f"🟢 POSITIVE (Inference time: {end_time - start_time:.2f} s)")
            else:
                st.warning(f"🟡 NEUTRAL (Inference time: {end_time - start_time:.2f} s)")

# --- SIDEBAR INFORMATION ---
st.sidebar.header("Model Details")
st.sidebar.markdown(f"**Model Name:** `{MODEL_NAME}`")
st.sidebar.markdown(f"**Device Used:** `{device}`")
st.sidebar.markdown("**Developed by:** theFeastofCrows")
st.sidebar.markdown("---")
st.sidebar.info("This model was fine-tuned on the Twitter US Airline Sentiment dataset for high performance across all sentiment classes.")


