# app.py - Gradio application for BERT-based Sentiment Analysis
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import re

# --- MODEL DEFINITION ---
MODEL_NAME = "theFeastofCrows/airline-sentiment-roberta" 
LABEL_NAMES = ['negative', 'neutral', 'positive'] 

# Detect device (GPU/CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables for model and tokenizer
TOKENIZER = None
MODEL = None

# --- MODEL LOADING (Function is called automatically when the Gradio Space starts) ---
def load_model():
    """Loads the model and tokenizer only once when the app initializes."""
    global TOKENIZER, MODEL
    if MODEL is None:
        try:
            TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
            MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
            MODEL.to(DEVICE)
            MODEL.eval()
            print(f"Model successfully loaded on {DEVICE}.")
        except Exception as e:
            print(f"CRITICAL ERROR: Could not load model {MODEL_NAME}. Error: {e}")
            raise e

# --- PREDICTION FUNCTION (Gradio interface uses this function) ---
def predict_sentiment_gradio(text_input):
    """Processes input text and returns the sentiment prediction string and color."""
    
    # 1. Load model if not already loaded (initial call)
    if MODEL is None:
        load_model()
        if MODEL is None:
             return "Model Loading Failed."

    # 2. Basic Cleanup
    text_input = re.sub(r'http\S+|www\S+|https\S+', '', text_input)
    
    # 3. Tokenization and Tensor conversion
    inputs = TOKENIZER(text_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # 4. Inference
    with torch.no_grad():
        outputs = MODEL(**inputs)

    logits = outputs.logits
    prediction_index = torch.argmax(logits, dim=-1).item() 
    sentiment = LABEL_NAMES[prediction_index]

    # 5. Return result with color coding
    if sentiment == 'negative':
        return f"🔴 NEGATIVE"
    elif sentiment == 'positive':
        return f"🟢 POSITIVE"
    else:
        return f"🟡 NEUTRAL"

# --- GRADI O INTERFACE SETUP ---

iface = gr.Interface(
    fn=predict_sentiment_gradio,
    inputs=gr.Textbox(lines=5, placeholder="Enter an airline tweet here...", label="Tweet Text"),
    outputs=gr.Textbox(label="Predicted Sentiment"),
    title="✈️ BERT Airline Sentiment Analyzer (Gradio)",
    description=f"Advanced NLP model fine-tuned on US airline tweets. Model: {MODEL_NAME}",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()




