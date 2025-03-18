from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import requests

# Create FastAPI instance
app = FastAPI()

# Load NLP models
summarizer = pipeline("summarization")
intent_model = pipeline("text-classification")

# Define Request Model
class TextRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Multi-API FastAPI Backend is Live!"}

# 📌 Summarization API
@app.post("/summarize")
def summarize(data: TextRequest):
    try:
        if not data.text:
            raise HTTPException(status_code=400, detail="No text provided")

        summary = summarizer(data.text, max_length=100, min_length=50, do_sample=False)
        return {"summary": summary[0]["summary_text"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 📌 Intent Detection API
@app.post("/intent")
def detect_intent(data: TextRequest):
    try:
        if not data.text:
            raise HTTPException(status_code=400, detail="No text provided")

        intent = intent_model(data.text)
        return {"intent": intent[0]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 📌 Fetch Data API
@app.get("/fetch")
def fetch_data():
    try:
        url = "https://jsonplaceholder.typicode.com/posts/1"  # Example API
        response = requests.get(url)
        return response.json()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))