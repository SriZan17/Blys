from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
from openai import OpenAI
import numpy as np
import pickle
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load Models
try:
    model_payload = joblib.load("recommendation_model.pkl")
    regressor = model_payload["regressor"]
    embedding_map = model_payload["embedding_map"]
    feature_names = model_payload["features"]
except:
    regressor = None

with open("chatbot_model.pkl", "rb") as f:
    chatbot_config = pickle.load(f)

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    api_key = api_key.strip('\"\'')
client = OpenAI(api_key=api_key)
analyzer = SentimentIntensityAnalyzer()

# App Init
app = FastAPI(title="Blys API")

# Schemas
class RecommendRequest(BaseModel):
    customer_id: int

class ChatRequest(BaseModel):
    message: str
    history: list = []

def get_sentiment(text):
    return analyzer.polarity_scores(text)['compound']

# Recommendation Logic
def get_recommendations(customer_id):
    #get necessary features for the customer_id from data/customer_data.csv
    df = pd.read_csv("data/customer_data.csv")
    customer_data = df[df["Customer_ID"] == customer_id].copy()
    
    if customer_data.empty:
        return []
        
    #calculate recency and sentiment_score
    customer_data["Recency"] = (datetime.now() - pd.to_datetime(customer_data["Last_Activity"])).dt.days
    customer_data["Sentiment_Score"] = customer_data["Review_Text"].astype(str).apply(get_sentiment)
    customer_features = customer_data[feature_names]
    
    pred_vector = model_payload["regressor"].predict(customer_features)[0]
    
    # Find closest service embeddings
    distances = {s: np.linalg.norm(pred_vector - emb) for s, emb in embedding_map.items()}
    sorted_services = sorted(distances, key=distances.get)
    return sorted_services[:2]

def reschedule_booking(date_time: str):
    #call api for rescheduling
    return f"Reschedule request sent for {date_time}. You will be notified once confirmed."

def cancel_booking():
    #call api for cancelling
    return "Your booking has been successfully cancelled."

def get_pricing():
    #call api for pricing
    return "Massage starts from $50. Final price depends on the service."

# Tool Definitions
tools = [
    {
        "type": "function",
        "function": {
            "name": "reschedule_booking",
            "description": "Reschedule a booking",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_time": {"type": "string"}
                },
                "required": ["date_time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_booking",
            "description": "Cancel a booking",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_pricing",
            "description": "Get pricing information",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

# Chatbot 
def run_chatbot(user_message, history):
    messages = [
        {"role": "system", "content": chatbot_config["system_prompt"]}
    ]

    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=chatbot_config["model"],
        messages=messages,
        tools=tools
    )

    msg = response.choices[0].message

    if msg.tool_calls:
        tool_call = msg.tool_calls[0]
        function_name = tool_call.function.name
        arguments = eval(tool_call.function.arguments)

        if function_name == "reschedule_booking":
            result = reschedule_booking(**arguments)
        elif function_name == "cancel_booking":
            result = cancel_booking()
        elif function_name == "get_pricing":
            result = get_pricing()
        else:
            result = "Unknown action."

        return {"response": result}

    return {"response": msg.content}

# Endpoints
@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/recommend")
def recommend(req: RecommendRequest):
    recs = get_recommendations(req.customer_id)
    return {"customer_id": req.customer_id, "recommendations": recs}

@app.post("/chatbot")
def chatbot(req: ChatRequest):
    return run_chatbot(req.message, req.history)