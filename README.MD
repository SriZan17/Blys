# Blys

This project implements an **end-to-end AI system** for analyzing customer behavior, generating personalized recommendations, and enabling intelligent customer interactions via an NLP-powered chatbot. The docs for the project can be found at: https://blys.onrender.com/docs

---

#  Features

##  1. Customer Behavior Analysis

* Data preprocessing (missing values, normalization)
* Sentiment analysis on customer reviews
* Optimal k value selection for clustering using elbow method and silhouette score
* Customer segmentation using clustering (K-Means)
* Identification of:

  * High-value customers and recommend retention strategies
  * Churn-risk customers and suggest engagement tactics
* Summary report provided in customer_analysis.md

---

##  2. Recommendation System

* Content-based recommendation model
* Uses:

  * Behavioral features (frequency, spending, recency)
  * Sentiment score
  * Service embeddings
* Model evaluation using various metrics
* Model saved as:

  ```
  recommendation_model.pkl
  ```
* Built using **scikit-learn** and optimized with **joblib**

---

## 3. AI Chatbot (NLP + Automation)

* Powered by OpenAI (LLM-based)
* Handles:

  * Booking queries
  * Cancellation requests
  * Rescheduling
  * Pricing inquiries

###  Key Capabilities

* Intent understanding (no manual training required)
* Entity extraction (date/time)
* Multi-turn conversations
* Function calling for real-world actions

Example:

```
User: Can I reschedule my booking?
AI: Sure! Please provide the new date and time.

User: 30 March 2025 at 10 am
AI: Reschedule request sent. You will be notified once confirmed.
```

* Chatbot configuration saved as:

  ```
  chatbot_model.pkl
  ```

---

##  4. FastAPI REST API

### Endpoints:

###  `/recommend`

**POST**

```json
{
  "customer_id": 1
}
```

**Response**

```json
{
  "customer_id": 1,
  "recommendations": ["Massage", "Facial"]
}
```

---

###  `/chatbot`

**POST**

```json
{
  "message": "Can I cancel my booking?",
  "history": []
}
```

**Response**

```json
{
  "response": "Your booking has been successfully cancelled."
}
```

---

# 🛠️ Tech Stack

* **Python**
* **FastAPI**
* **scikit-learn**
* **joblib**
* **OpenAI API (LLM for NLP)**
* **NumPy / Pandas**

---

# 📁 Project Structure

```
├── notebook.ipynb          # Data analysis + model training + chatbot setup
├── api.py                  # FastAPI application
├── recommendation_model.pkl
├── chatbot_model.pkl
├── requirements.txt
└── README.md
└── data/customer_data.csv
```

---

# Setup & Installation

## 1. Clone the Repository

```bash
git clone https://github.com/SriZan17/Blys
cd Blys
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Set OpenAI API Key

Set the value for OPENAI_API_KEY in the .env file.

---

## 4. Run the API

```bash
uvicorn api:app --reload
```

---

## 5. Access API Docs

Open:

```
http://127.0.0.1:8000/docs
```

---

#  Design Decisions

##  Why OpenAI for Chatbot?

* Eliminates need for large labeled datasets
* Handles intent detection + entity extraction in one step
* Enables function calling → real-world automation

##  Why joblib for Model?

* Efficient for NumPy-heavy ML models
* Faster loading than pickle
* Industry standard for scikit-learn

## Why FastAPI?

* High performance
* Built-in documentation (Swagger UI)
* Easy deployment

---

#  Future Improvements

* Persistent conversation memory (per user)
* Real-time database integration
* Advanced recommendation
* Deployment (Render / AWS / Docker)
* Authentication & rate limiting

---

#  Summary

This project demonstrates:

* End-to-end ML pipeline
* Applied NLP using modern LLMs
* API development for production use
* Clean, modular system design

---

#  Author

**Srijan Banet**

---

#  Notes

* Chatbot uses OpenAI API → requires API key
* Recommendation model currently uses sample input (can be extended with real data)
* Designed for clarity, scalability, and real-world applicability

---