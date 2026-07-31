import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import google.generativeai as genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-flash-latest")


class TripRequest(BaseModel):
    destination: str
    budget: float
    days: int
    activities: List[str]


@app.get("/")
def health_check():
    return {"status": "Trip Planner backend is running (Gemini)"}


@app.post("/generate-trip")
def generate_trip(req: TripRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set in Secrets")

    activities_text = ", ".join(req.activities) if req.activities else "a good general mix"

    prompt = f"""You are a travel planning assistant. Create a realistic {req.days}-day trip itinerary
for {req.destination} with a total budget of ₹{req.budget}.

The traveler wants to include these activities/themes: {activities_text}.

For each day, provide:
- A short title for the day
- A 1-2 sentence description (realistic, specific to {req.destination}, not generic)
- An estimated cost in INR for that day (numbers only, no currency symbol)

Also provide an overall budget breakdown across these 4 categories: stay, food, travel, activities
(as approximate percentages that add up to 100).

Respond ONLY with valid JSON in this exact structure, no other text, no markdown code fences:
{{
  "budget_breakdown": {{"stay": 35, "food": 20, "travel": 20, "activities": 25}},
  "days": [
    {{"day": 1, "title": "...", "description": "...", "cost": 1500}}
  ]
}}"""

    try:
        response = model.generate_content(prompt)
        raw_text = response.text.strip()

        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        parsed = json.loads(raw_text)
        return parsed

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Gemini returned invalid JSON. Try again.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
