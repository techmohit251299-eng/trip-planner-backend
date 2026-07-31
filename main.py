import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import anthropic

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


class TripRequest(BaseModel):
    destination: str
    budget: float
    days: int
    activities: List[str]


@app.get("/")
def health_check():
    return {"status": "Trip Planner backend is running"}


@app.post("/generate-trip")
def generate_trip(req: TripRequest):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set in Secrets")

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

Respond ONLY with valid JSON in this exact structure, no other text:
{{
  "budget_breakdown": {{"stay": 35, "food": 20, "travel": 20, "activities": 25}},
  "days": [
    {{"day": 1, "title": "...", "description": "...", "cost": 1500}}
  ]
}}"""

    try:
        message = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        raw_text = message.content[0].text.strip()

        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        parsed = json.loads(raw_text)
        return parsed

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Claude returned invalid JSON. Try again.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
