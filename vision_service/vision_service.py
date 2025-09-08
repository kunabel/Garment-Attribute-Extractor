# vision_service/vision_service.py
from fastapi import FastAPI
from pydantic import BaseModel
import random

app = FastAPI()

class RequestModel(BaseModel):
    image_urls: list[str]

@app.post("/classify")
async def classify(req: RequestModel):
    categories = ["t-shirt", "dress", "jacket", "jeans"]
    conditions = ["new", "like_new", "good", "fair", "poor"]

    return {
        "attributes": {
            "category": random.choice(categories),
            "condition": random.choice(conditions)
        },
        "model_info": {
            "category": "vision_classifier (simulated)",
            "condition": "vision_classifier (simulated)"
        }
    }
