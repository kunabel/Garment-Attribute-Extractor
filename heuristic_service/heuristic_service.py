# heuristic_service/heuristic_service.py
from fastapi import FastAPI
from pydantic import BaseModel
from common.utils.image_reader import read_image


app = FastAPI()

class RequestModel(BaseModel):
    image_urls: list[str]

@app.post("/extract")
async def extract(req: RequestModel):
    # Simulated output with only one brand option

    return {
        "attributes": {"brand": "BOSS"},
        "model_info": {"brand": "heuristic_model"}
    }
