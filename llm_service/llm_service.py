# llm_service/llm_service.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class RequestModel(BaseModel):
    image_urls: list[str]

@app.post("/generate")
async def generate(req: RequestModel):
    desc = "A stylish second-hand item, versatile and in good condition."
    return {
        "attributes": {"description": desc, "style": "casual", "fit": "regular"},
        "model_info": {
            "description": "llm_model (simulated)",
            "style": "llm_model (simulated)",
            "fit": "llm_model (simulated)"
        }
    }
