# llm_service/llm_service.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class RequestModel(BaseModel):
    image_urls: list[str]


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/generate")
async def generate(req: RequestModel):
    desc = "A stylish second-hand item, versatile and in good condition."
    return {
        "attributes": {"description": desc},
        "model_info": {
            "description": "llm_model (simulated)"
        }
    }
