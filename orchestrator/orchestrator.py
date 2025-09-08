# orchestrator/orchestrator.py
from fastapi import FastAPI
from pydantic import BaseModel
import httpx, uuid, time
from db import init_db, log_request

app = FastAPI()

class RequestModel(BaseModel):
    image_urls: list[str]

@app.on_event("startup")
def startup_event():
    init_db()  # Ensure DB is ready

@app.post("/v1/items/analyze")
async def analyze_item(req: RequestModel):
    if len(req.image_urls) != 4:
        return {"error": "Exactly 4 image URLs required."}

    request_id = str(uuid.uuid4())
    start = time.time()

    attributes = {}
    model_info = {}

    async with httpx.AsyncClient() as client:
        # Vision model
        vision_resp = await client.post("http://vision-service:8000/classify", json=req.dict())
        vision_json = vision_resp.json()
        attributes.update(vision_json["attributes"])
        model_info.update(vision_json["model_info"])

        # Heuristic model
        heur_resp = await client.post("http://heuristic-service:8000/extract", json=req.dict())
        heur_json = heur_resp.json()
        attributes.update(heur_json["attributes"])
        model_info.update(heur_json["model_info"])

        # LLM model
        llm_resp = await client.post("http://llm-service:8000/generate", json=req.dict())
        llm_json = llm_resp.json()
        attributes.update(llm_json["attributes"])
        model_info.update(llm_json["model_info"])

    processing_time = round(time.time() - start, 2)

    response = {
        "id": request_id,
        "attributes": attributes,
        "model_info": model_info,
        "processing": {"time_taken": processing_time, "status": "success"}
    }

    # Log to SQLite
    log_request(request_id, req.image_urls, attributes, model_info, processing_time, "success")

    return response
