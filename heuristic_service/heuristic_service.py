# heuristic_service/heuristic_service.py
from fastapi import FastAPI
from pydantic import BaseModel
from brand_detector import detect_brand_from_label, init_ocr
from common.utils.image_reader import read_image


app = FastAPI()


class RequestModel(BaseModel):
    image_urls: list[str]


@app.on_event("startup")
def startup_event():
    # CPU-only, lang='en' keeps model small
    init_ocr(lang="en")


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/extract")
async def extract(req: RequestModel):
    # Simulated output with only one brand option

    img = read_image(req.image_urls[3])

    result = detect_brand_from_label(img)

    return {
        "attributes": {"brand": result["brand"]},
        "model_info": {"brand": "heuristic-ocr-fuzzy"}
    }
