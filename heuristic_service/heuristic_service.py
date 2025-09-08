# heuristic_service/heuristic_service.py
from fastapi import FastAPI
from pydantic import BaseModel
from color_detector import detect_garment_color
from common.utils.image_reader import read_image


app = FastAPI()

class RequestModel(BaseModel):
    image_urls: list[str]

@app.post("/extract")
async def extract(req: RequestModel):

    image_front = req.image_urls[0]
    image_back = req.image_urls[1]

    color_name_front = detect_garment_color(read_image(image_front))
    color_name_back = detect_garment_color(read_image(image_back))

    if color_name_back == color_name_front:
        color_name = color_name_front
    else:
        color_name = "Ambiguous"

    return {
        "attributes": {"color": color_name},
        "model_info": {"color": "heuristic_model"}
    }
