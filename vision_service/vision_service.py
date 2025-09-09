# vision_service/vision_service.py
from fastapi import FastAPI
from pydantic import BaseModel
from vision_classifier import predict_fashion_attributes_from_cv, load_model, DEVICE
from common.utils.image_reader import read_image

MODEL_NAME = "patrickjohncyh/fashion-clip"

app = FastAPI()

# global state
model = None
processor = None

class RequestModel(BaseModel):
    image_urls: list[str]


# Pre-loading model on application start to make post event quicker
@app.on_event("startup")
def startup_event():
    global model, processor
    model, processor = load_model(MODEL_NAME, DEVICE)


@app.post("/classify")
async def classify(req: RequestModel):

    img = read_image(req.image_urls[0])

    results = predict_fashion_attributes_from_cv(
        img,
        top_k=5,
        refine_with_category=True,
        model=model,
        processor=processor,
        device=DEVICE
    )

    return {
        "attributes": {
            "category": results["category"].top1[0],
            "color": results["color"].top1[0],
            "material": results["material"].top1[0],
            "condition": results["condition"].top1[0],
            "style": results["style"].top1[0],
            "gender": results["gender"].top1[0],
            "season": results["season"].top1[0],
            "pattern": results["pattern"].top1[0],
            "sleeve length": results["sleeve length"].top1[0],
            "neckline": results["neckline"].top1[0],
            "closure type": results["closure type"].top1[0],
            "fit": results["fit"].top1[0]
        },
        "model_info": {
            "category": "vision_classifier",
            "color": "vision_classifier",
            "material": "vision_classifier",
            "condition": "vision_classifier",
            "style": "vision_classifier",
            "gender": "vision_classifier",
            "season": "vision_classifier",
            "pattern": "vision_classifier",
            "sleeve length": "vision_classifier",
            "neckline": "vision_classifier",
            "closure type": "vision_classifier",
            "fit": "vision_classifier"
        }
    }
