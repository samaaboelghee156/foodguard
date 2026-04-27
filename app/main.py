from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .schemas import AnalyzeFoodResponse
from .services.food_classifier import FoodClassifier

app = FastAPI(title="FoodGuard AI Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = FoodClassifier()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": classifier.model_name}


@app.post("/analyze-food", response_model=AnalyzeFoodResponse)
async def analyze_food(image: UploadFile = File(...)) -> AnalyzeFoodResponse:
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image file is empty.")

    try:
        return classifier.analyze(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode an image from the uploaded file. {exc}") from exc
