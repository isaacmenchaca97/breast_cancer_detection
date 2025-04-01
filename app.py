import pickle
from typing import List

from fastapi import FastAPI, HTTPException
from loguru import logger
import numpy as np
from pydantic import BaseModel

app = FastAPI(
    title="Breast Cancer Detection API",
    description="API for predicting breast cancer using machine learning model",
    version="1.0.0",
)

# Load the model
try:
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    logger.exception(f"Error loading model: {e}")
    model = None


class PredictionInput(BaseModel):
    features: List[float]


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    message: str


@app.get("/")
async def root():
    return {"message": "Welcome to Breast Cancer Detection API"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convert input features to numpy array and reshape
        features = np.array(input_data.features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]
        probability = float(model.predict_proba(features)[0][1])

        # Determine message based on prediction
        message = "Malignant" if prediction == 1 else "Benign"

        return PredictionResponse(
            prediction=int(prediction), probability=probability, message=message
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
