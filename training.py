from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Fraud Detection API")

# Load model once at startup
model_path = os.path.join("model", "xgb_model.joblib")
model = joblib.load(model_path)

# Define request schema
class TransactionData(BaseModel):
    features: list[float]  # Must be 18 features

@app.post("/predict")
def predict(data: TransactionData):
    if len(data.features) != 18:
        return {"error": "Input must contain exactly 18 features."}

    # Convert input into numpy array and reshape for single sample
    input_data = np.array(data.features).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    label = "Fraud" if prediction == 1 else "Non - Fraud"

    return {
        "prediction": int(prediction),
        "label": label
    }
