from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Union
import numpy as np
import pandas as pd
import joblib
import uvicorn
import os
from datetime import datetime
import asyncio

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pack = joblib.load(r"model\models.joblib")
model = pack['model']
enc1 = pack['enc1']
enc2 = pack['enc2']

df = pd.read_csv(r"dataset\cleaned_dataset.csv")
df2 = pd.read_csv(r"dataset/mildly_unsafe_transactions.csv")

class Transaction_data(BaseModel):
    acc_holder: str
    features: list[Union[float , str]]

def changes_in_dataset(label, data):
    new_row = [len(df)] + [label] + list(data) + [datetime.now().strftime("%d-%m-%Y %H:%M:%S")]
    df.loc[len(df)] = new_row
    df.to_csv(r"dataset\cleaned_dataset.csv", index=False)

@app.api_route("/ping", methods=["GET", "HEAD"])
async def ping():
    await asyncio.sleep(0.1)
    return {"message": "server is running"}

@app.post("/predict")
async def predict(data: Transaction_data):
    acc_holder = data.acc_holder
    data=data.features
    if len(data) != 18:
        print("Missing features. Expected 18 features.")
        return {"ML error": "Missing features. Expected 18 features."}

    data[-2] = enc1.transform(str([data[-2]]))[0]
    data[-1] = enc2.transform(str([data[-1]]))[0]

    input_data = np.array(data).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    confidence = model.predict_proba(input_data)[0][prediction]

    if prediction == 1 and confidence > 0.85:
        label = "Fraud"
        fr_type = "Unsafe Transaction"
        row_exists = ((df.iloc[:, 2:20] == data).all(axis=1)).any()
        if row_exists:
            print("Row is present in the dataset so no changes made.")
        else:
            print("Row is not present in the dataset so updating the csv file.")
            changes_in_dataset(label, data)

    elif confidence > 0.65 and confidence < 0.85:
        label = "Non - Fraud"
        fr_type = "Mildly Unsafe Transaction"

        if acc_holder in df2["IDs"].values:
            label = "Fraud"
            fr_type = "Unsafe Transaction"
            df2 = df2[df2["IDs"] != acc_holder]
            df2.to_csv(r"dataset/mildly_unsafe_transactions.csv", index=False)
            changes_in_dataset(label, data)
        else:
            print("Putting the mildly unsafe transaction into monitoring dataset.")
            new_row2 = [len(df2)] + [acc_holder] + [datetime.now().strftime("%d-%m-%Y %H:%M:%S")]
            df2.loc[len(df2)] = new_row2
            df2.to_csv(r"dataset/mildly_unsafe_transactions.csv", index=False)
    else:
        label = "Non - Fraud"
        fr_type = "Safe Transaction"
    
    print(f"Confidence: {confidence*100:.2f}%")

    return {
        "prediction": label,
        "Type": fr_type
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="127.0.0.1", port=port)