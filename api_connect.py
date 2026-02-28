from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Union, List
import numpy as np
import joblib
import os
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_
from db_init import SessionLocal, engine
from db_handling import Base, Transactions, MildlyUnsafeTransaction
from fault_reason import reason
import pandas as pd
import warnings

# --------------------------------------------------
# APP SETUP
# --------------------------------------------------

app = FastAPI()

@app.on_event("startup")
def startup():
    try:
        Base.metadata.create_all(bind=engine)
        print("Tables created successfully")
    except Exception as e:
        print(f"Database connection failed: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

try:
    pack = joblib.load(os.path.join("model", "models.joblib"))
    model_xgb = pack["xgb"]
    model_rf = pack["rf"]
    enc1 = pack["enc1"]
    enc2 = pack["enc2"]
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class TransactionData(BaseModel):
    acc_holder: str
    features: List[Union[float, str]]

# --------------------------------------------------
# DATABASE SESSION
# --------------------------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def safe_float(value):
    try:
        return float(value)
    except:
        return 0.0

def encode_value(encoder, value):
    if pd.isna(value) or value == "":
        value = "missing"
    try:
        return encoder.transform([value])[0]
    except:
        return encoder.transform(["missing"])[0]

def insert_transaction(prediction: float, data: list, db: Session):
    transaction = Transactions(
        flag=prediction,
        avg_min_between_sent_tnx=safe_float(data[0]),
        avg_min_between_received_tnx=safe_float(data[1]),
        time_diff_mins=safe_float(data[2]),
        sent_tnx=safe_float(data[3]),
        received_tnx=safe_float(data[4]),
        number_of_created_contracts=safe_float(data[5]),
        max_value_received=safe_float(data[6]),
        avg_val_received=safe_float(data[7]),
        avg_val_sent=safe_float(data[8]),
        total_ether_sent=safe_float(data[9]),
        total_ether_balance=safe_float(data[10]),
        erc20_total_ether_received=safe_float(data[11]),
        erc20_total_ether_sent=safe_float(data[12]),
        erc20_total_ether_sent_contract=safe_float(data[13]),
        erc20_uniq_sent_addr=safe_float(data[14]),
        erc20_uniq_rec_token_name=safe_float(data[15]),
        erc20_most_sent_token_type=str(data[16]),
        erc20_most_rec_token_type=str(data[17]),
        time=datetime.utcnow()
    )
    db.add(transaction)
    db.commit()

# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@app.get("/ping")
def ping():
    return {"message": "Server is running"}

@app.post("/predict")
def predict(data: TransactionData, db: Session = Depends(get_db)):

    if len(data.features) != 18:
        raise HTTPException(status_code=400, detail="Expected exactly 18 features")

    raw_features = ["missing" if pd.isna(x) else x for x in data.features]
    model_features = raw_features.copy()

    # Encode last two categorical
    model_features[16] = encode_value(enc1, model_features[16])
    model_features[17] = encode_value(enc2, model_features[17])

    # Convert numeric safely
    for i in range(16):
        model_features[i] = safe_float(model_features[i])

    input_array = np.array(model_features).reshape(1, -1)

    proba_xgb = model_xgb.predict_proba(input_array)

    try:
        proba_rf = model_rf.predict_proba(input_array)
    except Exception as e:
        warnings.warn(f"RF failed: {e}")
        proba_rf = proba_xgb

    avg_proba = (proba_xgb + proba_rf) / 2
    confidence = float(avg_proba[0][1])
    # Print debug info to terminal
    try:
        sklearn_version = pack.get("sklearn_version") if "pack" in globals() else None
    except Exception:
        sklearn_version = None

    print("[DEBUG] encoded_features:", model_features)
    print("[DEBUG] proba_xgb:", proba_xgb.tolist())
    print("[DEBUG] proba_rf:", proba_rf.tolist())
    print("[DEBUG] avg_proba:", avg_proba.tolist())
    print("[DEBUG] sklearn_version:", sklearn_version)
    prediction = 1.0 if confidence >= 0.5 else 0.0

    # --------------------------------------------------
    # DUPLICATE CHECK
    # --------------------------------------------------

    filters = [
        getattr(Transactions, col) == safe_float(raw_features[i])
        for i, col in enumerate([
            "avg_min_between_sent_tnx",
            "avg_min_between_received_tnx",
            "time_diff_mins",
            "sent_tnx",
            "received_tnx",
            "number_of_created_contracts",
            "max_value_received",
            "avg_val_received",
            "avg_val_sent",
            "total_ether_sent",
            "total_ether_balance",
            "erc20_total_ether_received",
            "erc20_total_ether_sent",
            "erc20_total_ether_sent_contract",
            "erc20_uniq_sent_addr",
            "erc20_uniq_rec_token_name"
        ])
    ]

    filters += [
        Transactions.erc20_most_sent_token_type == str(raw_features[16]),
        Transactions.erc20_most_rec_token_type == str(raw_features[17]),
        Transactions.flag == prediction
    ]

    row_exists = db.query(Transactions).filter(and_(*filters)).first()

    # --------------------------------------------------
    # DECISION
    # --------------------------------------------------

    if prediction == 1.0 and confidence > 0.85:
        label = "Fraud"
        fr_type = reason(str(raw_features))

        if not row_exists:
            insert_transaction(prediction, raw_features, db)

    elif 0.65 < confidence <= 0.85:
        label = "Non-Fraud"
        fr_type = f"Mildly Unsafe - {reason(str(raw_features))}"

    else:
        label = "Non-Fraud"
        fr_type = "Safe transaction"

    return {
        "prediction": label,
        "type": fr_type,
        "confidence": f"{confidence * 100:.2f}%"
    }

# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)