from fastapi import FastAPI, Depends
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

# --------------------------------------------------
# APP SETUP
# --------------------------------------------------

app = FastAPI()

# Create tables safely on startup
@app.on_event("startup")
def create_tables():
    Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# LOAD MODEL + ENCODERS
# --------------------------------------------------

pack = joblib.load(os.path.join("model", "models.joblib"))
# New ensemble keys: `xgb` and `rf` (RandomForest). Keep encoders names the same.
model_xgb = pack.get("xgb")
model_rf = pack.get("rf")
enc1 = pack["enc1"]
enc2 = pack["enc2"]

if model_xgb is None or model_rf is None:
    raise RuntimeError("Model package must contain 'xgb' and 'rf' keys")

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

def encode_value(encoder, value):
    try:
        if pd.isna(value) or value == "":
            value = "missing"
        return encoder.transform([value])[0]
    except ValueError:
        return encoder.transform(["missing"])[0]


def insert_transaction(prediction: float, data: list, db: Session):
    transaction = Transactions(
        flag=float(prediction),

        avg_min_between_sent_tnx=float(data[0]),
        avg_min_between_received_tnx=float(data[1]),
        time_diff_mins=float(data[2]),

        sent_tnx=float(data[3]),
        received_tnx=float(data[4]),
        number_of_created_contracts=float(data[5]),

        max_value_received=float(data[6]),
        avg_val_received=float(data[7]),
        avg_val_sent=float(data[8]),

        total_ether_sent=float(data[9]),
        total_ether_balance=float(data[10]),

        erc20_total_ether_received=float(data[11]),
        erc20_total_ether_sent=float(data[12]),
        erc20_total_ether_sent_contract=float(data[13]),

        erc20_uniq_sent_addr=float(data[14]),
        erc20_uniq_rec_token_name=float(data[15]),

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
        return {"error": "Expected exactly 18 features"}

    acc_holder = data.acc_holder

    # Replace NaN
    raw_features = ["missing" if pd.isna(x) else x for x in data.features]
    model_features = raw_features.copy()

    # Encode categorical
    model_features[-2] = encode_value(enc1, model_features[-2])
    model_features[-1] = encode_value(enc2, model_features[-1])

    # Convert numeric features to float
    for i in range(16):
        model_features[i] = float(model_features[i])

    # Predict using ensemble (average probabilities)
    input_array = np.array(model_features).reshape(1, -1)
    proba_xgb = model_xgb.predict_proba(input_array)

    try:
        proba_rf = model_rf.predict_proba(input_array)
    except Exception as e:
        import warnings
        warnings.warn(
            f"RandomForest predict_proba failed ({e}); falling back to XGB probabilities."
        )
        proba_rf = proba_xgb

    avg_proba = (proba_xgb + proba_rf) / 2.0
    prediction_idx = int((avg_proba[:, 1] >= 0.5)[0])
    prediction = float(prediction_idx)
    confidence = float(avg_proba[0][prediction_idx])

    # --------------------------------------------------
    # DUPLICATE CHECK
    # --------------------------------------------------

    filters = [
        Transactions.flag == prediction,
        Transactions.avg_min_between_sent_tnx == float(raw_features[0]),
        Transactions.avg_min_between_received_tnx == float(raw_features[1]),
        Transactions.time_diff_mins == float(raw_features[2]),
        Transactions.sent_tnx == float(raw_features[3]),
        Transactions.received_tnx == float(raw_features[4]),
        Transactions.number_of_created_contracts == float(raw_features[5]),
        Transactions.max_value_received == float(raw_features[6]),
        Transactions.avg_val_received == float(raw_features[7]),
        Transactions.avg_val_sent == float(raw_features[8]),
        Transactions.total_ether_sent == float(raw_features[9]),
        Transactions.total_ether_balance == float(raw_features[10]),
        Transactions.erc20_total_ether_received == float(raw_features[11]),
        Transactions.erc20_total_ether_sent == float(raw_features[12]),
        Transactions.erc20_total_ether_sent_contract == float(raw_features[13]),
        Transactions.erc20_uniq_sent_addr == float(raw_features[14]),
        Transactions.erc20_uniq_rec_token_name == float(raw_features[15]),
        Transactions.erc20_most_sent_token_type == str(raw_features[16]),
        Transactions.erc20_most_rec_token_type == str(raw_features[17]),
    ]

    row_exists = db.query(Transactions).filter(and_(*filters)).first() is not None

    # --------------------------------------------------
    # DECISION LOGIC
    # --------------------------------------------------

    if prediction == 1.0 and confidence > 0.85:
        label = "Fraud"
        fr_type = reason(str(raw_features))

        if not row_exists:
            insert_transaction(prediction, raw_features, db)

    elif 0.65 < confidence <= 0.85:
        label = "Non-Fraud"
        reason_text = reason(str(raw_features)) or "Unknown reason"
        fr_type = f"Mildly Unsafe Transaction - {reason_text}"

        count = db.query(MildlyUnsafeTransaction)\
                  .filter(MildlyUnsafeTransaction.acc_holder == acc_holder)\
                  .count()

        if count >= 2:
            db.query(MildlyUnsafeTransaction)\
              .filter(MildlyUnsafeTransaction.acc_holder == acc_holder)\
              .delete()
            db.commit()

            label = "Fraud"
            fr_type = "Receiver account flagged repeatedly."
        else:
            db.add(MildlyUnsafeTransaction(acc_holder=acc_holder))
            db.commit()

    else:
        label = "Non-Fraud"
        fr_type = "Safe transaction."

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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="127.0.0.1", port=port)