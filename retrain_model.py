import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import sklearn
from sklearn.ensemble import RandomForestClassifier

from db_init import SessionLocal
from db_handling import Transactions

warnings.filterwarnings("ignore")

print("🔄 Starting model retraining...")

# --------------------------------------------------
# 1️⃣ LOAD DATA FROM DATABASE
# --------------------------------------------------

session = SessionLocal()
df = pd.read_sql(session.query(Transactions).statement, session.bind)
session.close()

if df.empty:
    raise ValueError("❌ Database returned empty dataset. Cannot retrain model.")

print(f"✅ Loaded dataset with shape: {df.shape}")

# --------------------------------------------------
# 2️⃣ EXACT FEATURE ORDER (MUST MATCH FASTAPI)
# --------------------------------------------------

feature_columns = [
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
    "erc20_uniq_rec_token_name",
    "erc20_most_sent_token_type",
    "erc20_most_rec_token_type",
]

required_columns = feature_columns + ["flag"]

missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"❌ Missing required columns in DB: {missing_cols}")

# Keep only required columns in exact order
df = df[feature_columns + ["flag"]]

# --------------------------------------------------
# 3️⃣ HANDLE NUMERIC FEATURES (FIRST 16)
# --------------------------------------------------

numeric_cols = feature_columns[:16]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# --------------------------------------------------
# 4️⃣ ENCODE CATEGORICAL FEATURES (LAST 2)
# --------------------------------------------------

cat_col_1 = feature_columns[16]
cat_col_2 = feature_columns[17]

df[cat_col_1] = df[cat_col_1].fillna("missing").astype(str)
df[cat_col_2] = df[cat_col_2].fillna("missing").astype(str)

enc1 = LabelEncoder()
enc2 = LabelEncoder()

df[cat_col_1] = enc1.fit_transform(df[cat_col_1])
df[cat_col_2] = enc2.fit_transform(df[cat_col_2])

# --------------------------------------------------
# 5️⃣ SPLIT FEATURES & TARGET
# --------------------------------------------------

X = df[feature_columns]
y = df["flag"].astype(int)

if y.nunique() < 2:
    raise ValueError("❌ Only one class present. Cannot train classifier.")

# --------------------------------------------------
# 6️⃣ HANDLE CLASS IMBALANCE
# --------------------------------------------------

class_counts = y.value_counts()
neg = class_counts.get(0, 0)
pos = class_counts.get(1, 0)

scale_pos_weight = neg / pos if pos > 0 else 1

# --------------------------------------------------
# 7️⃣ TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

# --------------------------------------------------
# 8️⃣ TRAIN MODEL
# --------------------------------------------------

print("🧠 Training model...")

# Train XGBoost
model_xgb = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss",
)

model_xgb.fit(X_train, y_train)

# Train Random Forest (class_weight helps with imbalance)
model_rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)

model_rf.fit(X_train, y_train)

# --------------------------------------------------
# 9️⃣ EVALUATE MODEL
# --------------------------------------------------

# Ensemble: average predicted probabilities from both models
proba_xgb = model_xgb.predict_proba(X_test)
proba_rf = model_rf.predict_proba(X_test)

avg_proba = (proba_xgb + proba_rf) / 2.0
y_pred = (avg_proba[:, 1] >= 0.5).astype(int)

print("\n📊 Classification Report (Ensemble):\n")
print(classification_report(y_test, y_pred))

accuracy = (y_pred == y_test.values).mean() * 100
print(f"\n✅ Ensemble Accuracy: {accuracy:.2f}%")

# --------------------------------------------------
# 🔟 SAVE MODEL + ENCODERS
# --------------------------------------------------

save_path = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(save_path, exist_ok=True)

model_package = {
    "xgb": model_xgb,
    "rf": model_rf,
    "enc1": enc1,
    "enc2": enc2,
    "sklearn_version": sklearn.__version__,
}

joblib.dump(model_package, os.path.join(save_path, "models.joblib"))

print("💾 Model saved successfully.")
print("🎉 Retraining completed successfully.")