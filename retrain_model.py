import os
import re
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from db_init import SessionLocal
from db_handling import Transactions

warnings.filterwarnings("ignore")

print("🔄 Starting model retraining...")

# -----------------------------
# 1️⃣ Load Data from Database
# -----------------------------
session = SessionLocal()
df = pd.read_sql(session.query(Transactions).statement, session.bind)
session.close()

if df.empty:
    raise ValueError("❌ Database returned empty dataset. Cannot retrain model.")

print(f"✅ Loaded dataset with shape: {df.shape}")

# -----------------------------
# 2️⃣ Normalize & Clean Columns
# -----------------------------
# Normalize column names (lowercase + underscores)
df.columns = [re.sub(r"[^0-9a-z]", "_", c.lower()).strip("_") for c in df.columns]

# Replace empty strings with NaN
df = df.replace(r"^\s*$", np.nan, regex=True)

# Drop timestamp column if exists
if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.drop(columns=["time"])

# -----------------------------
# 3️⃣ Encode Categorical Columns
# -----------------------------
categorical_columns = [
    "erc20_most_sent_token_type",
    "erc20_most_rec_token_type",
]

encoders = {}

for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].fillna("missing").astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"✅ Encoded column: {col}")
    else:
        print(f"⚠️ Column {col} not found. Skipping.")

# -----------------------------
# 4️⃣ Convert Remaining Columns to Numeric
# -----------------------------
excluded_cols = {"id", "flag"} | set(encoders.keys())

for col in df.columns:
    if col not in excluded_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill numeric NaN with median
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# -----------------------------
# 5️⃣ Feature / Target Selection
# -----------------------------
if "flag" not in df.columns:
    raise ValueError("❌ Target column 'flag' not found in dataset.")

y = df["flag"].astype(int)
X = df.drop(columns=["id", "flag"], errors="ignore")

# Ensure no object columns remain
obj_cols = X.select_dtypes(include=["object"]).columns
if len(obj_cols) > 0:
    raise ValueError(f"❌ Object columns remain after preprocessing: {list(obj_cols)}")

if y.nunique() < 2:
    raise ValueError("❌ Only one class present in target. Cannot train classifier.")

# -----------------------------
# 6️⃣ Handle Class Imbalance
# -----------------------------
class_counts = y.value_counts()
neg, pos = class_counts.iloc[0], class_counts.iloc[1]
scale_pos_weight = neg / pos if pos > 0 else 1

# -----------------------------
# 7️⃣ Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

# -----------------------------
# 8️⃣ Train Model
# -----------------------------
print("🧠 Training model...")

model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss",
)

model.fit(X_train, y_train)

# -----------------------------
# 9️⃣ Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

accuracy = model.score(X_test, y_test) * 100
print(f"\n✅ Accuracy: {accuracy:.2f}%")

# -----------------------------
# 🔟 Save Model
# -----------------------------
save_path = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(save_path, exist_ok=True)

model_package = {
    "model": model,
    "encoders": encoders,
}

joblib.dump(model_package, os.path.join(save_path, "models.joblib"))

print("💾 Model saved successfully.")
print("🎉 Retraining completed successfully.")