import os
import re
import warnings
import numpy as np
import pandas as pd
import joblib

from sqlalchemy.orm import Session
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
query = session.query(Transactions)
df = pd.read_sql(query.statement, session.bind)
session.close()

if df.empty:
    raise ValueError("❌ Database returned empty dataset. Cannot retrain model.")

print(f"✅ Loaded dataset with shape: {df.shape}")

# -----------------------------
# 2️⃣ Clean & Normalize Data
# -----------------------------
# Normalize column names (strip accidental spaces)
df.columns = df.columns.str.strip()
# Replace empty strings with NaN
df = df.replace(r'^\s*$', np.nan, regex=True)

# If there's a time column, parse and drop it (we won't use timestamps as raw features)
if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.drop(columns=["time"])

# Utility to find a column from several candidate names
def find_column(candidates, columns):
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    # try normalized match (remove non-alnum)
    norm_map = {re.sub(r'[^0-9a-z]', '', c.lower()): c for c in columns}
    for cand in candidates:
        n = re.sub(r'[^0-9a-z]', '', cand.lower())
        if n in norm_map:
            return norm_map[n]
    return None

# Candidate names for the two categorical columns (handle different naming variants)
sent_candidates = [
    "erc20_most_sent_token_type",
    "erc20 most sent token type",
    "ERC20 most sent token type",
    "ERC20_most_sent_token_type",
]
rec_candidates = [
    "erc20_most_rec_token_type",
    "erc20_most_rec_token_type",
    "ERC20_most_rec_token_type",
    "ERC20 most rec token type",
]

sent_col = find_column(sent_candidates, df.columns)
rec_col = find_column(rec_candidates, df.columns)

# Coerce all remaining non-categorical, non-id/flag columns to numeric
excluded = {"id", "flag"}
if sent_col:
    excluded.add(sent_col)
if rec_col:
    excluded.add(rec_col)

for col in df.columns:
    if col in excluded:
        continue
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill numeric NaNs with median
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if numeric_cols:
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# -----------------------------
# 3️⃣ Encode categorical columns
# -----------------------------
enc1 = None
enc2 = None

if sent_col:
    df[sent_col] = df[sent_col].fillna("missing").astype(str)
    enc1 = LabelEncoder()
    df[sent_col] = enc1.fit_transform(df[sent_col])
    print(f"✅ Encoded column: {sent_col}")
else:
    print("⚠️ Warning: sent-token column not found. Creating fallback encoder.")
    enc1 = LabelEncoder()
    enc1.fit(["missing"])  # fallback
    df["erc20_most_sent_token_type"] = 0
    sent_col = "erc20_most_sent_token_type"

if rec_col:
    df[rec_col] = df[rec_col].fillna("missing").astype(str)
    enc2 = LabelEncoder()
    df[rec_col] = enc2.fit_transform(df[rec_col])
    print(f"✅ Encoded column: {rec_col}")
else:
    print("⚠️ Warning: rec-token column not found. Creating fallback encoder.")
    enc2 = LabelEncoder()
    enc2.fit(["missing"])  # fallback
    df["erc20_most_rec_token_type"] = 0
    rec_col = "erc20_most_rec_token_type"

# Ensure encoders are present
encoders = {"enc1": enc1, "enc2": enc2}

# -----------------------------
# 4️⃣ Feature / Target Selection
# -----------------------------
if "flag" not in df.columns:
    raise ValueError("❌ Target column 'flag' not found in dataset.")

y = df["flag"].astype(float)
X = df.drop(columns=["id", "flag"], errors="ignore")

# Ensure there are no object dtype columns left (XGBoost requires numeric or category with enable_categorical)
obj_cols = X.select_dtypes(include=[object]).columns.tolist()
if obj_cols:
    # convert remaining object columns to category codes
    for c in obj_cols:
        X[c] = X[c].astype(str).fillna("missing")
        X[c] = LabelEncoder().fit_transform(X[c])
    print(f"Converted object columns to numeric codes: {obj_cols}")

if len(y.value_counts()) < 2:
    raise ValueError("❌ Only one class present in target. Cannot train classifier.")

# -----------------------------
# 5️⃣ Handle Class Imbalance
# -----------------------------
class_counts = y.value_counts()
if len(class_counts) < 2 or class_counts.iloc[1] == 0:
    scale_pos_weight = 1
else:
    # weight for positive class
    scale_pos_weight = class_counts.iloc[0] / class_counts.iloc[1]

# -----------------------------
# 6️⃣ Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    shuffle=True,
    random_state=42,
    stratify=y
)

# -----------------------------
# 7️⃣ Train Model
# -----------------------------
model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

print("🧠 Training model...")
model.fit(X_train, y_train)

# -----------------------------
# 8️⃣ Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

accuracy = model.score(X_test, y_test) * 100
print(f"\n✅ Accuracy: {accuracy:.2f}%")

# -----------------------------
# 9️⃣ Save Model
# -----------------------------
save_path = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(save_path, exist_ok=True)

model_package = {
    "model": model,
    "enc1": enc1,
    "enc2": enc2,
}

joblib.dump(model_package, os.path.join(save_path, "models.joblib"))

print("💾 Model saved successfully to model/models.joblib")
print("🎉 Retraining completed successfully.")
