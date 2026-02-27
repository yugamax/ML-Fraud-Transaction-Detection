import os
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
# 2️⃣ Clean Data
# -----------------------------
df.columns = df.columns.str.strip()  # remove accidental spaces
df = df.replace(r'^\s*$', np.nan, regex=True)

# Fill numeric NaN with median
df = df.fillna(df.median(numeric_only=True))

# -----------------------------
# 3️⃣ Handle Categorical Columns Safely
# -----------------------------
cat_columns = [
    "ERC20 most sent token type",
    "ERC20_most_rec_token_type"
]

encoders = {}

for col in cat_columns:
    if col in df.columns:
        df[col] = df[col].fillna("missing").astype(str)
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder
        print(f"✅ Encoded column: {col}")
    else:
        print(f"⚠️ Warning: Column '{col}' not found. Skipping.")

# -----------------------------
# 4️⃣ Feature / Target Selection
# -----------------------------
if "flag" not in df.columns:
    raise ValueError("❌ Target column 'flag' not found in dataset.")

y = df["flag"]
X = df.drop(columns=["id", "flag"], errors="ignore")

if len(y.value_counts()) < 2:
    raise ValueError("❌ Only one class present in target. Cannot train classifier.")

# -----------------------------
# 5️⃣ Handle Class Imbalance
# -----------------------------
class_counts = y.value_counts()
scale_pos_weight = class_counts[0] / class_counts[1]

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
save_path = os.path.join("backend", "AIML", "model")
os.makedirs(save_path, exist_ok=True)

model_package = {
    "model": model,
    "encoders": encoders
}

joblib.dump(model_package, os.path.join(save_path, "models.joblib"))

print("💾 Model saved successfully.")
print("🎉 Retraining completed successfully.")