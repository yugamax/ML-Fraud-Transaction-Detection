import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv(r"dataset\transaction_recs.csv")

x = df.iloc[:,2:20]
y= df.iloc[:,1]

y = y.replace({'Fraud': 1, 'Non - Fraud': 0})
balancingf = y.value_counts()[0]/y.value_counts()[1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42, stratify=y)

model = XGBClassifier(scale_pos_weight=balancingf)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("\t\tClassification Report:\n")
print(classification_report(y_test, y_pred))
print(f"\nAccuracy of model : {model.score(x_test, y_test)*100:.2f} %")

joblib.dump(model, r"model/xgb_model.joblib")