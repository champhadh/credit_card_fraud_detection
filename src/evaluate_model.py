import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load trained model
model = joblib.load("models/fraud_detection_model.pkl")

# Load cleaned dataset
df = pd.read_csv("data/cleaned_creditcard.csv")

# Split into features (X) and target variable (y)
X = df.drop(columns=["Class"])
y = df["Class"]

# Predict fraud
y_pred = model.predict(X)

# Compute evaluation metrics
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, zero_division=1)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

# Print results
print("\n📊 **Model Evaluation Metrics:**")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"🎯 Precision: {precision:.4f}")
print(f"🔄 Recall: {recall:.4f}")
print(f"⚖️ F1 Score: {f1:.4f}")
