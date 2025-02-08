import pandas as pd
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load("models/fraud_detection_model.pkl")
scaler = joblib.load("models/scaler.pkl")
print("✅ Model and Scaler loaded successfully!")

# Load cleaned dataset
df = pd.read_csv("data/cleaned_creditcard.csv")

# Sample transactions for prediction
X_sample = df.sample(n=5, random_state=42).drop(columns=["Class"])
X_sample_scaled = scaler.transform(X_sample)  # Apply scaling

print("\n🔎 Sample transactions for prediction:")
print(X_sample.head())

# Predict fraud probability
y_prob = model.predict_proba(X_sample_scaled)[:, 1]  # Get probability scores
y_pred = (y_prob > 0.3).astype(int)  # Lower threshold to detect more fraud cases

# Display prediction results
result_df = X_sample.copy()
result_df["Fraud_Probability"] = y_prob
result_df["Predicted_Fraud"] = y_pred

print("\n🛑 Fraud Prediction Results:")
print(result_df[["Amount", "Fraud_Probability", "Predicted_Fraud"]])
