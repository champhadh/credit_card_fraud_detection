from fastapi import FastAPI
import numpy as np
import joblib
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load the trained model and scaler
model = joblib.load("models/fraud_detection_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Define input schema
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.get("/")
def home():
    return {"message": "Credit Card Fraud Detection API is running!"}

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        # Convert input data to numpy array
        input_data = np.array([[transaction.Time, transaction.V1, transaction.V2, transaction.V3,
                                transaction.V4, transaction.V5, transaction.V6, transaction.V7, transaction.V8, transaction.V9,
                                transaction.V10, transaction.V11, transaction.V12, transaction.V13, transaction.V14, transaction.V15,
                                transaction.V16, transaction.V17, transaction.V18, transaction.V19, transaction.V20, transaction.V21,
                                transaction.V22, transaction.V23, transaction.V24, transaction.V25, transaction.V26, transaction.V27,
                                transaction.V28, transaction.Amount]])

        # Scale input data
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict_proba(input_data_scaled)[:, 1][0]  # Probability of fraud
        predicted_label = int(prediction > 0.5)  # 1 if fraud, 0 otherwise

        # Convert numpy float32 to Python float
        return {
            "Fraud_Probability": float(prediction),
            "Predicted_Fraud": predicted_label
        }

    except Exception as e:
        return {"error": str(e)}

