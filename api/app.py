import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_creditcard.csv")

# Load trained models
logistic_model = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl"))
rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))

@app.route("/")
def home():
    return "✅ Credit Card Fraud Detection API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON input
        data = request.get_json()

        # Validate request format
        if not isinstance(data, list):
            return jsonify({"error": "Invalid input format. Expected a list of dictionaries."}), 400

        # Convert JSON to DataFrame
        df = pd.DataFrame(data)

        # Ensure all required features are present
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400

        # Reorder columns to match training data
        df = df[feature_columns]

        # Handle missing values by filling with 0
        df.fillna(0, inplace=True)

        # Make predictions
        logistic_pred = logistic_model.predict(df)
        rf_pred = rf_model.predict(df)

        # Return predictions
        return jsonify({
            "logistic_prediction": logistic_pred.tolist(),
            "rf_prediction": rf_pred.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5001)
