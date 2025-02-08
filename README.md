# Credit Card Fraud Detection API

## Overview
This API is designed to detect fraudulent credit card transactions based on input features derived from transaction data. Users can send transaction details via a POST request, and the API will return a fraud probability score along with a prediction indicating whether the transaction is fraudulent.

## Features
- Accepts transaction data in JSON format.
- Returns a fraud probability score.
- Predicts whether a transaction is fraudulent (1) or not (0).
- Fast response time using a trained machine learning model.

## Installation & Setup
### Prerequisites
- Python 3.12+
- Virtual environment (recommended)

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/credit_card_fraud_detection.git
   cd credit_card_fraud_detection
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use 'venv\Scripts\activate'
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the API server:
   ```bash
   uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
   ```

## API Endpoints

### Health Check
- **Endpoint:** `/`
- **Method:** `GET`
- **Response:**
  ```json
  {"message": "API is running"}
  ```

### Predict Fraudulent Transaction
- **Endpoint:** `/predict`
- **Method:** `POST`
- **Headers:**
  ```json
  {"Content-Type": "application/json"}
  ```
- **Request Body:** (Example)
  ```json
  {
    "Time": 10000.0,
    "V1": -1.5, "V2": 2.3, "V3": -0.8, "V4": 0.9, "V5": -1.2, "V6": 1.8, "V7": -0.5,
    "V8": 0.4, "V9": -0.9, "V10": -2.2, "V11": 1.0, "V12": -1.1, "V13": 0.6, "V14": -0.3,
    "V15": 1.5, "V16": -0.6, "V17": 0.8, "V18": -0.5, "V19": 1.3, "V20": -0.7, "V21": 0.2,
    "V22": -1.4, "V23": 0.5, "V24": -0.8, "V25": 0.9, "V26": -0.3, "V27": 0.1, "V28": -0.4,
    "Amount": 50.0
  }
  ```
- **Response:**
  ```json
  {
    "Fraud_Probability": 7.700162313994952e-06,
    "Predicted_Fraud": 0
  }
  ```
  - `Fraud_Probability`: Probability of the transaction being fraudulent.
  - `Predicted_Fraud`: `1` if fraudulent, `0` otherwise.

## Testing the API
You can test the API using `cURL`:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"Time": 10000.0, "V1": -1.5, "V2": 2.3, "V3": -0.8, "V4": 0.9, "V5": -1.2, "V6": 1.8, "V7": -0.5, "V8": 0.4, "V9": -0.9, "V10": -2.2, "V11": 1.0, "V12": -1.1, "V13": 0.6, "V14": -0.3, "V15": 1.5, "V16": -0.6, "V17": 0.8, "V18": -0.5, "V19": 1.3, "V20": -0.7, "V21": 0.2, "V22": -1.4, "V23": 0.5, "V24": -0.8, "V25": 0.9, "V26": -0.3, "V27": 0.1, "V28": -0.4, "Amount": 50.0}'
```

Or using Python:
```python
import requests

url = "http://127.0.0.1:8000/predict"
data = {
    "Time": 10000.0,
    "V1": -1.5, "V2": 2.3, "V3": -0.8, "V4": 0.9, "V5": -1.2, "V6": 1.8, "V7": -0.5,
    "V8": 0.4, "V9": -0.9, "V10": -2.2, "V11": 1.0, "V12": -1.1, "V13": 0.6, "V14": -0.3,
    "V15": 1.5, "V16": -0.6, "V17": 0.8, "V18": -0.5, "V19": 1.3, "V20": -0.7, "V21": 0.2,
    "V22": -1.4, "V23": 0.5, "V24": -0.8, "V25": 0.9, "V26": -0.3, "V27": 0.1, "V28": -0.4,
    "Amount": 50.0
}

response = requests.post(url, json=data)
print(response.json())
```

## Deployment
To deploy the API using Docker:
1. Build the Docker image:
   ```bash
   docker build -t fraud-detection-api -f deployment/Dockerfile .
   ```
2. Run the container:
   ```bash
   docker run -p 8000:8000 fraud-detection-api
   ```
3. The API will be available at `http://127.0.0.1:8000`.

## Contributing
Contributions are welcome. Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License.

