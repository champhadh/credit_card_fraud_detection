import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("data/cleaned_creditcard.csv")

# Split data
X = df.drop(columns=["Class"])  # Features
y = df["Class"]  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model
model = XGBClassifier(n_estimators=200, scale_pos_weight=10, use_label_encoder=False, eval_metric="logloss")
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"✅ Model training complete! Accuracy: {accuracy:.4f}")
print(f"🎯 Precision: {precision:.4f}")
print(f"🔄 Recall: {recall:.4f}")
print(f"⚖️ F1 Score: {f1:.4f}")

# Save model and scaler
joblib.dump(model, "models/fraud_detection_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("💾 Model and Scaler saved successfully!")
