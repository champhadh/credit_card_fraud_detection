import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load cleaned dataset
df = pd.read_csv("data/cleaned_creditcard.csv")

print(f"📝 Columns in dataset: {list(df.columns)}")

# Ensure "Class" column exists
if "Class" not in df.columns:
    print("❌ Error: 'Class' column not found in dataset. Please check the dataset.")
    exit()

# Reduce dataset size for faster training
df_sample = df.sample(frac=0.1, random_state=42)  # Using 10% of data
X = df_sample.drop(columns=["Class"])
y = df_sample["Class"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=10000, solver="saga")  # Increased iterations and changed solver
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model training complete! Accuracy: {accuracy:.4f}")

# Save the trained model
joblib.dump(model, "models/fraud_detection_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")  # Save scaler for future use

print("💾 Model saved to: models/fraud_detection_model.pkl")
