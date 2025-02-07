import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load preprocessed data
df = pd.read_csv("../data/cleaned_creditcard.csv")

# Split features and labels
X = df.drop(columns=['Class'])  # Features
y = df['Class']  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions & evaluation
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model
with open("../models/random_forest.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model training complete. Model saved as 'random_forest.pkl'")
