import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load cleaned dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_creditcard.csv")
df = pd.read_csv(CLEANED_DATA_PATH)

# Separate features and target
X = df.drop(columns=["Class"])  # Features
y = df["Class"]  # Target

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle Imbalance using SMOTE (Synthetic Minority Over-sampling)
smote = SMOTE(sampling_strategy=0.2, random_state=42)  # Adjust ratio as needed
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Save processed data
FEATURED_DATA_PATH = os.path.join(BASE_DIR, "data", "featured_creditcard.csv")
pd.DataFrame(X_resampled).to_csv(FEATURED_DATA_PATH, index=False)
print(f"✅ Feature-engineered data saved at: {FEATURED_DATA_PATH}")
