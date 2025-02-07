import os
import pandas as pd

# Define the base directory dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")

# Load the dataset
try:
    df = pd.read_csv(DATA_PATH)
    print("✅ Data successfully loaded!")
except FileNotFoundError:
    print(f"❌ File not found: {DATA_PATH}")
    exit(1)

# Display basic info
print("📊 Dataset Overview:")
print(df.info())
print(df.head())

# Drop duplicates if any
df.drop_duplicates(inplace=True)

# Handle missing values (if any)
df.fillna(0, inplace=True)

# Normalize the Amount column
if 'Amount' in df.columns:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    print("🔄 Amount column normalized.")

# Save the cleaned dataset
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_creditcard.csv")
df.to_csv(CLEANED_DATA_PATH, index=False)
print(f"💾 Cleaned data saved at: {CLEANED_DATA_PATH}")
