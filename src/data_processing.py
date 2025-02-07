import pandas as pd
import os

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

# Load dataset
df = pd.read_csv("data/creditcard.csv")  # Use correct path

print("✅ Data successfully loaded!")

# Print dataset overview
print("📊 Dataset Overview:")
print(df.info())
print(df.head())

# Normalize 'Amount' column
df["Amount"] = (df["Amount"] - df["Amount"].min()) / (df["Amount"].max() - df["Amount"].min())

print("🔄 Amount column normalized.")

# Save cleaned dataset **with column names** and without index
df.to_csv("data/cleaned_creditcard.csv", index=False)

print("💾 Cleaned data saved at: data/cleaned_creditcard.csv")
