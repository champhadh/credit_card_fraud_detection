import pandas as pd

# Load dataset
df = pd.read_csv("data/creditcard.csv")  
print("✅ Data successfully loaded!")

# Print dataset overview
print("📊 Dataset Overview:")
print(df.info())
print(df.head())

# Normalize 'Amount' column
df["Amount"] = (df["Amount"] - df["Amount"].min()) / (df["Amount"].max() - df["Amount"].min())
print("🔄 Amount column normalized.")

# Save cleaned dataset
df.to_csv("data/cleaned_creditcard.csv", index=False)
print("💾 Cleaned data saved at: data/cleaned_creditcard.csv")
