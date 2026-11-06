
import pandas as pd

# Read the customer dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

# Display basic information
print("Dataset loaded successfully!")

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 customers:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nCustomer churn distribution:")
print(df["Churn"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())
