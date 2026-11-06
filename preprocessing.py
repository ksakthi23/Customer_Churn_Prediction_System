
import pandas as pd

# Step 1: Load dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

print("Original dataset:", df.shape)

# Step 2: Remove duplicate records
df = df.drop_duplicates()

# Step 3: Convert TotalCharges into numeric format
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Step 4: Check missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Step 5: Handle missing TotalCharges
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Step 6: Remove unnecessary customer ID
df = df.drop(columns=["customerID"])

# Step 7: Convert target variable
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# Step 8: Check cleaned dataset
print("\nCleaned dataset:", df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nChurn values:")
print(df["Churn"].value_counts())

# Step 9: Save cleaned dataset
df.to_csv(
    "data/cleaned_customer_churn.csv",
    index=False
)

print("\nDATA PREPROCESSING COMPLETED!")
print("Cleaned dataset saved successfully.")
