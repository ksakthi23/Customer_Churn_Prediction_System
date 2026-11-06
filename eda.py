
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

print("DATASET LOADED SUCCESSFULLY")

# Step 2: Understand dataset
print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()

# Step 3: Display first 5 records
print("\nFirst 5 records:")
print(df.head())

# Step 4: Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Step 5: Check duplicate records
print("\nDuplicate records:")
print(df.duplicated().sum())

# Step 6: Customer churn count
print("\nChurn distribution:")
print(df["Churn"].value_counts())

# Step 7: Churn percentage
print("\nChurn percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)

# Step 8: Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Step 9: Visualize churn distribution
sns.set_theme(style="whitegrid")

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.savefig("churn_distribution.png")
plt.show()

# Step 10: Analyze contract type
plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("churn_by_contract.png")
plt.show()

# Step 11: Analyze tenure
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="Churn", y="tenure")
plt.title("Customer Tenure vs Churn")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")
plt.savefig("tenure_vs_churn.png")
plt.show()

# Step 12: Analyze monthly charges
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges vs Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.savefig("charges_vs_churn.png")
plt.show()

print("\nEDA COMPLETED SUCCESSFULLY!")
print("Graphs saved in project folder.")
