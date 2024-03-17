import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from CSV
df = pd.read_csv("healthcare_dataset.csv")

# Plot correlation between Blood Group Type, Gender, and Medical Condition
plt.figure(figsize=(10, 6))
sns.heatmap(pd.crosstab([df["Blood Group Type"], df["Gender"]], df["Medical Condition"]),
            annot=True, fmt="d", cmap="YlGnBu")
plt.title("Correlation between Blood Group Type, Gender, and Medical Condition")
plt.xlabel("Medical Condition")
plt.ylabel("Blood Group Type, Gender")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

# Calculate average billing amount per medical condition
avg_billing = df.groupby("Medical Condition")["Billing Amount"].mean().reset_index()

# Plot average billing amount per medical condition
plt.figure(figsize=(10, 6))
sns.barplot(x="Medical Condition", y="Billing Amount", data=avg_billing)
plt.title("Average Billing Amount per Medical Condition")
plt.xlabel("Medical Condition")
plt.ylabel("Average Billing Amount")
plt.xticks(rotation=45)
plt.show()