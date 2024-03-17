import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Save the trained model to a file

# Load the healthcare dataset
df = pd.read_csv("healthcare_dataset.csv")

# Data Preprocessing
# Drop irrelevant columns and handle missing values if necessary
df.drop(columns=["Name", "Date of Admission", "Doctor", "Hospital", "Insurance Provider", "Room Number",
                 "Admission Type", "Discharge Date", "Medication", "Test Results"], inplace=True)
df.dropna(inplace=True)  # Drop rows with missing values

# Encode categorical variables
df = pd.get_dummies(df, columns=["Gender", "Blood Group Type"])

# Feature Selection
# Based on the correlation analysis, select relevant features
selected_features = ["Age", "Gender_Female", "Gender_Male", "Blood Group Type_A+", "Blood Group Type_A-",
                     "Blood Group Type_AB+", "Blood Group Type_AB-", "Blood Group Type_B+", "Blood Group Type_B-",
                     "Blood Group Type_O+", "Blood Group Type_O-"]
X = df[selected_features]
y = df["Medical Condition"]

# Model Training
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Model Evaluation
# Predict on the testing data
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
joblib.dump(clf, "random_forest_model.pkl")