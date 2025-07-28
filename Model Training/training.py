import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Step 1: Generate or Load Dataset
data = pd.read_csv("earthing_data.csv")  # or create synthetic data
X = data.drop("Maintenance Required", axis=1)
y = data["Maintenance Required"]

# Step 2: Scale Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split and Train
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 4: Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Step 5: Save model and scaler
joblib.dump(model, "earthing_maintenance_model.pkl")
joblib.dump(scaler, "scaler.pkl")
