# train_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Generate synthetic training data
np.random.seed(42)
data = pd.DataFrame({
    'Soil_Resistivity': np.random.normal(60, 15, 500),
    'Ground_Resistance': np.random.uniform(1.0, 8.0, 500),
    'Ambient_Temperature': np.random.uniform(10.0, 50.0, 500),
    'Soil_Moisture': np.random.uniform(0.0, 50.0, 500),
    'Leakage_Current': np.random.uniform(0.0, 5.0, 500),
    'Fault_Current_Events': np.random.randint(0, 11, 500),
    'Corrosion_Level': np.random.uniform(0.0, 1.0, 500),
})

# Create a target label: 1 if bad conditions, else 0
data['Maintenance_Required'] = np.where(
    (data['Soil_Resistivity'] > 80) | 
    (data['Ground_Resistance'] > 6) | 
    (data['Leakage_Current'] > 3) | 
    (data['Corrosion_Level'] > 0.7), 1, 0
)

# Split features and target
X = data.drop('Maintenance_Required', axis=1)
y = data['Maintenance_Required']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier()
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, 'earthing_maintenance_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Model and scaler saved successfully!")