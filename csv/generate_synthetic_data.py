import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate synthetic features
data = pd.DataFrame({
    'Soil_Resistivity': np.random.normal(loc=60, scale=15, size=n_samples),           # ohm-m
    'Ground_Resistance': np.random.normal(loc=3.5, scale=0.8, size=n_samples),         # ohms
    'Ambient_Temperature': np.random.normal(loc=30, scale=5, size=n_samples),          # °C
    'Soil_Moisture': np.random.uniform(5, 40, size=n_samples),                         # %
    'Leakage_Current': np.random.normal(loc=1.2, scale=0.5, size=n_samples),           # mA
    'Fault_Current_Events': np.random.poisson(lam=2, size=n_samples),                  # events/week
    'Corrosion_Level': np.random.uniform(0, 1, size=n_samples),                        # scale 0–1
})

# Synthetic target variable based on some logic + randomness
# Higher resistance, higher corrosion, and more faults → more likely maintenance
prob_failure = (
    0.1 * data['Soil_Resistivity'] +
    0.3 * data['Ground_Resistance'] +
    0.2 * data['Corrosion_Level'] * 10 +
    0.3 * data['Fault_Current_Events']
)

# Normalize and convert to binary labels
threshold = np.percentile(prob_failure, 75)
data['Maintenance Required'] = (prob_failure > threshold).astype(int)

# Save to CSV
data.to_csv('synthetic_earthing_data.csv', index=False)

print("✅ Synthetic dataset generated and saved as 'synthetic_earthing_data.csv'.")
print(data.head())
