import pandas as pd
import numpy as np

np.random.seed(42)

# Number of samples
n_samples = 50

# Simulate features
data = {
    "Temp_air": np.random.uniform(25, 35, n_samples),            # °C
    "Humidity": np.random.uniform(60, 95, n_samples),            # %
    "Soil_moisture": np.random.uniform(40, 100, n_samples),      # %
    "Soil_pH": np.random.uniform(5.5, 7.5, n_samples),           # pH
    "N": np.random.uniform(50, 150, n_samples),                  # ppm
    "P": np.random.uniform(20, 80, n_samples),                   # ppm
    "K": np.random.uniform(40, 200, n_samples),                  # ppm
    "Light": np.random.uniform(200, 1000, n_samples),            # lux
    "Rainfall": np.random.uniform(0, 20, n_samples),             # mm
    "Wind": np.random.uniform(0, 5, n_samples),                  # m/s
    "CO2": np.random.uniform(350, 450, n_samples),               # ppm
    "Leaf_wet": np.random.randint(0, 2, n_samples),              # 0 = dry, 1 = wet
    "Canopy_temp": np.random.uniform(25, 38, n_samples),         # °C
    "NDVI": np.random.uniform(0.3, 0.9, n_samples),              # index 0–1
    "VOCs": np.random.uniform(0, 5, n_samples),                  # ppm
}

# Simulate Disease Classes based on a combination of features (simplified logic)
disease_classes = []
for i in range(n_samples):
    if data["Leaf_wet"][i] == 1 and data["Humidity"][i] > 80 and data["Temp_air"][i] > 30:
        disease_classes.append("Bacterial_Spot")
    elif data["NDVI"][i] < 0.5 and data["Canopy_temp"][i] > 32:
        disease_classes.append("Early_Blight")
    else:
        disease_classes.append("Healthy")

data["Disease_Class"] = disease_classes

# Create DataFrame
df_mock = pd.DataFrame(data)

# Save to CSV
df_mock.to_csv("mock_bellpepper_sensor_data.csv", index=False)

# Display first few rows
print(df_mock.head())
