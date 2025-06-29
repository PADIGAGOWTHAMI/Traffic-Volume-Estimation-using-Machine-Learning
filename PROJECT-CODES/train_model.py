import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

# Load dataset
df = pd.read_csv(r"C:\Users\srihari\OneDrive\Desktop\traffic volume.csv")
df.columns = df.columns.str.strip()

# Parse date & time
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df[['hours', 'minutes', 'seconds']] = df['Time'].str.split(':', expand=True).astype(int)
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Drop null rows
df.dropna(inplace=True)

# Feature & target
features = ["holiday", "temp", "rain", "snow", "weather", "year", "month", "day", "hours", "minutes", "seconds"]
target = "traffic_volume"

X = df[features]
y = df[target]

# Encode categorical features
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(X[["holiday", "weather"]])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["holiday", "weather"]))

X = X.drop(columns=["holiday", "weather"]).reset_index(drop=True)
X = pd.concat([X, encoded_df], axis=1)

# Train model (USE THIS)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model & encoder
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(encoder, open("encoder.pkl", "wb"))

print("✅ Model and encoder saved successfully with current Python version")
