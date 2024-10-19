from sklearn.ensemble import IsolationForest
import numpy as np
import joblib

# Sample data (CPU, memory, disk usage)
# Replace this with actual data for training, or load historical data from your metrics
# Ensure the data is a 2D array where each row is a data point
X = np.array([[30, 60, 70], 
              [32, 62, 65], 
              [45, 70, 80], 
              [48, 75, 90], 
              [50, 80, 95]])

# Train Isolation Forest model
model = IsolationForest(contamination=0.05)  # Adjust contamination as needed
model.fit(X)

# Save the trained model to a file
joblib.dump(model, 'isolation_forest_model.pkl')
print("Model trained and saved as 'isolation_forest_model.pkl'")
