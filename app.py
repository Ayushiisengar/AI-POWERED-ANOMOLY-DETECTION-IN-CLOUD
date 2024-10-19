from flask import Flask, render_template, jsonify, request, redirect, url_for, session
import random
import numpy as np
from sklearn.ensemble import IsolationForest

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a strong secret key

# Simulated metrics data for each PC
metrics_data = {
    'PC1': {'cpu': [], 'memory': [], 'disk': [], 'network': []},
    'PC2': {'cpu': [], 'memory': [], 'disk': [], 'network': []},
    'PC3': {'cpu': [], 'memory': [], 'disk': [], 'network': []}
}

# Anomalies detected
anomalies_data = {
    'PC1': [], 'PC2': [], 'PC3': []
}

# Isolation Forest Model for anomaly detection
model = IsolationForest(contamination=0.1, random_state=42)

# Helper function to simulate real-time data generation
def generate_fake_metrics():
    for pc_id in metrics_data.keys():
        metrics_data[pc_id]['cpu'].append(random.randint(0, 100))
        metrics_data[pc_id]['memory'].append(random.randint(0, 100))
        metrics_data[pc_id]['disk'].append(random.randint(0, 100))
        metrics_data[pc_id]['network'].append(random.randint(0, 100))

        # Keep only the last 20 entries for model training
        for key in metrics_data[pc_id]:
            if len(metrics_data[pc_id][key]) > 20:
                metrics_data[pc_id][key].pop(0)

# Train the Isolation Forest model on the most recent data
def train_isolation_forest():
    for pc_id in metrics_data.keys():
        if len(metrics_data[pc_id]['cpu']) >= 20:  # Only train if we have enough data points
            # Prepare the dataset with CPU, memory, disk, and network
            data = np.array([metrics_data[pc_id]['cpu'][-20:], 
                             metrics_data[pc_id]['memory'][-20:], 
                             metrics_data[pc_id]['disk'][-20:], 
                             metrics_data[pc_id]['network'][-20:]]).T
            
            # Fit the model on the data
            model.fit(data)

# Detect anomalies using Isolation Forest
def detect_anomalies():
    for pc_id in metrics_data.keys():
        if len(metrics_data[pc_id]['cpu']) >= 20:  # Only check for anomalies if we have enough data points
            # Prepare the current metric data for the last entry
            current_data = np.array([metrics_data[pc_id]['cpu'][-1], 
                                     metrics_data[pc_id]['memory'][-1], 
                                     metrics_data[pc_id]['disk'][-1], 
                                     metrics_data[pc_id]['network'][-1]]).reshape(1, -1)
            
            # Predict if the current data point is an anomaly (-1 indicates an anomaly)
            prediction = model.predict(current_data)
            
            if prediction == -1:  # If it's an anomaly
                anomalies_data[pc_id].append(f"Anomaly detected in PC {pc_id}: {current_data}")
            
            # Keep only the last 5 anomalies
            if len(anomalies_data[pc_id]) > 5:
                anomalies_data[pc_id].pop(0)

@app.route('/', methods=['GET'])
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == 'password':
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', message='Invalid username or password')
    return render_template('login.html', message='')

@app.route('/dashboard')
def dashboard():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/api/metrics/<pc_id>')
def get_metrics(pc_id):
    generate_fake_metrics()  # Simulate metrics generation
    train_isolation_forest()  # Train the Isolation Forest
    return jsonify(metrics_data[pc_id])

@app.route('/api/anomalies')
def get_anomalies():
    detect_anomalies()  # Detect anomalies
    return jsonify(anomalies_data)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
