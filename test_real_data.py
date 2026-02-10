import pandas as pd
import requests
import json

# --- CONFIGURATION ---
# 1. Point to your processed data file
csv_path = 'processed_sensor_data.csv' 
url = 'http://127.0.0.1:5000/predict'

# 2. Load the Data
print(f"Loading real data from {csv_path}...")
try:
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} rows of data.")
except FileNotFoundError:
    print("❌ Error: 'processed_sensor_data.csv' not found. Make sure it is in this folder.")
    exit()

# 3. Select a 'Failure' Row to Test
# We filter for rows where 'failure' == 1 to see if the API detects it.
failure_rows = df[df['failure'] == 1]

if not failure_rows.empty:
    print("✅ Found actual failure examples. Selecting one random failure...")
    # Pick one random row
    selected_row = failure_rows.sample(1)
    expected_label = "FAILURE"
else:
    print("ℹ️ No failures found in this file. Selecting a random normal row...")
    selected_row = df.sample(1)
    expected_label = "NORMAL"

# 4. Extract Only the Features the API Needs
# The API expects these exact 6 columns in this order:
api_columns = [
    'vibration', 
    'temperature', 
    'pressure',
    'vibration_rolling_mean_3h', 
    'temperature_rolling_mean_3h', 
    'pressure_rolling_mean_3h'
]

# Create the JSON payload
payload = {col: float(selected_row[col].values[0]) for col in api_columns}

# 5. Send to API
print("\n--- SENDING REAL SENSOR DATA ---")
print(json.dumps(payload, indent=4))
print(f"\nSending to {url}...")

try:
    response = requests.post(url, json=payload)
    
    print("\n--- API RESPONSE ---")
    if response.status_code == 200:
        result = response.json()
        print(json.dumps(result, indent=4))
        
        # Compare Results
        api_prediction = result.get('prediction')
        print("\n--- FINAL VERIFICATION ---")
        print(f"Expected Result (Data): {expected_label}")
        print(f"Actual Result (API):    {api_prediction}")
        
        if expected_label == api_prediction:
            print("✅ PASS: The API correctly identified the machine status!")
        else:
            print("⚠️ MISMATCH: The model prediction differs from the historical label.")
            
    else:
        print(f"❌ Server Error: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"❌ Connection Error: {e}")
    print("Ensure week4_api.py is running in a separate terminal.")