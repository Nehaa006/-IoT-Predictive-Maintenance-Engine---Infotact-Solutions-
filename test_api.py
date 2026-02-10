import requests
import json

# URL of your Flask API
url = 'http://127.0.0.1:5000/predict'

# Sample data (Simulating a machine about to fail)
# We send data in the exact order the model expects
data = {
    "vibration": 2.2,
    "temperature": 1.5,
    "pressure": 0.4,
    "vibration_rolling_mean_3h": 2.5,
    "temperature_rolling_mean_3h": 1.8,
    "pressure_rolling_mean_3h": 0.5
}

print(f"Sending data to {url}...")
print("-" * 30)

try:
    response = requests.post(url, json=data)
    
    # Check if the request was successful
    if response.status_code == 200:
        print("✅ SUCCESS! API Response:")
        print(json.dumps(response.json(), indent=4))
    else:
        print(f"❌ ERROR: Server returned {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"❌ CONNECTION ERROR: {e}")
    print("Make sure week4_api.py is running in a separate terminal!")