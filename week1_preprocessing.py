import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# 1. Load the Data
input_file = 'data/raw_sensor_data.csv'
output_file = 'data/processed_sensor_data.csv'

if not os.path.exists(input_file):
    print(f"Error: The file {input_file} was not found.")
    exit()

print("Loading data...")
# We use encoding='utf-8-sig' to handle potential hidden characters at the start of the file
df = pd.read_csv(input_file, encoding='utf-8-sig')

# --- DEBUGGING STEP ---
# This cleans up the column names by removing hidden spaces
df.columns = df.columns.str.strip()
print(f"Columns found: {df.columns.tolist()}") 
# ----------------------

# 2. Convert Timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 3. Feature Engineering (Smoothing)
sensor_cols = ['vibration', 'temperature', 'pressure']

print("Calculating rolling averages...")
# Double check machine_id exists before proceeding
if 'machine_id' not in df.columns:
    print("\nCRITICAL ERROR: 'machine_id' column is missing!")
    print("Please check your CSV file header.")
    exit()

for col in sensor_cols:
    df[f'{col}_rolling_mean_3h'] = df.groupby('machine_id')[col].transform(
        lambda x: x.rolling(window=3).mean()
    )

# Drop rows with NaN values (the first few hours)
df.dropna(inplace=True)

# 4. Scaling
scaler = StandardScaler()
cols_to_scale = [col for col in df.columns if 'vibration' in col or 'temperature' in col or 'pressure' in col]

print("Scaling features...")
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# 5. Save the Processed Data
print(f"Saving processed data to {output_file}...")
df.to_csv(output_file, index=False)
print("Processing complete!")