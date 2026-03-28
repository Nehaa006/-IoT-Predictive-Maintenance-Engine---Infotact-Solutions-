import pandas as pd
import numpy as np

def generate_dataset():
    print("Initializing generation for 500 Robotic Arms...")

    # --- 1. CONFIGURATION ---
    NUM_MACHINES = 500
    DAYS = 30
    HOURS_PER_DAY = 24
    
    # --- 2. CREATE BASE STRUCTURE (Fast Method) ---
    # Create the time range
    dates = pd.date_range(start='2025-12-01', periods=DAYS * HOURS_PER_DAY, freq='h')
    machine_ids = [f'RBT-{i:03d}' for i in range(1, NUM_MACHINES + 1)]
    
    print("Building DataFrame structure...")
    
    # Create a MultiIndex (This is much faster and safer than appending)
    index = pd.MultiIndex.from_product([machine_ids, dates], names=['machine_id', 'timestamp'])
    df = pd.DataFrame(index=index).reset_index()
    
    # --- 3. SIMULATE NORMAL SENSOR READINGS ---
    print("Simulating sensor data...")
    n = len(df)
    np.random.seed(42)
    
    # Generate random sensor values
    df['vibration'] = np.random.normal(loc=35, scale=5, size=n)   # Mean 35, Std 5
    df['temperature'] = np.random.normal(loc=70, scale=5, size=n) # Mean 70, Std 5
    df['pressure'] = np.random.normal(loc=12, scale=2, size=n)    # Mean 12, Std 2
    df['failure'] = 0 # Default to Healthy

    # --- 4. INJECT "PRE-FAILURE" PATTERNS ---
    # We select 0.5% of rows to be failures
    n_failures = int(n * 0.005)
    
    # Get random indices for failures
    fail_indices = np.random.choice(df.index, size=n_failures, replace=False)
    
    print(f"Injecting {n_failures} failure patterns (24h lead time)...")
    
    # Mark failures
    df.loc[fail_indices, 'failure'] = 1
    
    # Loop through failures to add the "Warning Signs"
    # (Rising temperature and vibration in the 24 hours BEFORE failure)
    for idx in fail_indices:
        fail_machine = df.at[idx, 'machine_id']
        fail_time = df.at[idx, 'timestamp']
        
        # Calculate start of the 24h window
        start_time = fail_time - pd.Timedelta(hours=24)
        
        # Find rows for THIS machine in THAT time window
        mask = (df['machine_id'] == fail_machine) & \
               (df['timestamp'] >= start_time) & \
               (df['timestamp'] < fail_time)
        
        # Add "Drift" (Increase values)
        # Check if mask has any True values to avoid errors
        if mask.any():
            df.loc[mask, 'temperature'] += np.random.uniform(10, 25, size=mask.sum())
            df.loc[mask, 'vibration'] += np.random.uniform(10, 20, size=mask.sum())

    # --- 5. SAVE ---
    filename = 'sensor_data.csv'
    df.to_csv(filename, index=False)
    print(f"\nSUCCESS: Dataset generated: '{filename}'")
    print(f"Total Rows: {len(df)}")

if __name__ == "__main__":
    generate_dataset()