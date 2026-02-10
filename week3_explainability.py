import pandas as pd
import xgboost as xgb
import shap
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- 1. SETUP PATHS & LOAD DATA ---
# Using the same logic as your Week 2 script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'processed_sensor_data.csv')
model_path = os.path.join(script_dir, 'models', 'predictive_maintenance_model.pkl')

if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}. Please run Week 2 script first.")
    exit()

# Load processed data
df = pd.read_csv(data_path)
X = df.drop(['failure', 'timestamp', 'machine_id'], axis=1)
y = df['failure']

# Split data exactly as in Week 2 to ensure we are explaining the 'unseen' test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load the trained XGBoost model
model = joblib.load(model_path)

print("Model and data loaded successfully. Starting SHAP analysis...")

# --- 2. INITIALIZE SHAP EXPLAINER ---
# TreeExplainer is specifically optimized for XGBoost
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# --- 3. SUMMARY PLOT (Global Interpretability) ---
# This plot shows which features are the most important across the whole dataset
print("Generating Summary Plot...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("FactoryGuard AI - Feature Importance Summary")
plt.savefig('shap_summary_plot.png', bbox_inches='tight')
plt.close()
print("Summary plot saved as 'shap_summary_plot.png'")

# --- 4. FORCE PLOT (Local Interpretability) ---
# We want to explain a specific case where the model predicted a FAILURE (y=1)
# Find the first index in the test set where a failure occurred
failure_indices = y_test[y_test == 1].index
if len(failure_indices) > 0:
    # Get the first actual failure case from the test set
    sample_idx = 0 
    
    print(f"Generating Force Plot for sample at index {sample_idx}...")
    
    # Generate the force plot
    # Note: matplotlib=True is required to save the plot as a static image
    shap.force_plot(
        explainer.expected_value, 
        shap_values[sample_idx, :], 
        X_test.iloc[sample_idx, :], 
        matplotlib=True, 
        show=False
    )
    plt.savefig('shap_force_plot_failure.png', bbox_inches='tight')
    plt.close()
    print("Force plot for a specific failure saved as 'shap_force_plot_failure.png'")
else:
    print("No failure cases found in test set to explain.")

# --- 5. LOG INTERPRETATION ---
print("\n--- XAI Analysis Complete ---")
print("Interpreting the Results:")
print("1. Summary Plot: If 'vibration_rolling_mean_3h' is on top and red, it means")
print("   high vibration is the primary driver for predicting failure.")
print("2. Force Plot: This 'pushes' the model prediction from the base value.")
print("   Features in RED increase failure risk; features in BLUE decrease it.")