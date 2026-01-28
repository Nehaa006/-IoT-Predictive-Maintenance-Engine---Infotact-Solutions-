import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from xgboost import XGBClassifier
import joblib
import os

# --- CONFIGURATION (FLAT STRUCTURE) ---
# This looks for the file in the SAME folder as this script.
script_dir = os.path.dirname(os.path.abspath(__file__))
input_file = r"C:\Users\sathe\Downloads\Infotact Project 1\-IoT-Predictive-Maintenance-Engine---Infotact-Solutions-\processed_sensor_data.csv"

# It will create a 'models' folder for you automatically if it's missing
model_folder = os.path.join(script_dir, 'models')
model_file = os.path.join(model_folder, 'predictive_maintenance_model.pkl')

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

# --- 1. LOAD DATA ---
print(f"Looking for file at: {input_file}")

if not os.path.exists(input_file):
    print("\nCRITICAL ERROR: File still not found.")
    print("Please make sure 'processed_sensor_data.csv' is in the SAME folder as this script.")
    exit()

df = pd.read_csv(input_file)

# --- 2. PREPARE DATA ---
X = df.drop(['failure', 'timestamp', 'machine_id'], axis=1)
y = df['failure']

# --- 3. SPLIT DATA ---
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 4. CALCULATE IMBALANCE ---
num_neg = (y_train == 0).sum()
num_pos = (y_train == 1).sum()
scale_weight = num_neg / num_pos
print(f"Imbalance Ratio: {scale_weight:.2f}")

# --- 5. TRAIN XGBOOST ---
print("Training XGBoost Classifier...")
clf = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    scale_pos_weight=scale_weight, 
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss' 
)

clf.fit(X_train, y_train)

# --- 6. EVALUATE ---
print("\n--- Model Evaluation ---")
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"True Negatives (Normal):         {cm[0][0]}")
print(f"False Positives (False Alarm):   {cm[0][1]}")
print(f"False Negatives (Missed Fail):   {cm[1][0]}")
print(f"True Positives  (Caught Fail):   {cm[1][1]}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

roc = roc_auc_score(y_test, y_prob)
f1 = f1_score(y_test, y_pred)
print(f"ROC-AUC Score: {roc:.4f}")
print(f"F1 Score:      {f1:.4f}")

# --- 7. SAVE ---
print(f"\nSaving model to {model_file}...")
joblib.dump(clf, model_file)
print("Training Complete!")