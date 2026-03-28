import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import shap
from flask import Flask, request, jsonify

# -------------------------------
# 1️⃣ LOAD MODEL
# -------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "models", "predictive_maintenance_model.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found. Run Week 2 training first.")

model = joblib.load(model_path)

# Initialize SHAP explainer once (IMPORTANT for performance)
explainer = shap.TreeExplainer(model)

# -------------------------------
# 2️⃣ INIT FLASK APP
# -------------------------------
app = Flask(__name__)

# -------------------------------
# 3️⃣ PREDICT ENDPOINT
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    start_time = time.time()

    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON payload received"}), 400

        # Convert JSON → DataFrame
        input_df = pd.DataFrame([data])

        # Ensure correct column order
        model_features = model.get_booster().feature_names
        input_df = input_df[model_features]

        # -----------------------
        # Prediction
        # -----------------------
        prob = model.predict_proba(input_df)[0][1]
        prediction = int(prob > 0.5)

        # -----------------------
        # SHAP Explanation
        # -----------------------
        shap_values = explainer.shap_values(input_df)

        shap_dict = dict(
            zip(model_features, shap_values[0])
        )

        # Get top 3 risk drivers
        top_features = sorted(
            shap_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]

        explanation = [
            {
                "feature": f,
                "impact": float(v),
                "direction": "increases risk" if v > 0 else "reduces risk"
            }
            for f, v in top_features
        ]

        latency_ms = (time.time() - start_time) * 1000

        response = {
            "failure_probability": round(float(prob), 4),
            "prediction": prediction,
            "risk_level": "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW",
            "top_risk_factors": explanation,
            "latency_ms": round(latency_ms, 2)
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------
@app.route("/")
def home():
    return "FactoryGuard AI is running successfully!"

# 4️⃣ RUN SERVER
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
