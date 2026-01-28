# -IoT-Predictive-Maintenance-Engine---Infotact-Solutions-

# FactoryGuard AI (Project 1)

## Overview
Predictive maintenance system for 500 robotic arms.
Predicts failures 24 hours in advance using rolling window sensor features.

## Setup
1. Install dependencies:
   `pip install -r requirements.txt`

2. Generate Synthetic Data:
   `python generate_data.py`

3. Run Week 1 Engineering:
   `python week1_preprocessing.py`

## Project Structure
- `data/raw_sensor_data.csv`: Generated time-series logs.
- `data/week1_engineered_data.csv`: Data with Rolling/Lag features ready for XGBoost.