# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 21:18:02 2025

@author: sanja
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

def evaluate_model_performance():
    """
    Loads the trained model and evaluates its performance on the test set.
    """
    print("Starting Model Evaluation ")
    
    # 1. Load Data, Model, and Scalers 
    model_ready_data = pd.read_excel('gold_rate_features.xlsx', index_col=0, parse_dates=True)
    model = load_model('gold_price_predictor.h5')
    x_scaler = joblib.load('x_scaler.pkl')
    y_scaler = joblib.load('y_scaler.pkl')

    # 2. Recreate Test Set
    target_column = 'Gold_Price_Poun_INR'
    features_to_use = model_ready_data.drop(columns=[target_column, 'Gold_Price_USD', 'Oil_Price_USD', 'USD_INR'])
    
    split_index = int(len(features_to_use) * 0.8)
    X_test = features_to_use[split_index:]
    y_test = model_ready_data[[target_column]][split_index:]

    # 3. Scale Data and Make Predictions
    X_test_scaled = x_scaler.transform(X_test)
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    # 4. Visualize Results
    print("-> Generating plot...")
    plt.figure(figsize=(15, 7))
    plt.plot(y_test.index, y_test.values, label="Actual Price", color="blue", linewidth=2)
    plt.plot(y_test.index, y_pred, label="Predicted Price", color="red", linestyle='--')
    plt.title("Gold Price: Actual vs. Predicted", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price (INR per Poun)", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    evaluate_model_performance()
