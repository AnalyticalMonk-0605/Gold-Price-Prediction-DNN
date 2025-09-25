import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

def evaluate_final_model():
    """
    Loads the final trained model and evaluates its performance on the test set.
    """
    print("--- Starting Final Model Evaluation ---")

    #1. Load Data
    model_ready_data = pd.read_excel('gold_rate_features.xlsx', index_col=0, parse_dates=True)

    # 2. Re-create the Test Set 
    # This must be identical to how it was created in the training script.
    target_column = 'Gold_Price_Poun_INR'
    y = model_ready_data[[target_column]]
    X = model_ready_data.drop(columns=[target_column, 'Gold_Price_USD', 'Oil_Price_USD', 'USD_INR'])

    split_index = int(len(X) * 0.8)
    X_test = X[split_index:]
    y_test = y[split_index:]

    #3. Load Scalers and the FINAL Model
    x_scaler = joblib.load('x_scaler.pkl')
    y_scaler = joblib.load('y_scaler.pkl')
    
    # CRITICAL: Load the correct, final model file
    print("-> Loading the final trained model (gold_price_predictor_final.h5)...")
    model = load_model('gold_price_predictor_final.h5')
    
    # 4. Scale Data and Make Predictions
    X_test_scaled = x_scaler.transform(X_test)
    
    print("-> Making predictions on the test set...")
    predicted_prices_scaled = model.predict(X_test_scaled)
    predicted_prices = y_scaler.inverse_transform(predicted_prices_scaled)

    #5. Calculate and Display Performance
    rmse = np.sqrt(mean_squared_error(y_test, predicted_prices))
    mae = mean_absolute_error(y_test, predicted_prices)
    
    print("\n--- FINAL MODEL PERFORMANCE ---")
    print(f"Root Mean Squared Error (RMSE): ₹{rmse:.2f}")
    print(f"Mean Absolute Error (MAE):    ₹{mae:.2f}")
    print("---------------------------------")
    
    #6. Visualize the Results
    plt.figure(figsize=(15, 7))
    plt.title('Final Model: Actual vs. Predicted Prices', fontsize=16)
    plt.plot(y_test.index, y_test.values, label='Actual Price', color='blue')
    plt.plot(y_test.index, predicted_prices, label='Predicted Price', color='red', linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    evaluate_final_model()
