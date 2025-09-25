import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

def forecast_future_prices(days_to_predict=30):
    """
    Loads the trained model and forecasts future prices for a specified number of days.
    """
    print(" Starting Future Price Forecast ")

    #1. Load Model, Scalers, and Full Dataset
    print("-> Loading artifacts...")
    try:
        model = load_model('gold_price_predictor.h5')
        x_scaler = joblib.load('x_scaler.pkl')
        y_scaler = joblib.load('y_scaler.pkl')
        full_data = pd.read_excel('gold_rate_features.xlsx', index_col=0, parse_dates=True)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please ensure 'gold_price_predictor.h5', 'x_scaler.pkl', 'y_scaler.pkl', and 'gold_rate_features.xlsx' are in the same folder.")
        return None, None
    print(" Artifacts loaded.")

    #2. Get the Last Known Features for the Starting Point
    # These are the feature names the model was trained on
    feature_columns = model.input_names if hasattr(model, 'input_names') else x_scaler.get_feature_names_out()
    last_known_features = full_data[feature_columns].iloc[[-1]]

    #3. Iterative Forecasting Loop
    print(f"-> Forecasting prices for the next {days_to_predict} days...")
    future_predictions = []

    for _ in range(days_to_predict):
        # Scale the current set of features
        input_scaled = x_scaler.transform(last_known_features)

        # Predict the next day's price (scaled)
        next_day_pred_scaled = model.predict(input_scaled, verbose=0)

        # Un-scale the prediction to get the real price
        next_day_pred = y_scaler.inverse_transform(next_day_pred_scaled)[0][0]
        future_predictions.append(next_day_pred)

        #4. Prepare Features for the NEXT Prediction
        # Create a new row for the next day by copying the last one
        new_features_row = last_known_features.iloc[[-1]].copy()

        # Update the lag features using the new prediction
        # This is the most crucial step in multi-step forecasting
        new_features_row['Lag_60'] = new_features_row['Lag_30']
        new_features_row['Lag_30'] = new_features_row['Lag_7']
        new_features_row['Lag_7'] = new_features_row['Lag_3']
        new_features_row['Lag_3'] = new_features_row['Lag_1']
        new_features_row['Lag_1'] = next_day_pred

        # For other features (MAs, RSI, Oil, etc.), we'll assume they stay constant
        # as predicting them would require their own models. This is a standard simplification.
        
        # Update the last_known_features for the next loop iteration
        last_known_features = new_features_row

    print("   ✅ Forecast complete.")
    
    # 5. Prepare the Final Forecast DataFrame
    last_date = full_data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict)
    forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_Price': future_predictions})
    forecast_df.set_index('Date', inplace=True)
    
    return full_data, forecast_df

if __name__ == "__main__":
    historical_data, forecast_data = forecast_future_prices(days_to_predict=30)
    
    if forecast_data is not None:
        print("\n--- Gold Price Forecast for the Next 30 Days ---")
        for date, row in forecast_data.iterrows():
            print(f"{date.date()}: Predicted Price = ₹{row['Predicted_Price']:.2f}")
