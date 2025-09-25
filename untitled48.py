import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def run_data_pipeline():
    """
    Fetches market data, engineers features, and returns the final dataset.
    """
    print("--- 1. Starting Data Pipeline ---")
    
    start = "2014-01-01"
    end = date.today().isoformat()
    
    tickers = {'GC=F': 'Gold_Price_USD', 'CL=F': 'Oil_Price_USD', 'INR=X': 'USD_INR'}
    data = yf.download(list(tickers.keys()), start=start, end=end, progress=False)['Close']
    data.rename(columns=tickers, inplace=True)
    
    price_per_gram = (data['Gold_Price_USD'] / 28.35) * data['USD_INR']
    data['Gold_Price_Poun_INR'] = price_per_gram * 8
    data.ffill(inplace=True)
    
    target = 'Gold_Price_Poun_INR'
    for lag in [1, 3, 7, 30, 60]:
        data[f'Lag_{lag}'] = data[target].shift(lag)
    for window in [5, 10, 15, 20]:
        data[f'MA_{window}'] = data[target].rolling(window=window).mean()
    
    delta = data[target].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    data['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    
    final_data = data.dropna()
    print("✅ Pipeline complete. Data is ready.")
    return final_data

def train_and_evaluate_model(model_ready_data):
    """
    Prepares data, trains the model, and evaluates its performance.
    """
    print("\n--- 2. Starting Model Training & Evaluation ---")

    target_column = 'Gold_Price_Poun_INR'
    y = model_ready_data[[target_column]]
    X = model_ready_data.drop(columns=[target_column])

    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = x_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(y_train)

    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    early_stopper = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

    print("-> Training model...")
    model.fit(
        X_train_scaled, y_train_scaled,
        epochs=300, batch_size=32, validation_split=0.1,
        callbacks=[early_stopper], verbose=0
    )
    print("Model training complete.")

    X_test_scaled = x_scaler.transform(X_test)
    predicted_scaled = model.predict(X_test_scaled, verbose=0)
    predicted_prices = y_scaler.inverse_transform(predicted_scaled)
    
    rmse = np.sqrt(mean_squared_error(y_test, predicted_prices))
    mape = np.mean(np.abs((y_test.values - predicted_prices) / y_test.values)) * 100
    accuracy = 100 - mape
    
    print("\n--- FINAL MODEL PERFORMANCE ---")
    print(f"Root Mean Squared Error (RMSE): ₹{rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Final Model Accuracy: {accuracy:.2f}%")
    print("---------------------------------")
    
    # Visualization
    plt.figure(figsize=(15, 7))
    plt.title('Model Evaluation: Actual vs. Predicted Prices', fontsize=16)
    plt.plot(y_test.index, y_test.values, label='Actual Price', color='blue')
    plt.plot(y_test.index, predicted_prices, label='Predicted Price', color='red', linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model, x_scaler, y_scaler, model_ready_data

def forecast_future(model, x_scaler, y_scaler, full_data):
    """
    Forecasts the gold price for the next 30 days.
    """
    print("\n--- 3. Forecasting Next 30 Days ---")
    
    last_known_features = full_data.drop(columns=['Gold_Price_Poun_INR']).iloc[[-1]]
    future_predictions = []

    for _ in range(30):
        input_scaled = x_scaler.transform(last_known_features)
        pred_scaled = model.predict(input_scaled, verbose=0)
        next_day_pred = y_scaler.inverse_transform(pred_scaled)[0][0]
        future_predictions.append(next_day_pred)
        
        new_features_row = last_known_features.copy()
        for lag in [60, 30, 7, 3]:
            if f'Lag_{lag}' in new_features_row.columns:
                prev_lag = [l for l in [30,7,3,1] if l < lag]
                if prev_lag:
                    new_features_row[f'Lag_{lag}'] = new_features_row[f'Lag_{max(prev_lag)}']

        new_features_row['Lag_1'] = next_day_pred
        last_known_features = new_features_row

    last_date = full_data.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30)
    
    forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_Price': future_predictions})
    print(forecast_df)

    # Visualization
    plt.figure(figsize=(15, 7))
    plt.title('30-Day Gold Price Forecast', fontsize=16)
    plt.plot(full_data.index[-100:], full_data['Gold_Price_Poun_INR'][-100:], label='Historical Price')
    plt.plot(future_dates, future_predictions, label='Forecasted Price', color='red', linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Run the entire pipeline
    processed_data = run_data_pipeline()
    trained_model, xs, ys, full_dataset = train_and_evaluate_model(processed_data)
    forecast_future(trained_model, xs, ys, full_dataset)
