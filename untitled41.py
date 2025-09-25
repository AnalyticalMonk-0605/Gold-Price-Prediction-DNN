import pandas as pd
import yfinance as yf
from datetime import date

def run_data_pipeline():
    """
    Fetches market data, engineers features, and saves the final dataset.
    """
    print("--- Starting Data Pipeline ---")
    
    # --- 1. Fetching Data ---
    print("-> Fetching market data...")
    start = "2014-01-01"
    end = date.today().isoformat()
    
    tickers = {'GC=F': 'Gold_Price_USD', 'CL=F': 'Oil_Price_USD', 'INR=X': 'USD_INR'}
    data = yf.download(list(tickers.keys()), start=start, end=end)['Close']
    data.rename(columns=tickers, inplace=True)
    
    # --- 2. Initial Processing ---
    price_per_gram = (data['Gold_Price_USD'] / 28.35) * data['USD_INR']
    data['Gold_Price_Poun_INR'] = price_per_gram * 8
    data.ffill(inplace=True)
    
    # --- 3. Feature Engineering ---
    print("-> Engineering features...")
    target = 'Gold_Price_Poun_INR'
    
    # Lags
    data['Lag_1'] = data[target].shift(1)
    data['Lag_3'] = data[target].shift(3)
    data['Lag_7'] = data[target].shift(7)
    data['Lag_30'] = data[target].shift(30)
    data['Lag_60'] = data[target].shift(60)
    
    # Moving Averages
    data['MA_5'] = data[target].rolling(window=5).mean()
    data['MA_10'] = data[target].rolling(window=10).mean()
    data['MA_15'] = data[target].rolling(window=15).mean()
    data['MA_20'] = data[target].rolling(window=20).mean()
    
    # RSI (Corrected Formula)
    delta = data[target].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # --- 4. Final Cleanup and Save ---
    final_data = data.dropna()
    output_filename = 'gold_rate_features.xlsx'
    final_data.to_excel(output_filename)
    
    print(f"✅ Pipeline complete. Clean data saved to '{output_filename}'")
    print(final_data.tail())

if __name__ == "__main__":
    run_data_pipeline()
    



import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def train_prediction_model():
    """
    Loads data, prepares it, and trains the DNN model.
    """
    print("--- Starting Model Training ---")

    # --- 1. Load Data ---
    model_ready_data = pd.read_excel('gold_rate_features.xlsx', index_col=0)

    # --- 2. Prepare Data for Model ---
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

    # Save scalers
    joblib.dump(x_scaler, 'x_scaler.pkl')
    joblib.dump(y_scaler, 'y_scaler.pkl')
    
    # --- 3. Build and Train Model ---
    model = Sequential([
        Dense(200, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(200, activation='relu'),
        Dense(100, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    print("\n-> Starting training...")
    model.fit(
        X_train_scaled, y_train_scaled,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    # --- 4. Save Model ---
    model.save('gold_price_predictor.h5')
    print("\n✅ Training complete. Model saved to 'gold_price_predictor.h5'")

if __name__ == "__main__":
    train_prediction_model()    

import joblib
import pandas as pd
from tensorflow.keras.models import load_model

# Load test data
model_ready_data = pd.read_excel('gold_rate_features.xlsx', index_col=0)
target_column = 'Gold_Price_Poun_INR'
X_test = model_ready_data.drop(columns=[target_column]).iloc[int(len(model_ready_data)*0.8):]
y_test = model_ready_data[[target_column]].iloc[int(len(model_ready_data)*0.8):]

# Load trained scalers
x_scaler = joblib.load('x_scaler.pkl')
y_scaler = joblib.load('y_scaler.pkl')

# Load trained model
model = load_model('gold_price_predictor.h5')

# Scale and predict
X_test_scaled = x_scaler.transform(X_test)
y_test_scaled = y_scaler.transform(y_test)
y_pred_scaled = model.predict(X_test_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled)

# Compare last 10 predictions
for i in range(10):
    print(f"Actual: {y_test.values[i][0]:.2f}, Predicted: {y_pred[i][0]:.2f}")
    
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(y_test, label="Actual", color="blue")
plt.plot(y_pred, label="Predicted", color="red")
plt.legend()
plt.title("Gold Price Prediction")
plt.show()

