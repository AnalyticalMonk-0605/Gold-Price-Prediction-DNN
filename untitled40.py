# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 15:18:31 2025

@author: sanja
"""
import openpyxl
import yfinance as yf
import pandas as pd

# --- Define the assets and the date range ---
start_date = "2014-01-01"
end_date = "2025-09-16" # Use a recent date

# Tickers for Yahoo Finance ---
# GC=F -> Gold Futures (in USD per ounce)
# CL=F -> Crude Oil Futures (in USD per barrel)
# INR=X -> USD/INR Exchange Rate


print("Downloading historical data...")
gold_usd_data = yf.download('GC=F', start=start_date, end=end_date)
oil_usd_data = yf.download('CL=F', start=start_date, end=end_date)
usd_inr_data = yf.download('INR=X', start=start_date, end=end_date)
print("Data download complete.")

# --- Prepare and Combine the Data ---
df = pd.DataFrame(index=gold_usd_data.index)

df['Gold_Price_USD'] = gold_usd_data['Close']
df['Oil_Price_USD'] = oil_usd_data['Close']
df['USD_INR'] = usd_inr_data['Close']

# --- NEW CALCULATION: Gold Price per Poun (8 grams) in INR ---
# 1. Get price per gram: (Price_in_USD / 28.35 grams_per_ounce) * USD_INR_Rate
# 2. Multiply by 8 to get the price for 8 grams (1 poun).
price_per_gram_inr = (df['Gold_Price_USD'] / 28.35) * df['USD_INR']
df['Gold_Price_Poun_INR'] = price_per_gram_inr * 8  # <-- This is the updated logic

# --- Clean the final dataframe ---
df.ffill(inplace=True)
df.dropna(inplace=True)

# Select the final columns for the project. Our target is now 'Gold_Price_Poun_INR'
final_real_df = df[['Gold_Price_Poun_INR', 'USD_INR', 'Oil_Price_USD']]

print("\n--- Real-World Data (Price per Poun) ---")
print(final_real_df.head())
print("\n--- Last few rows of data ---")
print(final_real_df.tail())
print(f"\nTotal rows of real data collected: {len(final_real_df)}")

df['PriceLag1'] = df['Gold_Price_Poun_INR'].shift(1)
df['PriceLag2'] = df['Gold_Price_Poun_INR'].shift(3)
df['PriceLag3'] = df['Gold_Price_Poun_INR'].shift(7)
df['PriceLag4'] = df['Gold_Price_Poun_INR'].shift(30)
df['PriceLog5'] = df['Gold_Price_Poun_INR'].shift(60)

df['MA_5'] = df['Gold_Price_Poun_INR'].rolling(window = 5).mean()
df['MA_10'] = df['Gold_Price_Poun_INR'].rolling(window = 10).mean()
df['MA_15'] = df['Gold_Price_Poun_INR'].rolling(window = 15).mean()
df['MA_20'] = df['Gold_Price_Poun_INR'].rolling(window = 20).mean()

delta = df['Gold_Price_Poun_INR'].diff()
gain = (delta.where(delta > 0,0)).rolling(window = 14).mean()
loss = (delta.where(-delta < 0,0)).rolling(window = 14).mean()
epsilon = 1e-10
rs = gain / (loss+epsilon)
df['RSI'] = 100/(100 - (1/rs))

print(df)
df.to_excel("F:\personal projec\gold rate\Df.xlsx")

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import joblib # For saving the scalers
model_ready_data = df.copy()

# Assume 'model_ready_data' is the final DataFrame from our last script.
# model_ready_data = pd.read_excel('gold_rate_features.xlsx', index_col=0)

# --- 1. Define Features (X) and Target (y) ---
target_column = 'Gold_Price_Poun_INR'

# The target 'y' is the column we want to predict.
y = model_ready_data[[target_column]]

# The features 'X' are all the other columns that the model will learn from.
X = model_ready_data.drop(columns=[target_column])

# --- 2. Perform a Sequential Train-Test Split (80% Train, 20% Test) ---
split_percentage = 0.8
split_index = int(len(X) * split_percentage)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print("--- Data Splitting ---")
print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size:  {len(X_test)} samples")

# --- 3. Scale the Data ---
# Create scalers to normalize data between 0 and 1
x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))

# Fit the scaler on the training data and transform it
X_train_scaled = x_scaler.fit_transform(X_train)
y_train_scaled = y_scaler.fit_transform(y_train)

# Use the already-fitted scaler to transform the test data
X_test_scaled = x_scaler.transform(X_test)
y_test_scaled = y_scaler.transform(y_test)

# --- Save the scalers for later use in live prediction ---
joblib.dump(x_scaler, 'x_scaler.pkl')
joblib.dump(y_scaler, 'y_scaler.pkl')
model_ready_data.dropna(inplace=True)
model_ready_data.to_excel("F:/personal projec/gold rate/Df.xlsx")

print("\n✅ Scalers have been saved to files (x_scaler.pkl, y_scaler.pkl).")







import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Assume the following variables are already created from the previous step:
# X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled

# --- Phase 1: Designing the Model's "Brain" ---
# We are building the structure of our robot's brain, layer by layer.
model = Sequential()

# Input Layer: The "eyes and ears." This layer is designed to accept all our features at once.
# The 'input_shape' part automatically figures out how many features we have.
model.add(Dense(200, activation='relu', input_shape=(X_train_scaled.shape[1],)))

# Hidden Layers: The "deep thinking" parts of the brain where complex patterns are found.
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))

# Output Layer: The "mouth." It has only one neuron to give us one single answer: the predicted price.
model.add(Dense(1))

# --- Phase 2: Giving the Model a "Learning Plan" ---
# Before the robot can learn, we need to tell it HOW to learn.
model.compile(optimizer='adam', loss='mean_squared_error')

# Let's print a summary of the brain we just designed.
print("--- Model Architecture ---")
model.summary()

# --- Phase 3: The "Study Session" (Training) ---
# Now we tell the robot to start studying our historical data.
print("\n--- Starting Model Training ---")

# We show the model the practice questions (X_train_scaled) and the answers (y_train_scaled).
# An 'epoch' is like studying the entire dataset one full time.
history = model.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# --- Save the "Smart" Model ---
# After studying, the robot's brain is now smart. We save it to a file.
model.save('F:\personal projec\gold rate\gold_price_predictor.h5')
print("\n✅ Training complete. The smart model has been saved to 'gold_price_predictor.h5'.")