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
    # The 'auto_adjust=False' is added to prevent potential future issues with yfinance updates
    data = yf.download(list(tickers.keys()), start=start, end=end, auto_adjust=False)['Close']
    data.rename(columns=tickers, inplace=True)
    
    # --- 2. Initial Processing ---
    price_per_gram = (data['Gold_Price_USD'] / 28.35) * data['USD_INR']
    data['Gold_Price_Poun_INR'] = price_per_gram * 8
    data.ffill(inplace=True)
    
    # --- 3. Feature Engineering ---
    print("-> Engineering features...")
    target = 'Gold_Price_Poun_INR'
    
    data['Lag_1'] = data[target].shift(1)
    data['Lag_3'] = data[target].shift(3)
    data['Lag_7'] = data[target].shift(7)
    data['Lag_30'] = data[target].shift(30)
    data['Lag_60'] = data[target].shift(60)
    
    data['MA_5'] = data[target].rolling(window=5).mean()
    data['MA_10'] = data[target].rolling(window=10).mean()
    data['MA_15'] = data[target].rolling(window=15).mean()
    data['MA_20'] = data[target].rolling(window=20).mean()
    
    delta = data[target].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10) # Added epsilon for safety
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # --- 4. Final Cleanup and Save ---
    final_data = data.dropna()
    output_filename = 'gold_rate_features.xlsx'
    final_data.to_excel(output_filename)
    
    print(f"✅ Pipeline complete. Clean data saved to '{output_filename}'")
    print("\n--- Final Data Sample ---")
    print(final_data.tail())

if __name__ == "__main__":
    run_data_pipeline()
