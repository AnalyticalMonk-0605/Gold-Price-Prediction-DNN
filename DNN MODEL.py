import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def train_final_model():
    
    print("Model Training")

    #1. Load Data
    model_ready_data = pd.read_excel('gold_rate_features.xlsx', index_col=0, parse_dates=True)

    #2.Prepare Data
    target_column = 'Gold_Price_Poun_INR'
    y = model_ready_data[[target_column]]
    X = model_ready_data.drop(columns=[target_column, 'Gold_Price_USD', 'Oil_Price_USD', 'USD_INR'])

    split_index = int(len(X) * 0.8)
    X_train, _ = X[:split_index], X[split_index:]
    y_train, _ = y[:split_index], y[split_index:]

    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = x_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(y_train)

    joblib.dump(x_scaler, 'x_scaler.pkl')
    joblib.dump(y_scaler, 'y_scaler.pkl')
    

    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.2),
        
        Dense(64, activation='relu'),
        Dropout(0.2),
        
        Dense(1)
    ])
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.000001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.summary()
    
    # Increase patience as our model is learning very carefully
    early_stopper = EarlyStopping(monitor='val_loss', patience=60, restore_best_weights=True)

    print("\n-> Starting training...")
    model.fit(
        X_train_scaled, y_train_scaled,
        epochs=300,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopper],
        verbose=1
    )

    model.save('gold_price_predictor_final.h5')
    print("\nFinal model training complete. Model saved to 'gold_price_predictor_final.h5'")

if __name__ == "__main__":
    train_final_model()
