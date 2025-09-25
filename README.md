## **Gold Price Prediction using Deep Learning**

A comprehensive project to forecast the daily price of gold using a Deep Neural Network (DNN). This repository explores the entire machine learning pipeline, from data collection and feature engineering to model training, evaluation, and future forecasting.



The model is trained on historical market data, incorporating a variety of technical and economic indicators to make context-aware predictions. This project serves as a practical application of time-series analysis and deep learning with TensorFlow.



### Key Features

Data Collection: Automatically fetches the latest historical data for Gold (USD), Crude Oil prices, and the USD/INR exchange rate using the yfinance library.



Feature Engineering: Enriches the dataset with powerful predictive features, including:



Lag Features: Price history from previous days (1, 3, 7, 30, 60 days ago).



Moving Averages (MA): Short and long-term trend indicators (5, 10, 15, 20-day MAs).



Relative Strength Index (RSI): A momentum indicator to identify overbought or oversold conditions.



Deep Learning Model: A robust Deep Neural Network built with TensorFlow and Keras, designed to capture complex, non-linear patterns in the financial data.



Model Evaluation: Provides a clear performance assessment using standard regression metrics like Root Mean Squared Error (RMSE) and an intuitive final accuracy percentage.



Forecasting: Includes a script to predict gold prices for the next 30 days based on the most recent available data.



Tech Stack

Python 3.x



TensorFlow \& Keras: For building and training the deep learning model.



Pandas: For data manipulation and analysis.



scikit-learn: For data preprocessing (scaling) and model evaluation.



yfinance: For downloading financial market data.



Matplotlib: For visualizing the results.



Git \& GitHub: For version control.



### Future Work

This project provides a strong foundation. Future improvements could include:



Adding Macroeconomic Indicators: Integrating key data like interest rates, inflation (CPI), and the VIX "fear index".



News Sentiment Analysis: Your original idea to incorporate a sentiment score from financial news headlines to capture market-moving events.



Advanced Model Architectures: Experimenting with models specifically designed for time-series data, such as LSTMs or GRUs.

