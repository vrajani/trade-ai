import yfinance as yf
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# 1. Fetch Data
ticker = "TSLA"
# 5m interval is available for up to 60 days
df = yf.download(ticker, interval="5m", period="60d")

if df.empty:
    print(f"No data found for {ticker}. Please check the ticker symbol or market status.")
    exit()

# Flatten MultiIndex columns if necessary (common in newer yfinance versions)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

print(f"Fetched {len(df)} data points for {ticker}")

# 2. Feature Engineering with pandas_ta
df['RSI'] = ta.rsi(df['Close'], length=14)
df['SMA_20'] = ta.sma(df['Close'], length=20)
df['SMA_50'] = ta.sma(df['Close'], length=50)
df['PCT_Change'] = df['Close'].pct_change()

# Define the prediction horizons (in 5-minute increments)
# 5 mins = 1 step, 15 mins = 3 steps, 30 mins = 6 steps, 60 mins = 12 steps
horizons = {
    "5 mins": 1,
    "15 mins": 3,
    "30 mins": 6,
    "60 mins": 12
}

# Features (X)
feature_cols = ['RSI', 'SMA_20', 'SMA_50', 'Close']

# 3. Clean and Train
print(f"--- {ticker} Multi-Horizon Analysis ---")
actual_now = df['Close'].iloc[-1]
print(f"Current Price: ${actual_now:.2f}")
print(f"Current RSI: {df['RSI'].iloc[-1]:.2f}")
print("-" * 30)

# We'll store the models and predictions
for label, steps in horizons.items():
    # Create target for this specific horizon
    df_temp = df.copy()
    df_temp['Target'] = df_temp['Close'].shift(-steps)
    
    # Drop NaNs for this specific training set
    df_temp = df_temp.dropna(subset=feature_cols + ['Target'])
    
    if len(df_temp) < 100:
        print(f"Not enough data to train for {label}")
        continue

    X = df_temp[feature_cols]
    y = df_temp['Target']

    # Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Predict for the very latest data point
    current_features = np.array(df[feature_cols].iloc[-1]).reshape(1, -1)
    prediction = model.predict(current_features)[0]
    
    diff = prediction - actual_now
    signal = "BULLISH" if diff > 0 and df['RSI'].iloc[-1] < 70 else "BEARISH/NEUTRAL"
    
    print(f"Prediction for {label}: ${prediction:.2f} ({diff:+.2f}) -> {signal}")

print("-" * 30)
