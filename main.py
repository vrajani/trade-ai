import yfinance as yf
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import sys

# 1. Fetch Data
ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "TSLA"
df = yf.download(ticker, interval="5m", period="60d")

if df.empty:
    print(f"No data found for {ticker}. Please check the ticker symbol or market status.")
    exit()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# 2. Feature Engineering
# Moving Averages
df['EMA_9'] = ta.ema(df['Close'], length=9)
df['EMA_21'] = ta.ema(df['Close'], length=21)
df['SMA_20'] = ta.sma(df['Close'], length=20)
df['SMA_50'] = ta.sma(df['Close'], length=50)
df['SMA_200'] = ta.sma(df['Close'], length=200)

# Indicators
df['RSI'] = ta.rsi(df['Close'], length=14)
df['PCT_Change'] = df['Close'].pct_change()

# Define horizons
horizons = {"5m": 1, "15m": 3, "30m": 6, "60m": 12}
feature_cols = ['RSI', 'EMA_9', 'EMA_21', 'SMA_50', 'Close']

print(f"--- {ticker} Technical & AI Analysis ---")
actual_now = df['Close'].iloc[-1]
rsi_now = df['RSI'].iloc[-1]
ema9 = df['EMA_9'].iloc[-1]
ema21 = df['EMA_21'].iloc[-1]
sma50 = df['SMA_50'].iloc[-1]
sma200 = df['SMA_200'].iloc[-1]

# 3. Calculate Signals
signals = []
score = 0 # -5 to +5 scale

# EMA 9/21 Cross
if ema9 > ema21:
    signals.append("EMA 9/21: BULLISH (Crossover Up)")
    score += 1
else:
    signals.append("EMA 9/21: BEARISH (Crossover Down)")
    score -= 1

# Price vs SMA 50
if actual_now > sma50:
    signals.append("Price vs SMA 50: BULLISH (Above Trend)")
    score += 1
else:
    signals.append("Price vs SMA 50: BEARISH (Below Trend)")
    score -= 1

# RSI Context
if rsi_now < 30:
    signals.append("RSI: OVERSOLD (Potential Buy)")
    score += 1.5
elif rsi_now > 70:
    signals.append("RSI: OVERBOUGHT (Potential Sell)")
    score -= 1.5
else:
    signals.append(f"RSI: NEUTRAL ({rsi_now:.2f})")

# 4. AI Predictions
ai_predictions = {}
for label, steps in horizons.items():
    df_temp = df.copy()
    df_temp['Target'] = df_temp['Close'].shift(-steps)
    df_temp = df_temp.dropna(subset=feature_cols + ['Target'])
    
    if len(df_temp) < 100: continue

    X = df_temp[feature_cols]
    y = df_temp['Target']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Predict for the very latest data point
    current_features = df[feature_cols].iloc[[-1]]
    pred = model.predict(current_features)[0]
    ai_predictions[label] = pred
    
    # AI Score weighting
    if pred > actual_now: score += 0.5
    else: score -= 0.5

# 5. Final Rating Logic
rating = "NEUTRAL"
if score >= 3: rating = "STRONG BUY"
elif 1 <= score < 3: rating = "BUY"
elif -1 < score <= -3: rating = "SELL"
elif score <= -3: rating = "STRONG SELL"

# 6. Output
print(f"Current Price: ${actual_now:.2f}")
print("-" * 40)
print("TECHNICAL SIGNALS:")
for s in signals: print(f"  [>] {s}")
print(f"  [>] SMA 200: ${sma200:.2f} (Long-term Anchor)")

print("-" * 40)
print("AI PRICE PREDICTIONS:")
for label, pred in ai_predictions.items():
    diff = pred - actual_now
    print(f"  [~] {label}: ${pred:.2f} ({diff:+.2f})")

print("-" * 40)
print(f"OVERALL RATING: {rating} (Score: {score:+.1f})")
print("-" * 40)
