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
    print(f"No data found for {ticker}.")
    exit()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# 2. Enhanced Feature Engineering
# Trend & Momentum
df['EMA_9'] = ta.ema(df['Close'], length=9)
df['EMA_21'] = ta.ema(df['Close'], length=21)
df['SMA_50'] = ta.sma(df['Close'], length=50)
df['SMA_200'] = ta.sma(df['Close'], length=200)
df['RSI'] = ta.rsi(df['Close'], length=14)

# Volatility & Volume
df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
df['VOL_SMA'] = ta.sma(df['Volume'], length=20)

# MACD
macd = ta.macd(df['Close'])
df = pd.concat([df, macd], axis=1) # Adds MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9

# Define horizons and features for AI
horizons = {"5m": 1, "15m": 3, "30m": 6, "60m": 12}
# Use MACD Histogram and VWAP in the AI features
feature_cols = ['RSI', 'EMA_9', 'EMA_21', 'SMA_50', 'Close', 'VWAP', 'MACDh_12_26_9']

print(f"--- {ticker} High-Confidence Analysis ---")
now = df.iloc[-1]
actual_now = now['Close']
vol_ratio = now['Volume'] / now['VOL_SMA']

# 3. Confidence & Signal Scoring
score = 0
reasons = []

# A. VWAP Analysis (The Day Trader's Anchor)
if actual_now > now['VWAP']:
    score += 1.5
    reasons.append("Price > VWAP (Bullish Bias)")
else:
    score -= 1.5
    reasons.append("Price < VWAP (Bearish Bias)")

# B. MACD Momentum
if now['MACDh_12_26_9'] > 0:
    score += 1
    reasons.append("MACD Histogram Positive (Improving Momentum)")
else:
    score -= 1
    reasons.append("MACD Histogram Negative (Declining Momentum)")

# C. Volume Confirmation
volume_confirmed = vol_ratio > 1.2 # 20% above average
if volume_confirmed:
    # Volume amplifies the existing score
    score *= 1.2
    reasons.append(f"High Volume Confirmation ({vol_ratio:.1f}x avg)")

# D. Standard EMA/SMA Checks
if now['EMA_9'] > now['EMA_21']: score += 0.5
if actual_now > now['SMA_50']: score += 0.5

# 4. AI Predictions
ai_preds = []
for label, steps in horizons.items():
    df_t = df.copy()
    df_t['Target'] = df_t['Close'].shift(-steps)
    df_t = df_t.dropna(subset=feature_cols + ['Target'])
    
    if len(df_t) < 500: continue
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(df_t[feature_cols], df_t['Target'])
    
    pred = model.predict(df[feature_cols].iloc[[-1]])[0]
    ai_preds.append((label, pred))
    if pred > actual_now: score += 0.25
    else: score -= 0.25

# 5. Determine Rating & Confidence
# Confidence is based on indicator agreement and volume
agreement_ratio = len([r for r in reasons if "Bullish" in r or "Positive" in r]) / len(reasons)
confidence = "LOW"
if volume_confirmed and (agreement_ratio > 0.8 or agreement_ratio < 0.2):
    confidence = "HIGH"
elif volume_confirmed or (0.7 > agreement_ratio > 0.3):
    confidence = "MEDIUM"

rating = "NEUTRAL"
if score >= 4: rating = "STRONG BUY"
elif 1.5 <= score < 4: rating = "BUY"
elif -1.5 < score <= -4: rating = "SELL"
elif score <= -4: rating = "STRONG SELL"

# 6. Output
print(f"Current Price: ${actual_now:.2f} | VWAP: ${now['VWAP']:.2f}")
print(f"Volatility (ATR): ${now['ATR']:.2f}")
print("-" * 50)
print(f"CONFIDENCE LEVEL: {confidence}")
print(f"OVERALL RATING:   {rating} (Score: {score:+.1f})")
print("-" * 50)
print("CONFIRMING FACTORS:")
for r in reasons: print(f"  [>] {r}")

print("-" * 50)
print("AI PRICE TARGETS:")
for label, pred in ai_preds:
    diff = pred - actual_now
    print(f"  [~] {label} Target: ${pred:.2f} ({diff:+.2f})")
print("-" * 50)
