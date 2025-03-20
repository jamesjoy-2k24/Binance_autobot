from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
import sys
from binance.client import Client
from dotenv import load_dotenv
import os
import talib
import numpy as np  # Import NumPy explicitly

# Fix Unicode printing issues
sys.stdout.reconfigure(encoding="utf-8")

# Load API keys
load_dotenv()
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

# Connect to Binance API
client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)

# Fetch Historical Data
def get_historical_data(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1MINUTE, lookback="1000"):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=lookback)
    df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                                       'quote_asset_volume', 'trades', 'taker_buy_base_asset_volume', 
                                       'taker_buy_quote_asset_volume', 'ignore'])
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    return df

# Compute Indicators
def compute_indicators(df):
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['Volume_change'] = df['volume'].pct_change()  # Use percentage change of volume
    return df

# Generate synthetic labels (example strategy)
def generate_labels(df):
    # Simple strategy: Buy if price increases significantly, Sell if it drops, Hold otherwise
    df['price_change'] = df['close'].pct_change()
    conditions = [
        (df['price_change'] > 0.001),  # Price up by 0.1%
        (df['price_change'] < -0.001),  # Price down by 0.1%
    ]
    choices = ["BUY", "SELL"]
    df['label'] = np.select(conditions, choices, default="HOLD")  # Use np.select instead of pd.np.select
    return df

# Fetch and prepare data
df = get_historical_data(lookback="1000")  # Get 1000 minutes of data
df = compute_indicators(df)
df = generate_labels(df)

# Define feature names
feature_names = ["RSI", "MACD", "Volume_change"]

# Prepare training data
X_train = df[feature_names].dropna()
y_train = df['label'][X_train.index]  # Align labels with feature rows after dropna

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "ai_trading_model.pkl")

print("✅ Model trained and saved as 'ai_trading_model.pkl'")
print(f"Training data shape: {X_train.shape}")
print(f"Feature names: {model.feature_names_in_}")
print(f"Label distribution:\n{y_train.value_counts()}")