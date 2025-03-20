# Load API Keys
from dotenv import load_dotenv
import os

load_dotenv()

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Connect to Binance API
from binance.client import Client

client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)

# Fetch Historical Data
import pandas as pd

def get_historical_data(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1MINUTE, lookback="50"):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=lookback)
    df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                                       'quote_asset_volume', 'trades', 'taker_buy_base_asset_volume', 
                                       'taker_buy_quote_asset_volume', 'ignore'])
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    return df

# Compute Indicators (RSI, MACD, Volume_change)
import talib

def compute_indicators(df):
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['Volume_change'] = df['volume'].pct_change()  # Use Volume_change to match the trained model
    return df

# AI model for Decision Making
import joblib

model = joblib.load("ai_trading_model.pkl")

def get_trade_signal(df):
    # Use the exact features the model was trained on: RSI, MACD, Volume_change
    latest_data = df[['RSI', 'MACD', 'Volume_change']].dropna().tail(1)
    prediction = model.predict(latest_data)[0]
    return prediction

# Place Trade on Binance
def place_order(signal, quantity=0.001):
    if signal == "BUY":
        order = client.order_market_buy(symbol="BTCUSDT", quantity=quantity)
        print("Buy Order Placed:", order)
    elif signal == "SELL":
        order = client.order_market_sell(symbol="BTCUSDT", quantity=quantity)
        print("Sell Order Placed:", order)

# Send Trade Signal to Telegram
import telegram
import asyncio

bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)

async def send_telegram_alert(message):
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)

# Main Execution Loop
import time

while True:
    try:
        # Fetch and process data
        df = get_historical_data()
        df = compute_indicators(df)
        
        # Get trade signal
        trade_signal = get_trade_signal(df)
        print(f"Trade Signal: {trade_signal}")
        
        # Execute trade if signal is BUY or SELL
        if trade_signal in ["BUY", "SELL"]:
            place_order(trade_signal)
            asyncio.run(send_telegram_alert(f"Trade Executed: {trade_signal} on BTC/USDT"))
        else:
            print("Signal is HOLD, no action taken.")
        
        # Wait 1 minute before next iteration
        time.sleep(60)
    
    except Exception as e:
        print(f"Error occurred: {e}")
        time.sleep(60)  # Wait before retrying on error