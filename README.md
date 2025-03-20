Trading Bot Project
This project is an automated trading bot that uses technical indicators (RSI, MACD, and Volume Change) to generate trade signals for BTC/USDT on Binance. It leverages a RandomForestClassifier model for decision-making, executes trades via the Binance API, and sends notifications to Telegram.

Table of Contents
Features
Prerequisites

Fetches historical BTC/USDT data from Binance.
Computes technical indicators: RSI, MACD, and Volume Change.
Uses a pre-trained RandomForestClassifier model to predict trade signals (BUY, SELL, HOLD).
Executes market orders on Binance based on trade signals.
Sends trade notifications to a Telegram chat.
Runs continuously, checking for signals every minute.
Prerequisites
Python 3.8 or higher
A Binance account with API keys
A Telegram bot and chat ID for notifications
Git (optional, for version control)

MIT License

Copyright (c) 2025 [James Joy]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
