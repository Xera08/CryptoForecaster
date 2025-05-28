# btc_data.py
import yfinance as yf

def get_btc_history():
    btc = yf.Ticker("BTC-USD")
    hist = btc.history(period="1y")  # 1 рік

    dates = hist.index.strftime('%Y-%m-%d').tolist()
    prices = hist['Close'].round(2).tolist()

    return dates, prices
