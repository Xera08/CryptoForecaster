# forecast_data.py

import io
import base64

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error



def get_forecast_graph(days_ahead=30):
    # 1) Завантажуємо історію за останні 2 роки
    df = yf.download('BTC-USD', period='2y', interval='1d')
    data = df['Close'].values.reshape(-1, 1)

    # 2) Масштабуємо
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    # 3) Готовимо дані для LSTM: вікно 60 днів → прогноз 1 день уперед
    window = 60
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # 4) Будуємо просту LSTM-модель
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    # 5) Прогнозуємо останні days_ahead днів для backtest
    test_actual = df['Close'].values[-days_ahead:]
    test_scaled = scaled[-(window + days_ahead):]
    preds = []
    current = test_scaled[:window].copy()
    for i in range(days_ahead):
        inp = current[-window:].reshape(1, window, 1)
        pred = model.predict(inp, verbose=0)[0, 0]
        preds.append(pred)
        current = np.vstack([current, [[test_scaled[window + i][0]]]])  # use actual next value for backtest
    
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    mae = mean_absolute_error(test_actual, preds)
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=days_ahead)
    
    # 6) Малюємо графік
    plt.figure(figsize=(8,4))
    # реальні останні 60 днів
    plt.plot(df.index[-window:], df['Close'].values[-window:], label='Last 60 days')
    # прогноз
    plt.plot(future_dates, preds, label=f'Forecast {days_ahead} days')
    plt.title('Bitcoin Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price USD')
    plt.legend()
    plt.tight_layout()

    # 7) Конвертуємо в base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('ascii')
    buf.close()
    plt.close()

    return preds, [d.strftime('%Y-%m-%d') for d in future_dates], preds.tolist(), mae