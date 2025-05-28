# backprop_forecast.py

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import datetime
import io
import base64

def generate_backprop_forecast_plot():
    # 1. Завантаження даних
    df = yf.download("BTC-USD", period="2y", interval="1d")
    data = df['Close'].values.reshape(-1, 1)

    # 2. Масштабування
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # 3. Підготовка вибірки
    window = 7
    forecast_days = 7
    X, y = [], []
    for i in range(len(data_scaled) - window - forecast_days):
        X.append(data_scaled[i:i+window].flatten())
        y.append(data_scaled[i+window:i+window+forecast_days].flatten())
    X, y = np.array(X), np.array(y)

    # 4. Тренувальний розподіл
    split = int(len(X) * 0.8)
    X_train = X[:split]
    y_train = y[:split]

    # 5. Модель
    np.random.seed(1)
    input_dim = window
    hidden_dim = 64
    output_dim = forecast_days

    W1 = np.random.randn(input_dim, hidden_dim) * 0.1
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, output_dim) * 0.1
    b2 = np.zeros((1, output_dim))

    def relu(x): return np.maximum(0, x)
    def relu_deriv(x): return (x > 0).astype(float)

    # 6. Навчання
    lr = 0.01
    epochs = 500
    for epoch in range(epochs):
        z1 = X_train @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        y_pred = z2
        loss = np.mean((y_pred - y_train) ** 2)

        dloss = 2 * (y_pred - y_train) / y_train.shape[0]
        dW2 = a1.T @ dloss
        db2 = np.sum(dloss, axis=0, keepdims=True)
        da1 = dloss @ W2.T
        dz1 = da1 * relu_deriv(z1)
        dW1 = X_train.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    # 7. Прогнозування
    last_seq = data_scaled[-window:].flatten().reshape(1, -1)
    z1 = last_seq @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    pred_scaled = z2.flatten()
    predicted = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

    # 8. Побудова графіка
    last_date = df.index[-1]
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 8)]

    plt.figure(figsize=(8,4))
    plt.plot(future_dates, predicted, label="Forecast (Backpropagation)", marker='o')
    plt.title("Bitcoin Forecast – 7-Day Backpropagation")
    plt.xlabel("Date")
    plt.ylabel("Price USD")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 9. Перетворення в base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('ascii')
    buf.close()
    plt.close()

    return img_b64
