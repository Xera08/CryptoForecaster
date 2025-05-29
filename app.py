from flask import Flask, render_template
from forecast_data import get_forecast_graph
from backprop_forecast import generate_backprop_forecast_plot
import io
import base64
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error



app = Flask(__name__)

@app.route('/')
def index():
    # 1. Графік історичних цін
    df = yf.download('BTC-USD', period='1y', interval='1d')
    plt.figure(figsize=(8,4))
    plt.plot(df.index, df['Close'], label='BTC Close Price')
    plt.title('Bitcoin Price – Last Year')
    plt.xlabel('Date')
    plt.ylabel('Price USD')
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    hist_b64 = base64.b64encode(buf.read()).decode('ascii')
    buf.close()
    plt.close()

    # 2. LSTM-прогноз
    _, lstm_dates, lstm_prices, lstm_mae = get_forecast_graph(days_ahead=7)

    # 3. Прогноз із backpropagation
    backprop_dates, backprop_prices, backprop_mae = generate_backprop_forecast_plot()

    # Prepare dates for chart.js or template
    if ('Close', 'BTC-USD') not in df.columns or df.empty:
        dates = []
        prices = []
    else:
        dates = df.index.strftime('%Y-%m-%d').tolist()
        prices = df[('Close', 'BTC-USD')].tolist()


    

    return render_template(
        'index.html',
        hist_graph=hist_b64,
        lstm_dates=lstm_dates,
        lstm_prices=lstm_prices,
        dates=dates,
        prices=prices,
        backprop_dates=[d.strftime('%Y-%m-%d') for d in backprop_dates],
        backprop_prices=backprop_prices,
        lstm_mae=lstm_mae,
        backprop_mae=backprop_mae
    )

if __name__ == '__main__':
    app.run(debug=True)
