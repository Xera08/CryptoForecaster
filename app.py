# app.py
from flask import Flask, render_template
from btc_data import get_btc_history

app = Flask(__name__)

@app.route('/')
def index():
    dates, prices = get_btc_history()
    return render_template('index.html', dates=dates, prices=prices)

if __name__ == '__main__':
    app.run(debug=True)
