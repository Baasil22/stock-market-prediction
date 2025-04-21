import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model
from flask import Flask, render_template, request, send_file, url_for
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os
import warnings
from newsapi import NewsApiClient

# Initialize NewsAPI
newsapi = NewsApiClient(api_key='1eb85be4251c4d8db6c1c3b7cac6fc9b')  # Replace with your actual API key

# Function to get stock-related news
def get_stock_news(stock_symbol):
    try:
        articles = newsapi.get_everything(
            q=stock_symbol,
            language='en',
            sort_by='publishedAt',
            page_size=5
        )
        return articles['articles']
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

# Initialize Flask app
app = Flask(__name__)
os.makedirs('static', exist_ok=True)

# Load model with input shape verification
try:
    model = load_model('stock_dl_model.h5')
    print("Model loaded successfully. Input shape:", model.input_shape)
except Exception as e:
    model = None
    print(f"Warning: Could not load model - {str(e)}")

# Calculate RSI function
def calculate_rsi(prices, periods=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@app.route('/', methods=['GET', 'POST'])
def index():
    current_date = dt.datetime.now().strftime('%Y-%m-%d')
    
    if request.method == 'POST':
        stock = request.form.get('stock', '').strip().upper()
        if not stock:
            return render_template('index.html', error="Please enter a stock symbol", current_date=current_date)
        
        try:
            df = yf.download(stock, period="2y", progress=False)
            if df.empty:
                return render_template('index.html', error=f"No data found for {stock}", current_date=current_date)
            
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['RSI'] = calculate_rsi(df['Close'])
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df[['Close']])
            
            X = []
            lookback = 100
            for i in range(lookback, len(scaled_data)):
                X.append(scaled_data[i-lookback:i, 0])
            
            if len(X) == 0:
                return render_template('index.html', error=f"Not enough data to create sequences (need at least {lookback} days)", current_date=current_date)
            
            X = np.array(X)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            if model:
                try:
                    predictions = model.predict(X)
                    predictions = scaler.inverse_transform(predictions)
                except Exception as e:
                    return render_template('index.html', error=f"Model prediction failed: {str(e)}", current_date=current_date)
            else:
                predictions = np.zeros(len(X))
            
            safe_symbol = ''.join(c for c in stock if c.isalnum())
            
            # Price and EMA chart
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df['Close'], label='Close Price')
            plt.plot(df.index, df['EMA_20'], label='20-day EMA', alpha=0.7)
            plt.plot(df.index, df['EMA_50'], label='50-day EMA', alpha=0.7)
            plt.title(f"{stock} Price and Moving Averages")
            plt.legend()
            ema_path = f"static/{safe_symbol}_ema.png"
            plt.savefig(ema_path)
            plt.close()
            
            # Prediction chart
            plt.figure(figsize=(12, 6))
            plt.plot(df.index[lookback:], df['Close'][lookback:], label='Actual Price')
            if model:
                plt.plot(df.index[lookback:], predictions, label='Predicted Price', alpha=0.7)
            plt.title(f"{stock} Price Prediction")
            plt.legend()
            pred_path = f"static/{safe_symbol}_prediction.png"
            plt.savefig(pred_path)
            plt.close()
            
            # Metrics
            latest_close = float(df['Close'].iloc[-1])
            previous_close = float(df['Close'].iloc[-2])
            change = latest_close - previous_close
            change_pct = (change / previous_close) * 100
            
            # Save CSV
            csv_filename = f"{safe_symbol}_data.csv"
            csv_path = f"static/{csv_filename}"
            df.to_csv(csv_path)
            
            # Fetch news
            news_list = get_stock_news(stock)
            
            return render_template('index.html',
                                   stock_symbol=stock,
                                   current_date=current_date,
                                   last_date=df.index[-1].strftime('%Y-%m-%d'),
                                   latest_close=round(latest_close, 2),
                                   change=round(change, 2),
                                   change_pct=round(change_pct, 2),
                                   ema_chart=ema_path,
                                   pred_chart=pred_path,
                                   data_desc=df.describe().to_html(classes='table table-striped'),
                                   csv_path=csv_filename,
                                   news_list=news_list)
        
        except Exception as e:
            return render_template('index.html', error=f"Error processing {stock}: {str(e)}", current_date=current_date)
    
    return render_template('index.html', current_date=current_date)

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_file(os.path.join('static', filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
