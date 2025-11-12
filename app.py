import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model
from flask import Flask, render_template, request, send_file, url_for
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import os
import warnings
import time
from pathlib import Path

# Create cache directory for yfinance
CACHE_DIR = ".cache"
Path(CACHE_DIR).mkdir(exist_ok=True)

# Import yfinance and set cache location
import yfinance as yf
yf.set_tz_cache_location(CACHE_DIR)

from newsapi import NewsApiClient

# FIXED: Get API key from environment variable (NO DEFAULT VALUE FOR SECURITY)
NEWS_API_KEY = os.environ.get('NEWS_API_KEY')
if NEWS_API_KEY:
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
else:
    newsapi = None
    print("Warning: NEWS_API_KEY not set. News feature will be disabled.")

# Suppress warnings
warnings.filterwarnings('ignore')

def get_stock_news(stock_symbol):
    """Get stock-related news from NewsAPI"""
    if not newsapi:
        return []
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

# Load model
try:
    model = load_model('stock_dl_model.h5', compile=False)
    print("Model loaded successfully. Input shape:", model.input_shape)
except Exception as e:
    model = None
    print(f"Warning: Could not load model - {str(e)}")

def calculate_rsi(prices, periods=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def download_stock_data(stock_symbol, retries=3):
    """
    FIXED: Download stock data with standard yfinance (removed curl_cffi)
    """
    for attempt in range(retries):
        try:
            print(f"Attempt {attempt + 1}: Fetching {stock_symbol}...")
            
            # Use standard yfinance
            ticker = yf.Ticker(stock_symbol)
            
            # Download historical data
            df = ticker.history(period="2y", auto_adjust=True)
            
            if not df.empty:
                print(f"Successfully fetched {len(df)} days of data for {stock_symbol}")
                return df
            
            # If empty, wait and retry
            if attempt < retries - 1:
                print(f"Empty data, retrying in 2 seconds...")
                time.sleep(2)
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < retries - 1:
                time.sleep(3)
            else:
                raise
    
    return pd.DataFrame()

@app.route('/', methods=['GET', 'POST'])
def index():
    current_date = dt.datetime.now().strftime('%Y-%m-%d')
    
    if request.method == 'POST':
        stock = request.form.get('stock', '').strip().upper()
        
        # Input validation
        if not stock:
            return render_template('index.html', error="Please enter a stock symbol", current_date=current_date)
        
        if not stock.isalnum() or len(stock) > 10:
            return render_template('index.html', error="Invalid stock symbol format", current_date=current_date)
        
        try:
            # Download data
            df = download_stock_data(stock)
            
            if df.empty:
                return render_template('index.html', 
                                     error=f"Unable to fetch data for {stock}. Please verify the symbol.", 
                                     current_date=current_date)
            
            # Calculate technical indicators
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['RSI'] = calculate_rsi(df['Close'])
            
            # Prepare data for model
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df[['Close']])
            
            X = []
            lookback = 100
            for i in range(lookback, len(scaled_data)):
                X.append(scaled_data[i-lookback:i, 0])
            
            if len(X) == 0:
                return render_template('index.html', 
                                     error=f"Not enough data (need at least {lookback} days)", 
                                     current_date=current_date)
            
            X = np.array(X)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            # Make predictions
            if model:
                try:
                    predictions = model.predict(X, verbose=0)
                    predictions = scaler.inverse_transform(predictions)
                    
                    # FIXED: Clear session to free memory
                    from keras import backend as K
                    import gc
                    K.clear_session()
                    gc.collect()
                except Exception as e:
                    print(f"Model prediction error: {str(e)}")
                    predictions = np.zeros((len(X), 1))
            else:
                predictions = np.zeros((len(X), 1))
            
            safe_symbol = ''.join(c for c in stock if c.isalnum())
            
            # Clean up old files with better error handling
            try:
                old_files = [f for f in os.listdir('static') if f.startswith(safe_symbol)]
                for f in old_files:
                    try:
                        os.remove(os.path.join('static', f))
                    except Exception as e:
                        print(f"Cleanup warning: {e}")
            except Exception as e:
                print(f"Cleanup warning: {e}")
            
            # Create EMA chart
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df['Close'], label='Close Price', linewidth=2)
            plt.plot(df.index, df['EMA_20'], label='20-day EMA', alpha=0.7, linewidth=1.5)
            plt.plot(df.index, df['EMA_50'], label='50-day EMA', alpha=0.7, linewidth=1.5)
            plt.title(f"{stock} Price and Moving Averages", fontsize=14, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Price (USD)', fontsize=12)
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            ema_path = f"static/{safe_symbol}_ema.png"
            plt.savefig(ema_path, dpi=100, bbox_inches='tight')
            plt.close('all')
            
            # Create prediction chart
            plt.figure(figsize=(12, 6))
            plt.plot(df.index[lookback:], df['Close'][lookback:], label='Actual Price', linewidth=2)
            if model:
                plt.plot(df.index[lookback:], predictions, label='Predicted Price', alpha=0.7, linewidth=2, linestyle='--')
            plt.title(f"{stock} Price Prediction", fontsize=14, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Price (USD)', fontsize=12)
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            pred_path = f"static/{safe_symbol}_prediction.png"
            plt.savefig(pred_path, dpi=100, bbox_inches='tight')
            plt.close('all')
            
            # Calculate metrics
            latest_close = float(df['Close'].iloc[-1])
            previous_close = float(df['Close'].iloc[-2])
            change = latest_close - previous_close
            change_pct = (change / previous_close) * 100
            
            # Save CSV
            csv_filename = f"{safe_symbol}_data.csv"
            csv_path = f"static/{csv_filename}"
            df.to_csv(csv_path)
            
            # Fetch news
            news_list = []
            try:
                news_list = get_stock_news(stock)
            except Exception as e:
                print(f"News API error: {str(e)}")
            
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
            error_msg = f"Error processing {stock}: {str(e)}"
            print(error_msg)
            return render_template('index.html', 
                                 error="Unable to process request. Please try again.", 
                                 current_date=current_date)
    
    return render_template('index.html', current_date=current_date)

@app.route('/download/<path:filename>')
def download_file(filename):
    try:
        return send_file(os.path.join('static', filename), as_attachment=True)
    except Exception as e:
        return f"Error downloading file: {str(e)}", 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
