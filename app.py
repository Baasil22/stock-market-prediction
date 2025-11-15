import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model
from flask import Flask, render_template, request, send_file
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import os
import warnings
import time
from pathlib import Path


# ============================================================================
# SANITY CHECK: Verify requests_cache is NOT loaded
# ============================================================================

import sys
if 'requests_cache' in sys.modules:
    raise ImportError(
        "CRITICAL ERROR: requests_cache is already loaded! "
        "This will break yfinance curl_cffi. "
        "Please uninstall: pip uninstall requests_cache -y"
    )


# ============================================================================
# CONFIGURATION
# ============================================================================

CACHE_DIR = ".cache"
Path(CACHE_DIR).mkdir(exist_ok=True)

warnings.filterwarnings('ignore')


# ============================================================================
# IMPORTS - SIMPLE AND CLEAN
# ============================================================================

import yfinance as yf
from newsapi import NewsApiClient


# ============================================================================
# VERIFY IMPORTS
# ============================================================================

# Double-check yfinance doesn't have requests_cache session
if 'requests_cache' in str(yf.__dict__):
    print("WARNING: requests_cache found in yfinance namespace")


# ============================================================================
# API CONFIGURATION
# ============================================================================

NEWS_API_KEY = os.environ.get('NEWS_API_KEY')
newsapi = NewsApiClient(api_key=NEWS_API_KEY) if NEWS_API_KEY else None

if not NEWS_API_KEY:
    print("Warning: NEWS_API_KEY not set. News feature will be disabled.")


# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================

app = Flask(__name__)
os.makedirs('static', exist_ok=True)


# ============================================================================
# MODEL LOADING
# ============================================================================

model = None
try:
    model = load_model('stock_dl_model.h5', compile=False)
    print("✓ Model loaded successfully. Input shape:", model.input_shape)
except Exception as e:
    print(f"⚠ Model load warning: {str(e)}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_rsi(prices, periods=14):
    """Calculate RSI (Relative Strength Index) indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def sanitize_filename(filename):
    """Sanitize filename to prevent directory traversal"""
    return ''.join(c for c in filename if c.isalnum() or c in ['_', '-'])


def download_stock_data(stock_symbol, retries=5):
    """
    Download stock data using yfinance.

    CRITICAL RULES:
    - Do NOT use any custom sessions
    - Do NOT pass session parameter to yf.Ticker()
    - Let yfinance handle requests natively
    - yfinance uses curl_cffi internally

    Args:
        stock_symbol (str): Stock ticker symbol (e.g., 'AAPL')
        retries (int): Number of retry attempts

    Returns:
        pd.DataFrame: Historical stock data
    """
    last_exception = None

    for attempt in range(1, retries + 1):
        try:
            print(f"Attempt {attempt}/{retries}: Fetching {stock_symbol}...")

            # RULE: Simple, direct call - NO custom session
            ticker = yf.Ticker(stock_symbol)
            df = ticker.history(period="2y", auto_adjust=True)

            # Validate data
            if df is not None and isinstance(df, pd.DataFrame) and len(df) > 100:
                print(f"✓ Successfully fetched {len(df)} days of data for {stock_symbol}")
                return df
            else:
                data_len = len(df) if df is not None else 0
                raise Exception(f"Insufficient data: only {data_len} rows returned (need >100)")

        except Exception as e:
            last_exception = e
            error_msg = str(e)

            # Don't log detailed errors, just retry
            print(f"✗ Attempt {attempt} failed: {error_msg[:80]}")

            # Exponential backoff
            if attempt < retries:
                wait_time = 2 ** (attempt - 1)
                print(f"  Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

    # All retries failed
    print(f"✗ Failed to fetch {stock_symbol} after {retries} attempts")
    if last_exception:
        raise last_exception
    else:
        raise Exception(f"Unable to fetch data for {stock_symbol}")


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
        return articles.get('articles', [])
    except Exception as e:
        print(f"News fetch warning: {e}")
        return []


def cleanup_old_files(stock_symbol, max_files_to_keep=2):
    """Clean up old generated files"""
    try:
        safe_symbol = sanitize_filename(stock_symbol)
        static_dir = 'static'

        if not os.path.exists(static_dir):
            return

        related_files = [
            f for f in os.listdir(static_dir) 
            if f.startswith(safe_symbol) and not f.endswith('.csv')
        ]

        if len(related_files) > max_files_to_keep:
            for f in related_files[max_files_to_keep:]:
                try:
                    file_path = os.path.join(static_dir, f)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Cleanup warning: {e}")
    except Exception as e:
        print(f"Cleanup error: {e}")


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main route"""
    current_date = dt.datetime.now().strftime('%Y-%m-%d')

    if request.method == 'POST':
        stock = request.form.get('stock', '').strip().upper()

        # Input validation
        if not stock:
            return render_template('index.html', 
                                   error="Please enter a stock symbol", 
                                   current_date=current_date)

        if not stock.isalnum() or len(stock) > 10:
            return render_template('index.html', 
                                   error="Invalid stock symbol", 
                                   current_date=current_date)

        try:
            print(f"\n{'='*60}")
            print(f"Processing: {stock}")
            print(f"{'='*60}")

            # Fetch data
            df = download_stock_data(stock)

            if df.empty or len(df) < 100:
                return render_template('index.html', 
                                       error=f"Insufficient data for {stock}", 
                                       current_date=current_date)

            # Calculate indicators
            print("Calculating indicators...")
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['RSI'] = calculate_rsi(df['Close'])

            # Prepare for model
            print("Preparing data...")
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df[['Close']])

            lookback = 100
            X = []
            for i in range(lookback, len(scaled_data)):
                X.append(scaled_data[i-lookback:i, 0])

            if not X:
                return render_template('index.html', 
                                       error="Not enough data", 
                                       current_date=current_date)

            X = np.array(X).reshape(len(X), lookback, 1)

            # Predictions
            predictions = None
            if model:
                try:
                    print("Making predictions...")
                    predictions = model.predict(X, verbose=0)
                    predictions = scaler.inverse_transform(predictions)

                    # Clear memory
                    from keras import backend as K
                    import gc
                    K.clear_session()
                    gc.collect()
                except Exception as e:
                    print(f"Prediction warning: {e}")

            safe_symbol = sanitize_filename(stock)
            cleanup_old_files(safe_symbol)

            # Charts
            print("Creating charts...")

            # EMA Chart
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df['Close'], label='Close', linewidth=2, color='#1f77b4')
            plt.plot(df.index, df['EMA_20'], label='EMA 20', alpha=0.7, color='#ff7f0e')
            plt.plot(df.index, df['EMA_50'], label='EMA 50', alpha=0.7, color='#2ca02c')
            plt.title(f"{stock} Price and EMAs", fontsize=14, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            ema_path = f"static/{safe_symbol}_ema.png"
            plt.savefig(ema_path, dpi=100)
            plt.close('all')

            # Prediction Chart
            plt.figure(figsize=(12, 6))
            plt.plot(df.index[lookback:], df['Close'][lookback:], 
                    label='Actual', linewidth=2, color='#1f77b4')
            if predictions is not None:
                plt.plot(df.index[lookback:], predictions, 
                        label='Predicted', alpha=0.7, linestyle='--', color='#d62728')
            plt.title(f"{stock} Price Prediction", fontsize=14, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            pred_path = f"static/{safe_symbol}_prediction.png"
            plt.savefig(pred_path, dpi=100)
            plt.close('all')

            # Metrics
            latest_close = float(df['Close'].iloc[-1])
            previous_close = float(df['Close'].iloc[-2])
            change = latest_close - previous_close
            change_pct = (change / previous_close) * 100
            avg_price = float(df['Close'].mean())
            high_52w = float(df['Close'].tail(252).max())
            low_52w = float(df['Close'].tail(252).min())

            # CSV
            csv_filename = f"{safe_symbol}_data.csv"
            df.to_csv(f"static/{csv_filename}")

            # News
            news_list = get_stock_news(stock)

            print(f"✓ {stock} processed successfully")
            print(f"{'='*60}\n")

            return render_template('index.html',
                                   stock_symbol=stock,
                                   current_date=current_date,
                                   last_date=df.index[-1].strftime('%Y-%m-%d'),
                                   latest_close=round(latest_close, 2),
                                   change=round(change, 2),
                                   change_pct=round(change_pct, 2),
                                   high_52w=round(high_52w, 2),
                                   low_52w=round(low_52w, 2),
                                   avg_price=round(avg_price, 2),
                                   ema_chart=ema_path,
                                   pred_chart=pred_path,
                                   data_desc=df.describe().to_html(classes='table table-striped'),
                                   csv_path=csv_filename,
                                   news_list=news_list,
                                   prediction_available=(predictions is not None))

        except Exception as e:
            print(f"\n✗ ERROR: {str(e)}\n")
            return render_template('index.html', 
                                   error="Unable to process request. Try again or verify symbol.", 
                                   current_date=current_date)

    return render_template('index.html', current_date=current_date)


@app.route('/download/<path:filename>')
def download_file(filename):
    """Download file"""
    try:
        if '..' in filename or '/' in filename:
            return "Invalid", 400

        file_path = os.path.join('static', filename)
        if not os.path.exists(file_path):
            return "Not found", 404

        return send_file(file_path, as_attachment=True)
    except Exception as e:
        return f"Error: {str(e)}", 500


@app.route('/health')
def health_check():
    """Health check"""
    return {
        'status': 'healthy',
        'timestamp': dt.datetime.now().isoformat(),
        'model_loaded': model is not None
    }, 200


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html', error="Page not found"), 404


@app.errorhandler(500)
def server_error(error):
    return render_template('index.html', error="Internal server error"), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n{'='*60}")
    print("Starting Stock Market Prediction App")
    print(f"Port: {port}")
    print(f"Model: {'Loaded' if model else 'Not found'}")
    print(f"News API: {'Enabled' if newsapi else 'Disabled'}")
    print(f"{'='*60}\n")

    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)