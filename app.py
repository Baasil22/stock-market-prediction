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
import logging
from functools import wraps
import signal


# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# SANITY CHECK
# ============================================================================

import sys
if 'requests_cache' in sys.modules:
    raise ImportError("requests_cache is loaded - uninstall it!")


# ============================================================================
# CONFIGURATION
# ============================================================================

CACHE_DIR = ".cache"
Path(CACHE_DIR).mkdir(exist_ok=True)

warnings.filterwarnings('ignore')

IS_RENDER = os.environ.get('RENDER') == 'true'
YFINANCE_TIMEOUT = 45  # Max 45 seconds for yfinance
DATA_PERIOD = "1y"     # Only 1 year of data (faster)


# ============================================================================
# IMPORTS
# ============================================================================

import yfinance as yf
from newsapi import NewsApiClient


# ============================================================================
# API CONFIGURATION
# ============================================================================

NEWS_API_KEY = os.environ.get('NEWS_API_KEY')
newsapi = NewsApiClient(api_key=NEWS_API_KEY) if NEWS_API_KEY else None


# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__)
os.makedirs('static', exist_ok=True)


# ============================================================================
# MODEL LOADING
# ============================================================================

model = None
try:
    model = load_model('stock_dl_model.h5', compile=False)
    logger.info(f"✓ Model loaded. Shape: {model.input_shape}")
except Exception as e:
    logger.warning(f"Model: {str(e)}")


# ============================================================================
# TIMEOUT DECORATOR
# ============================================================================

class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Function call timed out")


def with_timeout(seconds):
    """Decorator to add timeout to function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if IS_RENDER:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
                if IS_RENDER:
                    signal.alarm(0)
                return result
            except TimeoutException:
                logger.error(f"TIMEOUT: {func.__name__}")
                raise Exception(f"{func.__name__} timed out after {seconds}s")
            except Exception as e:
                if IS_RENDER:
                    signal.alarm(0)
                raise
        return wrapper
    return decorator


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_rsi(prices, periods=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def sanitize_filename(filename):
    """Sanitize filename"""
    return ''.join(c for c in filename if c.isalnum() or c in ['_', '-'])


@with_timeout(YFINANCE_TIMEOUT)
def fetch_yfinance_data(stock_symbol):
    """Fetch data from yfinance with timeout"""
    logger.info(f"Fetching {stock_symbol}...")
    ticker = yf.Ticker(stock_symbol)
    df = ticker.history(period=DATA_PERIOD, auto_adjust=True)

    if df is None or len(df) < 50:
        raise Exception(f"Insufficient data: {len(df) if df is not None else 0} rows")

    logger.info(f"✓ Got {len(df)} rows")
    return df


def download_stock_data(stock_symbol, retries=2):
    """Download stock data with timeout protection"""
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Attempt {attempt}/{retries}")
            df = fetch_yfinance_data(stock_symbol)
            return df

        except Exception as e:
            last_error = e
            logger.warning(f"✗ Attempt {attempt}: {str(e)[:60]}")

            if attempt < retries:
                wait = 5 * attempt
                logger.info(f"Waiting {wait}s...")
                time.sleep(wait)

    raise last_error or Exception("Failed to fetch data")


def get_stock_news(stock_symbol):
    """Get news (with timeout handling)"""
    if not newsapi:
        return []
    try:
        articles = newsapi.get_everything(
            q=stock_symbol,
            language='en',
            sort_by='publishedAt',
            page_size=3
        )
        return articles.get('articles', [])
    except Exception as e:
        logger.warning(f"News: {e}")
        return []


def cleanup_old_files(stock_symbol):
    """Clean up old files"""
    try:
        safe_symbol = sanitize_filename(stock_symbol)
        for f in os.listdir('static'):
            if f.startswith(safe_symbol) and not f.endswith('.csv'):
                try:
                    os.remove(os.path.join('static', f))
                except:
                    pass
    except:
        pass


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main route"""
    current_date = dt.datetime.now().strftime('%Y-%m-%d')

    if request.method == 'POST':
        stock = request.form.get('stock', '').strip().upper()

        if not stock:
            return render_template('index.html', 
                                   error="Enter a stock symbol", 
                                   current_date=current_date)

        if not stock.isalnum() or len(stock) > 10:
            return render_template('index.html', 
                                   error="Invalid symbol", 
                                   current_date=current_date)

        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing: {stock}")
            logger.info(f"{'='*50}")

            # STEP 1: Fetch data
            logger.info("STEP 1: Fetching data...")
            df = download_stock_data(stock)

            if df.empty or len(df) < 50:
                return render_template('index.html', 
                                       error=f"Insufficient data", 
                                       current_date=current_date)

            # STEP 2: Calculate indicators
            logger.info("STEP 2: Calculating indicators...")
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['RSI'] = calculate_rsi(df['Close'])

            # STEP 3: Prepare data
            logger.info("STEP 3: Preparing data...")
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

            # STEP 4: Predictions
            logger.info("STEP 4: Predictions...")
            predictions = None
            if model:
                try:
                    predictions = model.predict(X, verbose=0)
                    predictions = scaler.inverse_transform(predictions)

                    from keras import backend as K
                    import gc
                    K.clear_session()
                    gc.collect()
                except Exception as e:
                    logger.warning(f"Prediction: {e}")

            safe_symbol = sanitize_filename(stock)
            cleanup_old_files(safe_symbol)

            # STEP 5: Charts (LOW DPI - faster)
            logger.info("STEP 5: Creating charts...")

            plt.figure(figsize=(10, 5))
            plt.plot(df.index, df['Close'], label='Close', linewidth=2, color='#1f77b4')
            plt.plot(df.index, df['EMA_20'], label='EMA 20', alpha=0.7, color='#ff7f0e')
            plt.plot(df.index, df['EMA_50'], label='EMA 50', alpha=0.7, color='#2ca02c')
            plt.title(f"{stock} Price", fontsize=12, fontweight='bold')
            plt.xlabel('Date', fontsize=10)
            plt.ylabel('Price (USD)', fontsize=10)
            plt.legend(loc='best', fontsize=9)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            ema_path = f"static/{safe_symbol}_ema.png"
            plt.savefig(ema_path, dpi=60)  # Ultra low DPI
            plt.close('all')

            plt.figure(figsize=(10, 5))
            plt.plot(df.index[lookback:], df['Close'][lookback:], 
                    label='Actual', linewidth=2, color='#1f77b4')
            if predictions is not None:
                plt.plot(df.index[lookback:], predictions, 
                        label='Predicted', alpha=0.7, linestyle='--', color='#d62728')
            plt.title(f"{stock} Prediction", fontsize=12, fontweight='bold')
            plt.xlabel('Date', fontsize=10)
            plt.ylabel('Price (USD)', fontsize=10)
            plt.legend(loc='best', fontsize=9)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            pred_path = f"static/{safe_symbol}_prediction.png"
            plt.savefig(pred_path, dpi=60)
            plt.close('all')

            # STEP 6: Metrics
            logger.info("STEP 6: Calculating metrics...")
            latest_close = float(df['Close'].iloc[-1])
            previous_close = float(df['Close'].iloc[-2])
            change = latest_close - previous_close
            change_pct = (change / previous_close) * 100

            # STEP 7: Save CSV
            csv_filename = f"{safe_symbol}_data.csv"
            df.to_csv(f"static/{csv_filename}")

            # STEP 8: News
            logger.info("STEP 8: Fetching news...")
            news_list = get_stock_news(stock)

            logger.info(f"✓ Success")
            logger.info(f"{'='*50}\n")

            return render_template('index.html',
                                   stock_symbol=stock,
                                   current_date=current_date,
                                   last_date=df.index[-1].strftime('%Y-%m-%d'),
                                   latest_close=round(latest_close, 2),
                                   change=round(change, 2),
                                   change_pct=round(change_pct, 2),
                                   high_52w=round(df['Close'].max(), 2),
                                   low_52w=round(df['Close'].min(), 2),
                                   avg_price=round(df['Close'].mean(), 2),
                                   ema_chart=ema_path,
                                   pred_chart=pred_path,
                                   data_desc=df.describe().to_html(classes='table table-striped'),
                                   csv_path=csv_filename,
                                   news_list=news_list,
                                   prediction_available=(predictions is not None))

        except Exception as e:
            logger.error(f"ERROR: {str(e)[:100]}")
            return render_template('index.html', 
                                   error="Unable to process. Try again.", 
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
        logger.error(f"Download: {e}")
        return "Error", 500


@app.route('/health')
def health_check():
    """Health check"""
    return {
        'status': 'healthy',
        'timestamp': dt.datetime.now().isoformat(),
        'model_loaded': model is not None,
        'environment': 'Render' if IS_RENDER else 'Local'
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
    logger.info(f"\n{'='*50}")
    logger.info("Starting Stock Prediction App")
    logger.info(f"Port: {port}")
    logger.info(f"Environment: {'Render' if IS_RENDER else 'Local'}")
    logger.info(f"Data Period: {DATA_PERIOD}")
    logger.info(f"YFinance Timeout: {YFINANCE_TIMEOUT}s")
    logger.info(f"Model: {'Loaded' if model else 'Not found'}")
    logger.info(f"{'='*50}\n")

    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)