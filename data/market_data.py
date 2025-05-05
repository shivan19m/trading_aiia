import yfinance as yf
from datetime import datetime, timedelta
import os
import pandas as pd
def load_market_data(symbol: str, force_refresh: bool = False) -> dict:
    """
    Load 7-day hourly price data for the given stock symbol using yfinance.
    Returns a dictionary with timestamps and OHLCV data.
    """
    cache_dir = 'data/cache'
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = os.path.join(cache_dir, f'{symbol}_ohlcv.csv')
    if not force_refresh and os.path.exists(cache_file):
        print(f"Loading cached OHLCV data for {symbol}")
        df = pd.read_csv(cache_file, index_col=0, skiprows=2)
        df.index = pd.to_datetime(df.index)
    else:
        end = datetime.now()
        start = end - timedelta(days=7)
        df = yf.download(symbol, start=start, end=end, interval='1h', auto_adjust=False)
        if df.empty:
            return {}
        df.to_csv(cache_file)
        print(f"Cached OHLCV data for {symbol} to {cache_file}")
    
    # Flatten multi-index columns (if any), and lowercase column names
    df.columns = [
        col[0].lower() if isinstance(col, tuple) else col.lower()
        for col in df.columns
    ]

    data = {
        'symbol': symbol,
        'timestamps': df.index.strftime('%Y-%m-%d %H:%M').tolist(),
        'open': df['open'].tolist() if 'open' in df else [],
        'high': df['high'].tolist() if 'high' in df else [],
        'low': df['low'].tolist() if 'low' in df else [],
        'close': df['close'].tolist() if 'close' in df else [],
        'volume': df['volume'].tolist() if 'volume' in df else [],
    }
    return data

# def load_market_data(symbol: str, force_refresh: bool = False) -> dict:
#     """
#     Load 7-day hourly price data for the given stock symbol using yfinance.
#     Returns a dictionary with timestamps and OHLCV data.
#     """
#     # Create cache directory if it doesn't exist
#     cache_dir = 'data/cache'
#     os.makedirs(cache_dir, exist_ok=True)
    
#     # Check for cached data
#     cache_file = os.path.join(cache_dir, f'{symbol}_ohlcv.csv')
#     if not force_refresh and os.path.exists(cache_file):
#         print(f"Loading cached OHLCV data for {symbol}")
#         df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
#     else:
#         end = datetime.now()
#         start = end - timedelta(days=7)
#         df = yf.download(symbol, start=start, end=end, interval='1h', auto_adjust=False)
#         if df.empty:
#             return {}
#         # Save to cache
#         df.to_csv(cache_file)
#         print(f"Cached OHLCV data for {symbol} to {cache_file}")
    
#     # Flatten columns if MultiIndex, then lowercase
#     df.columns = [
#         col[0].lower() if isinstance(col, tuple) else col.lower()
#         for col in df.columns
#     ]
    
#     data = {
#         'symbol': symbol,
#         'timestamps': df.index.strftime('%Y-%m-%d %H:%M').tolist(),
#         'open': df['open'].tolist() if 'open' in df else [],
#         'high': df['high'].tolist() if 'high' in df else [],
#         'low': df['low'].tolist() if 'low' in df else [],
#         'close': df['close'].tolist() if 'close' in df else [],
#         'volume': df['volume'].tolist() if 'volume' in df else [],
#     }
#     return data 