import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')

BASE_URL = "https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"

INTERVAL_MAP = {
    '1d': (1, 'day'),
    '1h': (1, 'hour'),
    '1m': (1, 'minute'),
}

def load_market_data(symbol: str, start_date: str, end_date: str, lookback_days: int = 30, interval: str = '1d', force_refresh: bool = False) -> pd.DataFrame:
    """
    Load historical price data for the given stock symbol using Polygon.io.
    Downloads data from (start_date - lookback_days) to end_date.
    Returns a DataFrame indexed by datetime.
    """
    cache_dir = 'data/cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'{symbol}_{start_date}_{end_date}_{interval}.csv')
    if not force_refresh and os.path.exists(cache_file):
        print(f"Loading cached OHLCV data for {symbol} from {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df

    if POLYGON_API_KEY is None:
        raise ValueError("POLYGON_API_KEY not set in environment.")

    # Compute lookback start
    start_dt = pd.to_datetime(start_date) - pd.Timedelta(days=lookback_days)
    end_dt = pd.to_datetime(end_date)
    from_date = start_dt.strftime('%Y-%m-%d')
    to_date = end_dt.strftime('%Y-%m-%d')

    # Map interval
    if interval not in INTERVAL_MAP:
        raise ValueError(f"Unsupported interval: {interval}")
    multiplier, timespan = INTERVAL_MAP[interval]

    url = BASE_URL.format(
        symbol=symbol.upper(),
        multiplier=multiplier,
        timespan=timespan,
        from_date=from_date,
        to_date=to_date
    )
    url += f"?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}"

    print(f"Fetching Polygon data: {url}")
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"Polygon API error: {resp.status_code} {resp.text}")
        return pd.DataFrame()
    data = resp.json()
    if 'results' not in data:
        print(f"No results in Polygon response: {data}")
        return pd.DataFrame()
    results = data['results']
    if not results:
        print(f"No data returned from Polygon for {symbol}")
        return pd.DataFrame()

    # Build DataFrame
    df = pd.DataFrame(results)
    # Polygon returns timestamps in ms since epoch
    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.rename(columns={
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'c': 'close',
        'v': 'volume',
        'n': 'transactions'
    })
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df.to_csv(cache_file)
    print(f"Cached OHLCV data for {symbol} to {cache_file}")
    return df

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