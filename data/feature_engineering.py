import numpy as np
import pandas as pd
import os
import json
from datetime import datetime

def compute_features(df, symbol: str = None, force_refresh: bool = False) -> dict:
    """
    Compute features for a given ticker DataFrame.
    Returns a dict with RSI, MACD, z-score, Bollinger Bands, avg volume ratio.
    """
    if symbol is not None:
        # Create cache directory if it doesn't exist
        cache_dir = 'data/cache'
        os.makedirs(cache_dir, exist_ok=True)
        
        # Check for cached features
        cache_file = os.path.join(cache_dir, f'{symbol}_features.json')
        if not force_refresh and os.path.exists(cache_file):
            print(f"Loading cached features for {symbol}")
            with open(cache_file, 'r') as f:
                return json.load(f)
    
    features = {}
    close = df['Close']
    volume = df['Volume']
    # RSI
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    features['rsi'] = float(rsi.iloc[-1]) if not rsi.empty else np.nan
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    features['macd'] = float(macd.iloc[-1]) if not macd.empty else np.nan
    features['macd_signal'] = float(signal.iloc[-1]) if not signal.empty else np.nan
    # z-score of returns
    returns = close.pct_change().dropna()
    zscore = (returns.iloc[-1] - returns.mean()) / (returns.std() + 1e-9) if not returns.empty else np.nan
    features['zscore'] = float(zscore)
    # Bollinger Bands
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    features['bb_upper'] = float((ma20 + 2 * std20).iloc[-1]) if not ma20.empty else np.nan
    features['bb_lower'] = float((ma20 - 2 * std20).iloc[-1]) if not ma20.empty else np.nan
    features['bb_ma'] = float(ma20.iloc[-1]) if not ma20.empty else np.nan
    # Avg volume ratio
    # Avg volume ratio
    # Avg volume ratio
    avg_vol_5 = volume.rolling(5).mean().iloc[-1] if len(volume) >= 5 else np.nan
    avg_vol_30 = volume.rolling(30).mean().iloc[-1] if len(volume) >= 30 else np.nan

    # Ensure scalar before checking np.isnan
    try:
        avg_vol_5_val = float(avg_vol_5)
        avg_vol_30_val = float(avg_vol_30)
    except (ValueError, TypeError):
        avg_vol_5_val = avg_vol_30_val = np.nan

    if not np.isnan(avg_vol_5_val) and not np.isnan(avg_vol_30_val):
        features['avg_vol_ratio'] = avg_vol_5_val / (avg_vol_30_val + 1e-9)
    else:
        features['avg_vol_ratio'] = np.nan
    # features['avg_vol_ratio'] = float(avg_vol_5 / (avg_vol_30 + 1e-9)) if avg_vol_30 and not np.isnan(avg_vol_5) else np.nan
    
    # Cache features if symbol provided
    if symbol is not None:
        with open(cache_file, 'w') as f:
            json.dump(features, f)
        print(f"Cached features for {symbol} to {cache_file}")
    
    return features 