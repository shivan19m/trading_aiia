# import numpy as np
# import pandas as pd
# import os
# import json
# from datetime import datetime

# def compute_features(df, symbol: str = None, force_refresh: bool = False) -> dict:
#     """
#     Compute features for a given ticker DataFrame.
#     Returns a dict with RSI, MACD, z-score, Bollinger Bands, avg volume ratio, typical price, dollar volume, prev day's data, dividends, splits, pre/post-market prices if available.
#     """
#     if symbol is not None:
#         # Create cache directory if it doesn't exist
#         cache_dir = 'data/cache'
#         os.makedirs(cache_dir, exist_ok=True)
        
#         # Check for cached features
#         cache_file = os.path.join(cache_dir, f'{symbol}_features.json')
#         if not force_refresh and os.path.exists(cache_file):
#             print(f"Loading cached features for {symbol}")
#             with open(cache_file, 'r') as f:
#                 return json.load(f)
    
#     features = {}
#     # Use lower-case column names for compatibility
#     df = df.copy()
#     df.columns = [c.lower() for c in df.columns]
#     close = df['close']
#     volume = df['volume']
#     high = df['high']
#     low = df['low']
#     open_ = df['open']
#     # Typical price
#     typical_price = (high + low + close) / 3
#     features['typical_price'] = float(typical_price.iloc[-1]) if not typical_price.empty else np.nan
#     # Dollar volume
#     dollar_volume = close * volume
#     features['dollar_volume'] = float(dollar_volume.iloc[-1]) if not dollar_volume.empty else np.nan
#     # Previous day's close, high, low
#     features['prev_close'] = float(close.shift(1).iloc[-1]) if len(close) > 1 else np.nan
#     features['prev_high'] = float(high.shift(1).iloc[-1]) if len(high) > 1 else np.nan
#     features['prev_low'] = float(low.shift(1).iloc[-1]) if len(low) > 1 else np.nan
#     # RSI
#     delta = close.diff()
#     up = delta.clip(lower=0)
#     down = -1 * delta.clip(upper=0)
#     roll_up = up.rolling(14).mean()
#     roll_down = down.rolling(14).mean()
#     rs = roll_up / (roll_down + 1e-9)
#     rsi = 100 - (100 / (1 + rs))
#     features['rsi'] = float(rsi.iloc[-1]) if not rsi.empty else np.nan
#     # MACD
#     ema12 = close.ewm(span=12, adjust=False).mean()
#     ema26 = close.ewm(span=26, adjust=False).mean()
#     macd = ema12 - ema26
#     signal = macd.ewm(span=9, adjust=False).mean()
#     features['macd'] = float(macd.iloc[-1]) if not macd.empty else np.nan
#     features['macd_signal'] = float(signal.iloc[-1]) if not signal.empty else np.nan
#     # z-score of returns
#     returns = close.pct_change().dropna()
#     zscore = (returns.iloc[-1] - returns.mean()) / (returns.std() + 1e-9) if not returns.empty else np.nan
#     features['zscore'] = float(zscore)
#     # Bollinger Bands
#     ma20 = close.rolling(20).mean()
#     std20 = close.rolling(20).std()
#     features['bb_upper'] = float((ma20 + 2 * std20).iloc[-1]) if not ma20.empty else np.nan
#     features['bb_lower'] = float((ma20 - 2 * std20).iloc[-1]) if not ma20.empty else np.nan
#     features['bb_ma'] = float(ma20.iloc[-1]) if not ma20.empty else np.nan
#     # Avg volume ratio
#     avg_vol_5 = volume.rolling(5).mean().iloc[-1] if len(volume) >= 5 else np.nan
#     avg_vol_30 = volume.rolling(30).mean().iloc[-1] if len(volume) >= 30 else np.nan
#     try:
#         avg_vol_5_val = float(avg_vol_5)
#         avg_vol_30_val = float(avg_vol_30)
#     except (ValueError, TypeError):
#         avg_vol_5_val = avg_vol_30_val = np.nan
#     if not np.isnan(avg_vol_5_val) and not np.isnan(avg_vol_30_val):
#         features['avg_vol_ratio'] = avg_vol_5_val / (avg_vol_30_val + 1e-9)
#     else:
#         features['avg_vol_ratio'] = np.nan
#     # VWAP and trades (from enhanced loader)
#     if 'vwap' in df.columns:
#         features['vwap'] = float(df['vwap'].iloc[-1]) if not df['vwap'].empty else np.nan
#     if 'trades' in df.columns:
#         features['trades'] = float(df['trades'].iloc[-1]) if not df['trades'].empty else np.nan
#     # Pre/post-market prices (from open-close endpoint, if merged in loader)
#     if 'pre_market' in df.columns:
#         features['pre_market'] = float(df['pre_market'].iloc[-1]) if not df['pre_market'].empty else np.nan
#     if 'after_market' in df.columns:
#         features['after_market'] = float(df['after_market'].iloc[-1]) if not df['after_market'].empty else np.nan
#     # Dividends and splits (if merged in loader)
#     if 'dividend' in df.columns:
#         features['dividend'] = float(df['dividend'].iloc[-1]) if not df['dividend'].empty else np.nan
#     if 'split' in df.columns:
#         features['split'] = float(df['split'].iloc[-1]) if not df['split'].empty else np.nan
#     # Cache features if symbol provided
#     if symbol is not None:
#         with open(cache_file, 'w') as f:
#             json.dump(features, f)
#         print(f"Cached features for {symbol} to {cache_file}")
#     return features 

import numpy as np
import pandas as pd
import os
import json
import random  # UPDATED: imported to generate event-driven features
from datetime import datetime

def compute_features(
    data, 
    symbol: str = None, 
    force_refresh: bool = False, 
    momentum_window: int = 5,           # UPDATED: accepted hyperparameter
    meanrev_z_thresh: float = 1.0,      # UPDATED: accepted hyperparameter
    event_weight_cap: float = 0.2       # UPDATED: accepted hyperparameter
) -> dict:
    """
    Compute features for one ticker or a dict of tickers.

    If `data` is a dict of {symbol: DataFrame}, returns a dict of feature‐dicts
    for each symbol. Otherwise, computes for a single DataFrame.

    Supports:
      - momentum (window = momentum_window)
      - RSI, MACD, z-score, Bollinger Bands, avg volume ratio, typical price, dollar volume
      - previous day's OHLC
      - optional VWAP, trades, pre/post-market, dividends, splits
      - mean-reversion signal flag (abs(zscore) > meanrev_z_thresh)
      - event-driven placeholders: news_sentiment, earnings_surprise, macro_score
    """

    # 1) If passed a dict of DataFrames, recurse per symbol
    if isinstance(data, dict):
        all_feats = {}
        for sym, df in data.items():
            all_feats[sym] = compute_features(
                df,
                symbol=sym,
                force_refresh=force_refresh,
                momentum_window=momentum_window,
                meanrev_z_thresh=meanrev_z_thresh,
                event_weight_cap=event_weight_cap
            )
        return all_feats

    df = data.copy()
    df.columns = [c.lower() for c in df.columns]

    # 2) Caching logic per-symbol
    if symbol is not None:
        cache_dir = 'data/cache'
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f'{symbol}_features.json')
        if not force_refresh and os.path.exists(cache_file):
            print(f"Loading cached features for {symbol}")
            with open(cache_file, 'r') as f:
                return json.load(f)

    features = {}
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    open_ = df['open']

    # Typical price and dollar volume
    features['typical_price'] = float(((high + low + close) / 3).iloc[-1])
    features['dollar_volume']  = float((close * volume).iloc[-1])

    # Previous day's OHLC
    features['prev_close'] = float(close.shift(1).iloc[-1])
    features['prev_high']  = float(high.shift(1).iloc[-1])
    features['prev_low']   = float(low.shift(1).iloc[-1])

    # RSI (14-period)
    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = up / (down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    features['rsi'] = float(rsi.iloc[-1])

    # MACD & signal (12/26/9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    features['macd']        = float(macd.iloc[-1])
    features['macd_signal'] = float(signal.iloc[-1])

    # Momentum over custom window
    # UPDATED: use `momentum_window` hyperparam
    if len(close) >= momentum_window:
        features['momentum'] = float(close.pct_change(periods=momentum_window).iloc[-1])
    else:
        features['momentum'] = np.nan

    # Returns z-score
    returns = close.pct_change().dropna()
    z = (returns.iloc[-1] - returns.mean()) / (returns.std() + 1e-9)
    features['zscore'] = float(z)

    # Bollinger Bands (20-period)
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    features['bb_upper'] = float((ma20 + 2*std20).iloc[-1])
    features['bb_lower'] = float((ma20 - 2*std20).iloc[-1])
    features['bb_ma']    = float(ma20.iloc[-1])

    # Average volume ratio (5 vs 30)
    vol5  = volume.rolling(5).mean().iloc[-1]
    vol30 = volume.rolling(30).mean().iloc[-1]
    features['avg_vol_ratio'] = float(vol5 / (vol30 + 1e-9)) if vol30 != 0 else np.nan

    # Optional VWAP, trades, pre/post-market, dividends, splits
    for col in ['vwap','trades','pre_market','after_market','dividend','split']:
        if col in df.columns:
            features[col] = float(df[col].iloc[-1])

    # Mean-reversion signal flag
    # UPDATED: indicate if abs(zscore) exceeds threshold
    features['meanrev_signal'] = abs(features['zscore']) > meanrev_z_thresh

    # Event-driven placeholder features
    # UPDATED: generate random event-based signals
    features['news_sentiment']    = random.uniform(-1, 1)
    features['earnings_surprise'] = random.uniform(-0.2, 0.2)
    features['macro_score']       = random.choice([0.0, 0.1, -0.1])
    # Store cap parameter so downstream can enforce event weight cap
    features['event_weight_cap']  = float(event_weight_cap)

    # 3) Cache to disk
    if symbol is not None:
        with open(cache_file, 'w') as f:
            json.dump(features, f)
        print(f"Cached features for {symbol}")

    return features