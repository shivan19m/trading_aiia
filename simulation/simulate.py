import pandas as pd
import numpy as np
from typing import List, Dict, Any
from pipeline import TradingPipeline

def simulate_portfolio(
    tickers: List[str],
    all_data: Dict[str, pd.DataFrame],  # Not used, kept for interface compatibility
    agent,
    start_date: str,
    end_date: str,
    window: int = 30,
    step: int = 15,
    initial_cash: float = 100_000,
    metrics: List[str] = None
) -> Dict[str, Any]:
    """
    Simulate rolling-window portfolio execution using TradingPipeline for robust feature engineering and plan generation.
    Returns a dict with portfolio value history, trades, and metrics.
    """
    if metrics is None:
        metrics = ['total_return', 'max_drawdown', 'sharpe', 'rachev']

    # Prepare time windows
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    windows = []
    i = 0
    while i + window <= len(dates):
        windows.append((dates[i], dates[i+window-1]))
        i += step

    # Initialize pipeline with the specified tickers
    pipeline = TradingPipeline(tickers, lookback_days=35, interval='1d')
    agent.set_tickers(tickers)

    # Initialize portfolio
    cash = initial_cash
    holdings = {ticker: 0 for ticker in tickers}
    portfolio_history = []
    plan_history = []

    for win_start, win_end in windows:
        window_start_str = str(win_start.date())
        window_end_str = str(win_end.date())
        print(f"\n[SIM] Window {window_start_str} to {window_end_str} | Tickers: {tickers}")
        # 1. Use pipeline to load data and compute features for this window
        market_data_dict = pipeline.load_data(window_start_str, window_end_str)
        features_dict = {}
        for ticker in tickers:
            if ticker.lower() == 'cash':
                continue
            features = pipeline._get_features_for_ticker(ticker, window_start_str, window_end_str)
            features_dict[ticker] = features
        print(f"[SIM] Features: {features_dict}")
        # 2. Agent proposes a plan (allocations), portfolio-aware
        plan = agent.propose_plan(
            features_dict,
            context={},
            current_holdings=holdings.copy(),
            cash=cash,
            portfolio_history=portfolio_history.copy()
        )
        print(f"[SIM] Plan: {plan}")
        plan_history.append({'start': win_start, 'end': win_end, 'plan': plan})

        # 3. Simulate execution: rebalance portfolio to match plan
        total_value = cash + sum(
            holdings[ticker] * market_data_dict[ticker]['close'].iloc[-1]
            for ticker in tickers if ticker.lower() != 'cash' and not market_data_dict[ticker].empty
        )
        new_holdings = {}
        spent_cash = 0
        for ticker in tickers:
            if ticker.lower() == 'cash':
                continue
            if market_data_dict[ticker].empty:
                print(f"[WARN] No data for {ticker} in window {window_start_str} to {window_end_str}, skipping.")
                new_holdings[ticker] = 0
                continue
            weight = plan.get(ticker, {}).get('weight', 0)
            alloc_value = total_value * weight
            price = market_data_dict[ticker]['close'].iloc[-1]
            max_affordable_shares = int((cash + sum(holdings[t]*price for t in tickers if t!=ticker and t.lower()!='cash')) // price)
            shares = int(min(alloc_value // price, max_affordable_shares))
            new_holdings[ticker] = shares
            spent_cash += shares * price
        cash = total_value - spent_cash
        holdings = new_holdings.copy()
        if cash < 0:
            print(f"[WARN] Negative cash after rebalancing: {cash}. Setting to 0.")
            cash = 0
        value = cash + sum(
            holdings[ticker] * market_data_dict[ticker]['close'].iloc[-1]
            for ticker in tickers if ticker.lower() != 'cash' and not market_data_dict[ticker].empty
        )
        print(f"[SIM] Holdings: {holdings}, Cash: {cash}, Portfolio Value: {value}")
        portfolio_history.append({
            'date': win_end,
            'value': value,
            'cash': cash,
            'holdings': holdings.copy(),
            'plan': plan
        })
        # Warn if features are missing or fallback is used
        if not features_dict or any(not v for v in features_dict.values()):
            print(f"[WARN] Features missing or empty for window {window_start_str} to {window_end_str}")
        if any(isinstance(v, dict) and 'even distribution' in (v.get('reason', '').lower()) for v in plan.values()):
            print(f"[WARN] Agent fallback plan used in window {window_start_str} to {window_end_str}")

    # Convert to DataFrame for metrics
    pf_df = pd.DataFrame(portfolio_history).set_index('date')
    returns = pf_df['value'].pct_change().dropna()

    # Metrics
    results = {
        'portfolio_history': pf_df,
        'plan_history': plan_history,
        'metrics': {}
    }
    if 'total_return' in metrics:
        results['metrics']['total_return'] = pf_df['value'].iloc[-1] / pf_df['value'].iloc[0] - 1
    if 'max_drawdown' in metrics:
        peak = pf_df['value'].cummax()
        drawdown = (pf_df['value'] - peak) / peak
        results['metrics']['max_drawdown'] = drawdown.min()
    if 'sharpe' in metrics:
        results['metrics']['sharpe'] = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
    if 'rachev' in metrics:
        right = returns[returns > returns.quantile(0.95)].mean()
        left = abs(returns[returns < returns.quantile(0.05)].mean())
        results['metrics']['rachev'] = right / left if left != 0 else np.nan

    return results 

# --- Real-time generator version ---
def simulate_portfolio_realtime(
    tickers: List[str],
    all_data: Dict[str, pd.DataFrame],
    agent,
    start_date: str,
    end_date: str,
    window: int = 30,
    step: int = 15,
    initial_cash: float = 100_000
):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    windows = []
    i = 0
    while i + window <= len(dates):
        windows.append((dates[i], dates[i+window-1]))
        i += step

    pipeline = TradingPipeline(tickers, lookback_days=35, interval='1d')
    agent.set_tickers(tickers)
    cash = initial_cash
    holdings = {ticker: 0 for ticker in tickers}
    portfolio_history = []
    plan_history = []
    step_summaries = []

    for win_start, win_end in windows:
        window_start_str = str(win_start.date())
        window_end_str = str(win_end.date())
        market_data_dict = pipeline.load_data(window_start_str, window_end_str)
        features_dict = {}
        for ticker in tickers:
            if ticker.lower() == 'cash':
                continue
            features = pipeline._get_features_for_ticker(ticker, window_start_str, window_end_str)
            features_dict[ticker] = features
        plan = agent.propose_plan(
            features_dict,
            context={},
            current_holdings=holdings.copy(),
            cash=cash,
            portfolio_history=portfolio_history.copy()
        )
        plan_history.append({'start': win_start, 'end': win_end, 'plan': plan})
        total_value = cash + sum(
            holdings[ticker] * market_data_dict[ticker]['close'].iloc[-1]
            for ticker in tickers if ticker.lower() != 'cash' and not market_data_dict[ticker].empty
        )
        new_holdings = {}
        spent_cash = 0
        trades = {}
        for ticker in tickers:
            if ticker.lower() == 'cash':
                continue
            if market_data_dict[ticker].empty:
                new_holdings[ticker] = 0
                continue
            weight = plan.get(ticker, {}).get('weight', 0)
            alloc_value = total_value * weight
            price = market_data_dict[ticker]['close'].iloc[-1]
            max_affordable_shares = int((cash + sum(holdings[t]*price for t in tickers if t!=ticker and t.lower()!='cash')) // price)
            shares = int(min(alloc_value // price, max_affordable_shares))
            if shares != holdings[ticker]:
                trades[ticker] = {
                    'old_shares': holdings[ticker],
                    'new_shares': shares,
                    'price': price,
                    'value': abs(shares - holdings[ticker]) * price
                }
            new_holdings[ticker] = shares
            spent_cash += shares * price
        cash = total_value - spent_cash
        if cash < 0:
            cash = 0
        holdings = new_holdings.copy()
        final_position_values = {}
        for ticker in tickers:
            if ticker.lower() == 'cash':
                final_position_values[ticker] = cash
                continue
            if market_data_dict[ticker].empty:
                final_position_values[ticker] = 0
                continue
            price = market_data_dict[ticker]['close'].iloc[-1]
            final_position_values[ticker] = holdings[ticker] * price
        value = sum(final_position_values.values())
        allocation = {k: v/value for k, v in final_position_values.items() if value > 0}
        step_summary = {
            'date': str(win_end.date()),
            'window': {'start': window_start_str, 'end': window_end_str},
            'portfolio': {
                'total_value': float(value),
                'cash': float(cash),
                'holdings': {k: int(v) for k, v in holdings.items()},
                'position_values': final_position_values,
                'allocation': allocation
            },
            'trades': trades,
            'plan': plan,
            'market_data': {
                ticker: {
                    'price': float(df['close'].iloc[-1]),
                    'volume': float(df['volume'].iloc[-1]),
                    'vwap': float(df['vwap'].iloc[-1]) if 'vwap' in df.columns else None
                }
                for ticker, df in market_data_dict.items()
            }
        }
        if len(portfolio_history) > 0:
            prev_value = portfolio_history[-1]['value']
            daily_return = (value - prev_value) / prev_value
            step_summary['metrics'] = {
                'daily_return': float(daily_return),
                'cumulative_return': float(value / initial_cash - 1)
            }
        step_summaries.append(step_summary)
        portfolio_history.append({
            'date': str(win_end.date()),
            'value': float(value),
            'cash': float(cash),
            'holdings': {k: int(v) for k, v in holdings.items()},
            'plan': plan
        })
        yield step_summary, list(portfolio_history), list(plan_history), list(step_summaries) 