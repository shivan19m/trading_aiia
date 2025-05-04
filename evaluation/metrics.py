import numpy as np
import pandas as pd

def total_return(portfolio_value_history, initial_cash):
    final_value = portfolio_value_history[-1]['value']
    return (final_value - initial_cash) / initial_cash

def daily_returns(portfolio_value_history):
    values = [x['value'] for x in portfolio_value_history]
    returns = np.diff(values) / values[:-1]
    return returns

def sharpe_ratio(portfolio_value_history, initial_cash):
    rets = daily_returns(portfolio_value_history)
    if len(rets) == 0 or np.std(rets) == 0:
        return 0.0
    return np.mean(rets) / np.std(rets) * np.sqrt(252)

def max_drawdown(portfolio_value_history):
    values = np.array([x['value'] for x in portfolio_value_history])
    running_max = np.maximum.accumulate(values)
    drawdowns = (values - running_max) / running_max
    return np.min(drawdowns)

def hit_rate(portfolio_value_history):
    rets = daily_returns(portfolio_value_history)
    return np.mean(rets > 0) 