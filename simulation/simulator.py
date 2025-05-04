import yfinance as yf
import pandas as pd
from data.feature_engineering import compute_features
from tqdm import tqdm

# TODO: Import agents and memory as needed

def simulate(start_date, end_date, initial_cash, tickers, agents, frequency="1d"):
    """
    Simulate agent portfolio allocation and execution over time.
    - start_date, end_date: simulation window
    - initial_cash: starting capital
    - tickers: list of tickers
    - agents: list of instantiated strategy agents
    - frequency: rebalancing frequency (default: daily)
    Returns: portfolio value history, holdings, agent plans
    """
    dates = pd.date_range(start=start_date, end=end_date, freq=frequency.upper())
    cash_balance = initial_cash
    holdings = {symbol: 0 for symbol in tickers}
    portfolio_value_history = []
    plan_history = []
    prev_allocations = None
    for current_date in tqdm(dates, desc="Simulating portfolio"):
        # Load price data up to current date for all tickers
        price_data = {}
        for symbol in tickers:
            df = yf.download(symbol, end=current_date + pd.Timedelta(days=1), period="90d", interval="1d", progress=False)
            if df.empty or len(df) < 20:
                continue
            price_data[symbol] = df
        # Compute features for each ticker
        features = {symbol: compute_features(df) for symbol, df in price_data.items()}
        # Pass features to each agent, get portfolio allocation plans
        agent_plans = []
        for agent in agents:
            plan = agent.propose_plan(features, context={})
            agent_plans.append(plan)
        # TODO: MetaPlanner/ensemble logic; for now, use first agent's plan
        selected_plan = agent_plans[0] if agent_plans else None
        plan_history.append({'date': str(current_date), 'plan': selected_plan})
        # Rebalancing logic
        allocations = selected_plan['allocations'] if selected_plan else []
        if prev_allocations != allocations:
            # Sell all holdings
            for symbol in holdings:
                if holdings[symbol] > 0 and symbol in price_data:
                    cash_balance += holdings[symbol] * price_data[symbol]['Close'].iloc[-1]
                    holdings[symbol] = 0
            # Allocate to new plan
            for alloc in allocations:
                symbol = alloc['symbol']
                weight = alloc['weight']
                if symbol == 'cash':
                    continue
                if symbol in price_data:
                    alloc_cash = cash_balance * weight
                    price = price_data[symbol]['Close'].iloc[-1]
                    shares = int(alloc_cash // price)
                    holdings[symbol] += shares
                    cash_balance -= shares * price
        prev_allocations = allocations
        # Compute portfolio value
        value = cash_balance
        for symbol in holdings:
            if holdings[symbol] > 0 and symbol in price_data:
                value += holdings[symbol] * price_data[symbol]['Close'].iloc[-1]
        portfolio_value_history.append({'date': str(current_date), 'value': value, 'cash': cash_balance, 'holdings': holdings.copy()})
    # TODO: Add benchmark comparison (SPY)
    # TODO: Use vector memory in simulation loop
    # TODO: Dynamic agent role reassignment (ALAS extension)
    return portfolio_value_history, plan_history 