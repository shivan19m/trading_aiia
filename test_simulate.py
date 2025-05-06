from data.market_data import load_market_data
from agents.strategy.momentum_agent import MomentumAgent
from simulation.simulate import simulate_portfolio

# Parameters
tickers = ["AAPL", "MSFT", "GOOG"]
start_date = "2023-01-01"
end_date = "2023-03-31"
lookback_days = 35

# Load data for all tickers
all_data = {}
for ticker in tickers:
    print(f"Loading data for {ticker}...")
    df = load_market_data(ticker, start_date, end_date, lookback_days=lookback_days, interval='1d')
    if df is None or df.empty:
        print(f"Failed to load data for {ticker}.")
        exit(1)
    all_data[ticker] = df

# Instantiate agent
agent = MomentumAgent()
agent.set_tickers(tickers)

# Run simulation
print("Running simulation...")
results = simulate_portfolio(
    tickers=tickers,
    all_data=all_data,
    agent=agent,
    start_date=start_date,
    end_date=end_date,
    window=30,
    step=15,
    initial_cash=100_000
)

# Print results
print("\nMetrics:", results['metrics'])
print("\nPortfolio Value History (head):")
print(results['portfolio_history'].head())
print("\nPlan History (last 2):")
print(results['plan_history'][-2:]) 