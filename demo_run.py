from data.ticker_selector import select_top_tickers
from agents.strategy.momentum_agent import MomentumAgent
from agents.strategy.mean_reversion_agent import MeanReversionAgent
from agents.strategy.event_driven_agent import EventDrivenAgent
from agents.meta_planner import MetaPlannerAgent
from simulation.simulator import simulate
from evaluation.metrics import total_return, sharpe_ratio, max_drawdown, hit_rate
from memory.memory_agent import MemoryAgent
from visualize.plots import plot_portfolio_vs_spy, plot_drawdown_curve, plot_final_allocation
from report.export import save_csvs
import yfinance as yf
import os

START_DATE = "2024-01-01"
END_DATE = "2024-01-07"
CAPITAL = 100_000
N_TICKERS = 5

print("Selecting top tickers...")
tickers = select_top_tickers(N_TICKERS)
print(f"Selected tickers: {tickers}")

# Initialize agents
agents = [MomentumAgent(), MeanReversionAgent(), EventDrivenAgent()]
meta_planner = MetaPlannerAgent()
memory_agent = MemoryAgent()

# Set tickers for all agents
for agent in agents:
    agent.set_tickers(tickers)
meta_planner.set_tickers(tickers)

print("Running demo simulation (ensemble mode)...")
# Custom ensemble simulation loop to use vector memory
from data.feature_engineering import compute_features
import pandas as pd
from tqdm import tqdm

def simulate_with_ensemble_memory(start_date, end_date, initial_cash, tickers, agents, memory_agent, frequency="1d"):
    dates = pd.date_range(start=start_date, end=end_date, freq=frequency.upper())
    cash_balance = initial_cash
    holdings = {symbol: 0 for symbol in tickers}
    portfolio_value_history = []
    plan_history = []
    prev_allocations = None
    for current_date in tqdm(dates, desc="Simulating portfolio"):
        price_data = {}
        for symbol in tickers:
            df = yf.download(symbol, end=current_date + pd.Timedelta(days=1), period="90d", interval="1d", progress=False)
            if df.empty or len(df) < 20:
                continue
            price_data[symbol] = df
        features = {symbol: compute_features(df) for symbol, df in price_data.items()}
        context = f"{current_date.date()} features: " + ", ".join([f"{k}: {v}" for k, v in features.items()])
        agent_plans = []
        for agent in agents:
            plan = agent.propose_plan(features, context, memory_agent=memory_agent)
            agent_plans.append(plan)
        # Ensemble: select best plan (highest average weight on non-cash assets)
        def plan_score(plan):
            return sum(v['weight'] for k, v in plan.items() if k != 'cash')
        selected_plan = max(agent_plans, key=plan_score)
        plan_history.append({'date': str(current_date), 'plans': agent_plans, 'selected': selected_plan})
        allocations = selected_plan
        if prev_allocations != allocations:
            for symbol in holdings:
                if holdings[symbol] > 0 and symbol in price_data:
                    cash_balance += holdings[symbol] * price_data[symbol]['Close'].iloc[-1]
                    holdings[symbol] = 0
            for symbol, alloc in allocations.items():
                if symbol == 'cash':
                    continue
                if symbol in price_data:
                    alloc_cash = cash_balance * alloc['weight']
                    # Ensure price is a scalar float
                    # 安全提取价格（假设 price_data 是 dict of DataFrame）
                    if symbol not in price_data or 'Close' not in price_data[symbol]:
                        print(f"[WARN] No price data for {symbol}, skipping...")
                        continue

                    price_series = price_data[symbol]['Close']
                    if price_series.empty:
                        print(f"[WARN] Empty price series for {symbol}, skipping...")
                        continue

                    price = float(price_series.iloc[-1])  # ✅ 保证是标量
                    shares = int(alloc_cash // price)
                    
                    holdings[symbol] += shares
                    cash_balance -= shares * price
        prev_allocations = allocations
        value = cash_balance
        for symbol in holdings:
            if holdings[symbol] > 0 and symbol in price_data:
                value += holdings[symbol] * price_data[symbol]['Close'].iloc[-1]
        portfolio_value_history.append({'date': str(current_date), 'value': value, 'cash': cash_balance, 'holdings': holdings.copy()})
        print(f"{current_date.date()} | Value: ${value:,.2f} | Allocations: {allocations}")
    return portfolio_value_history, plan_history

portfolio_value_history, plan_history = simulate_with_ensemble_memory(
    start_date=START_DATE,
    end_date=END_DATE,
    initial_cash=CAPITAL,
    tickers=tickers,
    agents=agents,
    memory_agent=memory_agent,
    frequency="1d"
)

final_value = portfolio_value_history[-1]['value']
tr = total_return(portfolio_value_history, CAPITAL)
sr = sharpe_ratio(portfolio_value_history, CAPITAL)
mdd = max_drawdown(portfolio_value_history)
hr = hit_rate(portfolio_value_history)

spy_df = yf.download('SPY', start=START_DATE, end=END_DATE, interval='1d', progress=False)
spy_values = []
spy_final_value = None
spy_return = None
if not spy_df.empty:
    start_price = spy_df['Close'].iloc[0]
    for i, row in spy_df.iterrows():
        value = CAPITAL * (row['Close'] / start_price)
        spy_values.append({'date': str(i.date()), 'value': value})
    spy_final_value = spy_values[-1]['value']
    spy_return = (spy_final_value - CAPITAL) / CAPITAL

def print_summary():
    print("\n=== Demo Simulation Results ===")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Total Return: {tr*100:.2f}%")
    print(f"Sharpe Ratio: {sr:.2f}")
    print(f"Max Drawdown: {mdd*100:.2f}%")
    print(f"Hit Rate: {hr*100:.2f}%")
    if spy_final_value is not None:
        # Handle Series objects
        spy_final = spy_final_value.iloc[-1] if hasattr(spy_final_value, "iloc") else spy_final_value
        print(f"SPY Benchmark: Final Value: ${spy_final:,.2f} | Return: {spy_return*100:.2f}%")
        print(f"Excess Return vs SPY: {(tr - spy_return)*100:.2f}%")
    else:
        print("SPY Benchmark data unavailable.")

print_summary()

out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)
plot_portfolio_vs_spy(portfolio_value_history, spy_values, save_path=os.path.join(out_dir, "portfolio_vs_spy.png"))
plot_drawdown_curve(portfolio_value_history, save_path=os.path.join(out_dir, "drawdown_curve.png"))
final_allocations = plan_history[-1]['selected']
plot_final_allocation([
    {"symbol": k, "weight": v["weight"]} for k, v in final_allocations.items()
], save_path=os.path.join(out_dir, "final_allocation.png"))

# Save CSVs
allocations = []
for plan in plan_history:
    for symbol, alloc in plan['selected'].items():
        alloc_row = alloc.copy()
        alloc_row['symbol'] = symbol
        alloc_row['date'] = plan['date']
        allocations.append(alloc_row)
metrics = {
    'final_value': final_value,
    'total_return': tr,
    'sharpe': sr,
    'max_drawdown': mdd,
    'hit_rate': hr,
    'spy_final_value': spy_final_value,
    'spy_return': spy_return
}
save_csvs(portfolio_value_history, allocations, metrics, out_dir=out_dir)
print(f"\nDemo results saved in {out_dir}/") 