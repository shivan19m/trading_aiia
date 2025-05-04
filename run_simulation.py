import argparse
from data.ticker_selector import select_top_tickers
from agents.strategy.momentum_agent import MomentumAgent
from agents.strategy.mean_reversion_agent import MeanReversionAgent
from agents.strategy.event_driven_agent import EventDrivenAgent
from agents.meta_planner import MetaPlannerAgent
from simulation.simulator import simulate
from evaluation.metrics import total_return, sharpe_ratio, max_drawdown, hit_rate
from memory.memory_agent import MemoryAgent
import yfinance as yf
import pandas as pd
from visualize.plots import plot_portfolio_vs_spy, plot_drawdown_curve, plot_final_allocation
from report.export import save_csvs
import os

# TODO: Add vector memory for plan retrieval, output graphs, export report

def get_spy_benchmark(start, end, initial_cash):
    df = yf.download('SPY', start=start, end=end, interval='1d', progress=False)
    if df.empty:
        return None, None
    start_price = df['Close'].iloc[0]
    end_price = df['Close'].iloc[-1]
    spy_return = (end_price - start_price) / start_price
    spy_final_value = initial_cash * (1 + spy_return)
    return spy_final_value, spy_return

class SimpleMetaPlannerAgent:
    """
    Selects the best plan based on highest confidence.
    TODO: Use MemoryAgent for historical performance.
    """
    def select_best_plan(self, agent_plans):
        best = max(agent_plans, key=lambda p: p.get('confidence', 0))
        return best

def print_summary(final_value, tr, sr, mdd, hr, spy_final_value, spy_return):
    print("\n=== Simulation Results ===")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Total Return: {tr*100:.2f}%")
    print(f"Sharpe Ratio: {sr:.2f}")
    print(f"Max Drawdown: {mdd*100:.2f}%")
    print(f"Hit Rate: {hr*100:.2f}%")
    if spy_final_value is not None:
        print(f"SPY Benchmark: Final Value: ${spy_final_value:,.2f} | Return: {spy_return*100:.2f}%")
        print(f"Excess Return vs SPY: {(tr - spy_return)*100:.2f}%")
    else:
        print("SPY Benchmark data unavailable.")

def main():
    parser = argparse.ArgumentParser(description="Run portfolio simulation with agent-based planning.")
    parser.add_argument('--start', type=str, required=True, help='Simulation start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='Simulation end date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--n_tickers', type=int, default=5, help='Number of tickers to select')
    parser.add_argument('--agent', type=str, default='momentum', choices=['momentum', 'mean', 'event', 'ensemble'], help='Agent strategy to use')
    args = parser.parse_args()

    print(f"Selecting top {args.n_tickers} tickers...")
    tickers = select_top_tickers(args.n_tickers)
    print(f"Selected tickers: {tickers}")

    # Instantiate agent(s)
    if args.agent == 'momentum':
        agents = [MomentumAgent()]
        meta_planner = None
    elif args.agent == 'mean':
        agents = [MeanReversionAgent()]
        meta_planner = None
    elif args.agent == 'event':
        agents = [EventDrivenAgent()]
        meta_planner = None
    elif args.agent == 'ensemble':
        agents = [MomentumAgent(), MeanReversionAgent(), EventDrivenAgent()]
        meta_planner = SimpleMetaPlannerAgent()
    else:
        raise ValueError("Unknown agent type")

    memory_agent = MemoryAgent()

    print("Running simulation...")
    # Wrap simulate to support ensemble/meta-planner
    def simulate_with_ensemble(*, start_date, end_date, initial_cash, tickers, agents, frequency):
        from data.feature_engineering import compute_features
        import yfinance as yf
        import pandas as pd
        from tqdm import tqdm
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
            agent_plans = []
            for agent in agents:
                plan = agent.propose_plan(features, context={})
                agent_plans.append(plan)
            # Ensemble: select best plan
            if meta_planner:
                selected_plan = meta_planner.select_best_plan(agent_plans)
            else:
                selected_plan = agent_plans[0] if agent_plans else None
            plan_history.append({'date': str(current_date), 'plans': agent_plans, 'selected': selected_plan})
            allocations = selected_plan['allocations'] if selected_plan else []
            if prev_allocations != allocations:
                for symbol in holdings:
                    if holdings[symbol] > 0 and symbol in price_data:
                        cash_balance += holdings[symbol] * price_data[symbol]['Close'].iloc[-1]
                        holdings[symbol] = 0
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
            value = cash_balance
            for symbol in holdings:
                if holdings[symbol] > 0 and symbol in price_data:
                    value += holdings[symbol] * price_data[symbol]['Close'].iloc[-1]
            portfolio_value_history.append({'date': str(current_date), 'value': value, 'cash': cash_balance, 'holdings': holdings.copy()})
            # Print daily log
            print(f"{current_date.date()} | Value: ${value:,.2f} | Allocations: {allocations}")
        return portfolio_value_history, plan_history

    if args.agent == 'ensemble':
        portfolio_value_history, plan_history = simulate_with_ensemble(
            start_date=args.start,
            end_date=args.end,
            initial_cash=args.capital,
            tickers=tickers,
            agents=agents,
            frequency="1d"
        )
    else:
        portfolio_value_history, plan_history = simulate(
            start_date=args.start,
            end_date=args.end,
            initial_cash=args.capital,
            tickers=tickers,
            agents=agents,
            frequency="1d"
        )

    # Evaluation
    final_value = portfolio_value_history[-1]['value']
    tr = total_return(portfolio_value_history, args.capital)
    sr = sharpe_ratio(portfolio_value_history, args.capital)
    mdd = max_drawdown(portfolio_value_history)
    hr = hit_rate(portfolio_value_history)

    # SPY Benchmark
    spy_final_value, spy_return = get_spy_benchmark(args.start, args.end, args.capital)
    # Prepare SPY value history for plotting
    spy_df = yf.download('SPY', start=args.start, end=args.end, interval='1d', progress=False)
    spy_values = []
    if not spy_df.empty:
        start_price = spy_df['Close'].iloc[0]
        for i, row in spy_df.iterrows():
            value = args.capital * (row['Close'] / start_price)
            spy_values.append({'date': str(i.date()), 'value': value})

    # Print summary
    print_summary(final_value, tr, sr, mdd, hr, spy_final_value, spy_return)

    # Visualization
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    plot_portfolio_vs_spy(portfolio_value_history, spy_values, save_path=os.path.join(out_dir, "portfolio_vs_spy.png"))
    plot_drawdown_curve(portfolio_value_history, save_path=os.path.join(out_dir, "drawdown_curve.png"))
    # Final allocation pie chart
    if args.agent == 'ensemble':
        final_allocations = plan_history[-1]['selected']['allocations']
    else:
        final_allocations = plan_history[-1]['plan']['allocations']
    plot_final_allocation(final_allocations, save_path=os.path.join(out_dir, "final_allocation.png"))

    # Save CSVs
    allocations = []
    for plan in plan_history:
        if args.agent == 'ensemble':
            for alloc in plan['selected']['allocations']:
                alloc_row = alloc.copy()
                alloc_row['date'] = plan['date']
                allocations.append(alloc_row)
        else:
            for alloc in plan['plan']['allocations']:
                alloc_row = alloc.copy()
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
    print(f"\nPlots and CSVs saved in {out_dir}/")

    # Log to memory
    for plan in plan_history:
        if args.agent == 'ensemble':
            memory_agent.store_plan(plan['selected'], status='simulated')
        else:
            memory_agent.store_plan(plan['plan'], status='simulated')
    memory_agent.log_performance({'final_value': final_value}, {'total_return': tr, 'sharpe': sr, 'max_drawdown': mdd, 'hit_rate': hr})

    print("\nAgent plans and performance logged to MemoryAgent.")
    # TODO: Use vector memory for plan retrieval
    # TODO: Export simulation report (CSV/PDF)

if __name__ == "__main__":
    main() 