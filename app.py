import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
from data.ticker_selector import select_top_tickers
from data.market_data import load_market_data
from pipeline import TradingPipeline
from agents.strategy.momentum_agent import MomentumAgent
from agents.strategy.mean_reversion_agent import MeanReversionAgent
from agents.strategy.event_driven_agent import EventDrivenAgent
from agents.meta_planner import MetaPlannerAgent
from simulation.simulate import simulate_portfolio, simulate_portfolio_realtime
from evaluation.metrics import total_return, sharpe_ratio, max_drawdown, hit_rate
from memory.memory_agent import MemoryAgent
import os
from data.feature_engineering import compute_features
import time
import numpy as np

class SimpleMetaPlannerAgent:
    """
    Selects the best plan based on highest confidence.
    """
    def select_best_plan(self, agent_plans):
        if not agent_plans:
            return None
        # For now, just select the first plan
        # TODO: Implement proper plan selection logic
        return agent_plans[0]

# Page config
st.set_page_config(
    page_title="Trading Strategy Simulator",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .positive {
        color: #00ac69;
    }
    .negative {
        color: #ff4b4b;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📈 Trading Strategy Simulator")

# Sidebar for inputs
st.sidebar.header("Simulation Parameters")

# Agent selection
agent_type = st.sidebar.selectbox(
    "Select Strategy",
    ["Momentum", "Mean Reversion", "Event Driven", "Ensemble"],
    help="Choose the trading strategy to simulate"
)

# Date range
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        datetime(2023, 1, 1),  # Default to 2023-01-01
        help="Start date for simulation"
    )
with col2:
    end_date = st.date_input(
        "End Date",
        datetime(2023, 3, 31),  # Default to 2023-03-31
        help="End date for simulation"
    )

# Capital and tickers
initial_capital = st.sidebar.number_input(
    "Initial Capital ($)",
    min_value=1000,
    max_value=1000000,
    value=100000,
    step=10000,
    help="Starting capital for simulation"
)

# Ticker selection (allow multiple)
tickers = st.sidebar.multiselect(
    "Select Tickers",
    ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "META", "NVDA", "SPY"],
    default=["AAPL", "MSFT", "GOOG"]
)
if not tickers:
    st.sidebar.warning("Please select at least one ticker.")
    st.stop()

# Run simulation button
if st.sidebar.button("Run Simulation", type="primary"):
    with st.spinner("Running simulation..."):
        try:
            # Load all data for selected tickers
            all_data = {}
            for ticker in tickers:
                cache_file = f"data/cache/{ticker}_{str(start_date)}_{str(end_date)}_1d_vwap.csv"
                if os.path.exists(cache_file):
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                else:
                    df = load_market_data(ticker, str(start_date), str(end_date), lookback_days=35, interval='1d')
                if df is None or df.empty:
                    st.error(f"Failed to load market data for {ticker}.")
                    st.stop()
                all_data[ticker] = df

            # Agent selection
            agent_map = {
                "Momentum": MomentumAgent,
                "Mean Reversion": MeanReversionAgent,
                "Event Driven": EventDrivenAgent
            }
            if agent_type == "Ensemble":
                # Initialize all agents for ensemble
                agents = {
                    "Momentum": MomentumAgent(),
                    "Mean Reversion": MeanReversionAgent(),
                    "Event Driven": EventDrivenAgent()
                }
                for agent in agents.values():
                    agent.set_tickers(tickers)
                meta_planner = MetaPlannerAgent(use_llm=False, use_ml=False)
            else:
                agent_cls = agent_map[agent_type]
                agent = agent_cls()
                agent.set_tickers(tickers)

            # Initialize containers for final results at the top
            final_results = st.empty()
            final_metrics_container = st.empty()
            final_allocation_container = st.empty()
            final_history_container = st.empty()
            
            # (No global progress bar or subheader here; these are now only inside the simulation mode blocks)
            
            # Run simulation with realtime updates
            if agent_type == "Ensemble":
                st.subheader("Ensemble Mode: Agent Comparison")
                agent_names = ["Momentum", "Mean Reversion", "Event Driven"]
                agent_status = {name: st.empty() for name in agent_names}
                # UI placeholders
                progress_bar = st.progress(0)
                ensemble_value_placeholder = st.empty()
                st.markdown("---")
                # For storing per-agent histories
                agent_histories = {name: [] for name in agent_names}
                agent_metrics = {name: [] for name in agent_names}
                ensemble_history = []
                ensemble_metrics = []
                selected_agents = []
                comparison_table = st.empty()
                selected_agent_status = st.empty()
                eval_period = 2
                best_agent_name = None
                best_agent_return = float('-inf')

                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                windows = []
                i = 0
                window = 30
                step = 15
                while i + window <= len(dates):
                    windows.append((dates[i], dates[i+window-1]))
                    i += step

                cash = initial_capital
                holdings = {ticker: 0 for ticker in tickers}
                agent_cash = {name: initial_capital for name in agent_names}
                agent_holdings = {name: {ticker: 0 for ticker in tickers} for name in agent_names}
                meta_planner = MetaPlannerAgent(use_llm=False, use_ml=False)

                for idx, (win_start, win_end) in enumerate(windows):
                    window_start_str = str(win_start.date())
                    window_end_str = str(win_end.date())
                    progress = (idx + 1) / len(windows)
                    progress_bar.progress(min(progress, 1.0))

                    # Get market data and features
                    market_data_dict = {}
                    features_dict = {}
                    for ticker in tickers:
                        if ticker.lower() == 'cash':
                            continue
                        if ticker in all_data:
                            df = all_data[ticker]
                            window_data = df[(df.index >= win_start) & (df.index <= win_end)]
                            if not window_data.empty:
                                market_data_dict[ticker] = window_data
                                features = compute_features(window_data)
                                if features is not None and isinstance(features, dict) and features:
                                    features_dict[ticker] = features
                                else:
                                    features_dict[ticker] = {}

                    # Get plans and simulate for each agent
                    agent_plans = {}
                    agent_step_values = {}
                    agent_allocations = {}
                    agent_returns = {}
                    for name, agent in agents.items():
                        try:
                            plan = agent.propose_plan(
                                features_dict,
                                context={},
                                current_holdings=agent_holdings[name].copy(),
                                cash=agent_cash[name],
                                portfolio_history=agent_histories[name].copy()
                            )
                            agent_plans[name] = plan
                            # Simulate execution for this agent
                            total_value = agent_cash[name] + sum(
                                agent_holdings[name][ticker] * market_data_dict[ticker]['close'].iloc[-1]
                                for ticker in tickers if ticker.lower() != 'cash' and not market_data_dict[ticker].empty
                            )
                            new_holdings = {}
                            spent_cash = 0
                            for ticker in tickers:
                                if ticker.lower() == 'cash':
                                    continue
                                if market_data_dict[ticker].empty:
                                    new_holdings[ticker] = 0
                                    continue
                                weight = plan.get(ticker, {}).get('weight', 0)
                                alloc_value = total_value * weight
                                price = market_data_dict[ticker]['close'].iloc[-1]
                                max_affordable_shares = int((agent_cash[name] + sum(agent_holdings[name][t]*price for t in tickers if t!=ticker and t.lower()!='cash')) // price)
                                shares = int(min(alloc_value // price, max_affordable_shares))
                                new_holdings[ticker] = shares
                                spent_cash += shares * price
                            new_cash = total_value - spent_cash
                            if new_cash < 0:
                                new_cash = 0
                            agent_cash[name] = new_cash
                            agent_holdings[name] = new_holdings.copy()
                            # Calculate final values
                            final_position_values = {}
                            for ticker in tickers:
                                if ticker.lower() == 'cash':
                                    final_position_values[ticker] = new_cash
                                    continue
                                if market_data_dict[ticker].empty:
                                    final_position_values[ticker] = 0
                                    continue
                                price = market_data_dict[ticker]['close'].iloc[-1]
                                final_position_values[ticker] = new_holdings[ticker] * price
                            value = sum(final_position_values.values())
                            allocation = {k: v/value for k, v in final_position_values.items() if value > 0}
                            agent_step_values[name] = value
                            agent_allocations[name] = allocation
                            # Calculate returns
                            if agent_histories[name]:
                                prev_value = agent_histories[name][-1]['value']
                                daily_return = (value - prev_value) / prev_value if prev_value != 0 else float('nan')
                                cumulative_return = value / initial_capital - 1
                            else:
                                daily_return = None
                                cumulative_return = None
                            agent_returns[name] = {
                                'daily_return': daily_return,
                                'cumulative_return': cumulative_return
                            }
                            # Update agent history
                            agent_histories[name].append({
                                'date': str(win_end.date()),
                                'value': value,
                                'cash': new_cash,
                                'holdings': new_holdings.copy(),
                                'plan': plan
                            })
                            agent_metrics[name].append({
                                'date': str(win_end.date()),
                                'portfolio_value': value,
                                'cash': new_cash,
                                'daily_return': daily_return if daily_return is not None else '—',
                                'cumulative_return': cumulative_return if cumulative_return is not None else '—',
                                **{f"pos_{k}": v for k, v in final_position_values.items()},
                                **{f"alloc_{k}": v for k, v in allocation.items()}
                            })
                        except Exception as e:
                            agent_status[name].warning(f"{name} failed: {e}")
                            continue

                    # Meta-planner selection logic with evaluation period
                    selected_agent_name = None
                    if idx < eval_period:
                        # During evaluation period, do not select a single agent
                        selected_agent_status.info(f"Meta-planner evaluation period: running all agents (window {idx+1}/{eval_period})")
                        plan = None  # No ensemble plan yet
                    else:
                        # After evaluation period, select best-performing agent so far
                        if best_agent_name is None:
                            # Evaluate best agent based on cumulative return over eval period
                            best_agent_name = agent_names[0]
                            best_agent_return = float('-inf')
                            for name in agent_names:
                                # Use last cumulative return in eval period
                                eval_returns = [m['cumulative_return'] for m in agent_metrics[name][:eval_period] if isinstance(m['cumulative_return'], (float, int))]
                                if eval_returns:
                                    avg_return = sum(eval_returns) / len(eval_returns)
                                    if avg_return > best_agent_return:
                                        best_agent_return = avg_return
                                        best_agent_name = name
                        selected_agent_status.info(f"Meta-planner selected: {best_agent_name}")
                        plan = agent_histories[best_agent_name][-1]['plan']
                        selected_agents.append(best_agent_name)

                    # Execute plan for ensemble (after eval period)
                    if idx >= eval_period and plan is not None:
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
                            prev_shares = holdings.get(ticker, 0)
                            new_holdings[ticker] = shares
                            spent_cash += shares * price
                            if shares != prev_shares:
                                trades[ticker] = {
                                    'old_shares': prev_shares,
                                    'new_shares': shares,
                                    'price': price,
                                    'value': abs(shares - prev_shares) * price
                                }
                        cash = total_value - spent_cash
                        if cash < 0:
                            cash = 0
                        holdings = new_holdings.copy()
                        # Calculate final values for ensemble
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
                        # Calculate returns for ensemble
                        if ensemble_history:
                            prev_value = ensemble_history[-1]['value']
                            daily_return = (value - prev_value) / prev_value if prev_value != 0 else float('nan')
                            cumulative_return = value / initial_capital - 1
                        else:
                            daily_return = None
                            cumulative_return = None
                        ensemble_metrics.append({
                            'date': str(win_end.date()),
                            'portfolio_value': value,
                            'cash': cash,
                            'daily_return': daily_return if daily_return is not None else '—',
                            'cumulative_return': cumulative_return if cumulative_return is not None else '—',
                            **{f"pos_{k}": v for k, v in final_position_values.items()},
                            **{f"alloc_{k}": v for k, v in allocation.items()},
                            'selected_agent': best_agent_name
                        })
                        ensemble_history.append({
                            'date': str(win_end.date()),
                            'value': value,
                            'cash': cash,
                            'holdings': holdings.copy(),
                            'plan': plan
                        })
                        # Update live ensemble value
                        ensemble_value_placeholder.metric("Current Ensemble Portfolio Value", f"${value:,.2f}")

                    # Show live comparison table
                    comp_data = {
                        'Agent': [],
                        'Portfolio Value': [],
                        'Daily Return': [],
                        'Cumulative Return': [],
                        'Selected': []
                    }
                    for name in agent_names:
                        comp_data['Agent'].append(name)
                        comp_data['Portfolio Value'].append(agent_step_values.get(name, 0))
                        dr = agent_returns.get(name, {}).get('daily_return', '—')
                        cr = agent_returns.get(name, {}).get('cumulative_return', '—')
                        comp_data['Daily Return'].append(dr if dr is not None else '—')
                        comp_data['Cumulative Return'].append(cr if cr is not None else '—')
                        comp_data['Selected'].append('✅' if name == best_agent_name and idx >= eval_period else '')
                    comp_data['Agent'].append('Ensemble')
                    if idx >= eval_period and ensemble_metrics:
                        comp_data['Portfolio Value'].append(ensemble_metrics[-1]['portfolio_value'])
                        comp_data['Daily Return'].append(ensemble_metrics[-1]['daily_return'])
                        comp_data['Cumulative Return'].append(ensemble_metrics[-1]['cumulative_return'])
                        comp_data['Selected'].append('')
                    else:
                        comp_data['Portfolio Value'].append('—')
                        comp_data['Daily Return'].append('—')
                        comp_data['Cumulative Return'].append('—')
                        comp_data['Selected'].append('')
                    comp_df = pd.DataFrame(comp_data)
                    comparison_table.dataframe(comp_df, use_container_width=True)

                # --- FINAL RESULTS SECTION (move to top) ---
                st.markdown("---")
                st.subheader("Final Results: Ensemble Portfolio")
                if ensemble_history:
                    final_value = ensemble_history[-1]['value']
                    st.metric("Final Portfolio Value (Ensemble)", f"${final_value:,.2f}")
                # Final metrics for each agent and ensemble
                def calc_final_metrics(metrics):
                    if not metrics:
                        return {'total_return': 0, 'max_drawdown': 0, 'sharpe': 0}
                    values = [m['portfolio_value'] for m in metrics if isinstance(m['portfolio_value'], (float, int))]
                    daily_returns = [m['daily_return'] for m in metrics if isinstance(m.get('daily_return'), (float, int))]
                    total_return = (values[-1] / initial_capital - 1) * 100 if values else 0
                    max_drawdown = min((v - max(values[:i+1])) / max(values[:i+1]) * 100 for i, v in enumerate(values)) if len(values) > 1 else 0
                    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 and np.std(daily_returns) > 0 else 0
                    return {'total_return': total_return, 'max_drawdown': max_drawdown, 'sharpe': sharpe}
                final_metrics = {}
                for name in agent_names:
                    final_metrics[name] = calc_final_metrics(agent_metrics[name])
                final_metrics['Ensemble'] = calc_final_metrics(ensemble_metrics)
                metrics_comp = pd.DataFrame([
                    {'Agent': name,
                     'Total Return (%)': final_metrics[name]['total_return'],
                     'Max Drawdown (%)': final_metrics[name]['max_drawdown'],
                     'Sharpe Ratio': final_metrics[name]['sharpe']}
                    for name in agent_names + ['Ensemble']
                ])
                st.dataframe(metrics_comp, use_container_width=True)
                # Final allocation pie charts
                st.subheader("Final Portfolio Allocations")
                alloc_cols = st.columns(len(agent_names) + 1)
                for idx, name in enumerate(agent_names):
                    if agent_metrics[name]:
                        last_alloc = {k.replace('alloc_', ''): v for k, v in agent_metrics[name][-1].items() if k.startswith('alloc_') and v > 0}
                        if last_alloc:
                            alloc_df = pd.DataFrame([{"Ticker": k, "Allocation": v} for k, v in last_alloc.items()])
                            alloc_cols[idx].plotly_chart(px.pie(alloc_df, names="Ticker", values="Allocation", title=f"{name} Allocation"))
                # Ensemble allocation (fix: use ensemble_history for final allocation)
                if ensemble_history and 'holdings' in ensemble_history[-1]:
                    final_allocs = {}
                    last_holdings = ensemble_history[-1]['holdings']
                    last_value = ensemble_history[-1]['value']
                    for k, v in last_holdings.items():
                        if last_value > 0:
                            final_allocs[k] = v * 1.0 / last_value
                    alloc_df = pd.DataFrame([
                        {"Ticker": k, "Allocation": v}
                        for k, v in final_allocs.items() if isinstance(v, (int, float)) and v > 0
                    ])
                    if not alloc_df.empty:
                        alloc_cols[-1].plotly_chart(px.pie(alloc_df, names="Ticker", values="Allocation", title="Ensemble Allocation"))
                    else:
                        alloc_cols[-1].write("No allocation data available for the final window.")
                else:
                    alloc_cols[-1].write("No final allocation data available.")
                st.markdown("---")
                # --- AGENT COMPARISON SECTION ---
                st.subheader("Agent Comparison Table (All Windows)")
                # Show full agent comparison table for all windows
                all_comp_data = []
                for idx in range(len(windows)):
                    for name in agent_names:
                        m = agent_metrics[name][idx] if idx < len(agent_metrics[name]) else {}
                        all_comp_data.append({
                            'Window': idx+1,
                            'Agent': name,
                            'Portfolio Value': m.get('portfolio_value', '—'),
                            'Daily Return': m.get('daily_return', '—'),
                            'Cumulative Return': m.get('cumulative_return', '—'),
                        })
                    if idx >= eval_period and ensemble_metrics and idx-eval_period < len(ensemble_metrics):
                        em = ensemble_metrics[idx-eval_period]
                        all_comp_data.append({
                            'Window': idx+1,
                            'Agent': 'Ensemble',
                            'Portfolio Value': em.get('portfolio_value', '—'),
                            'Daily Return': em.get('daily_return', '—'),
                            'Cumulative Return': em.get('cumulative_return', '—'),
                        })
                all_comp_df = pd.DataFrame(all_comp_data)
                # Convert all columns to string to avoid ArrowTypeError
                for col in ['Portfolio Value', 'Daily Return', 'Cumulative Return']:
                    all_comp_df[col] = all_comp_df[col].apply(lambda x: f"{x}" if x is not None else "—")
                st.dataframe(all_comp_df, use_container_width=True)

            else:
                # Original single-agent simulation
                portfolio_value_placeholder = st.empty()
                investment_breakdown_placeholder = st.empty()
                st.markdown("---")
                progress_bar = st.progress(0)
                status_text = st.empty()
                metrics_placeholder = st.empty()
                portfolio_chart = st.empty()
                step_details = st.empty()
                all_metrics = []
                all_portfolio_values = []
                for step_summary, portfolio_history, plan_history, step_summaries in simulate_portfolio_realtime(
                    tickers=tickers,
                    all_data=all_data,
                    agent=agent,
                    start_date=str(start_date),
                    end_date=str(end_date),
                    window=30,
                    step=15,
                    initial_cash=initial_capital
                ):
                    # Update live portfolio value and investment breakdown at the top
                    value = step_summary['portfolio']['total_value']
                    cash = step_summary['portfolio']['cash']
                    holdings = step_summary['portfolio']['holdings']
                    position_values = step_summary['portfolio']['position_values']
                    allocation = step_summary['portfolio']['allocation']
                    portfolio_value_placeholder.metric("Current Portfolio Value", f"${value:,.2f}")
                    breakdown_data = []
                    for ticker in tickers:
                        shares = holdings.get(ticker, 0)
                        val = position_values.get(ticker, 0)
                        alloc = allocation.get(ticker, 0)
                        breakdown_data.append({
                            "Ticker": ticker,
                            "Shares": shares,
                            "Value": f"${val:,.2f}",
                            "Allocation %": f"{alloc*100:.2f}%" if alloc > 0 else "0.00%"
                        })
                    breakdown_df = pd.DataFrame(breakdown_data)
                    investment_breakdown_placeholder.table(breakdown_df)
                    
                    # Update progress
                    progress = len(step_summaries) / ((end_date - start_date).days // 15)
                    progress_bar.progress(min(progress, 1.0))
                    status_text.text(f"Processing window {step_summary['window']['start']} to {step_summary['window']['end']}")
                    
                    # Accumulate data
                    all_metrics.append({
                        'date': step_summary['date'],
                        'portfolio_value': step_summary['portfolio']['total_value'],
                        'cash': step_summary['portfolio']['cash'],
                        'daily_return': step_summary.get('metrics', {}).get('daily_return', None),
                        'cumulative_return': step_summary.get('metrics', {}).get('cumulative_return', None),
                        **{f"pos_{k}": v for k, v in step_summary['portfolio']['position_values'].items()},
                        **{f"alloc_{k}": v for k, v in step_summary['portfolio']['allocation'].items()}
                    })
                    all_portfolio_values.append({
                        'date': step_summary['date'],
                        'value': step_summary['portfolio']['total_value']
                    })
                    
                    # Update metrics display
                    if step_summary.get('metrics'):
                        metrics_df = pd.DataFrame(all_metrics)
                        metrics_placeholder.dataframe(metrics_df.tail(5))
                    
                    # Update portfolio value chart
                    if all_portfolio_values:
                        pf_df = pd.DataFrame(all_portfolio_values)
                        fig = px.line(pf_df, x='date', y='value', title="Portfolio Value (Live)")
                        portfolio_chart.plotly_chart(fig, use_container_width=True)
                    
                    # Show current step details
                    step_details.json(step_summary)
                    
                    # Small delay to make updates visible
                    time.sleep(0.1)
            
            # After simulation completes, show final results at the top
            with final_results:
                st.success("Simulation completed!")
                st.subheader("Final Results")
                # Show final portfolio value for single agent
                if all_portfolio_values:
                    final_value = all_portfolio_values[-1]['value']
                    st.metric("Final Portfolio Value", f"${final_value:,.2f}")
            
            # Show final metrics
            with final_metrics_container:
                st.subheader("Final Performance Metrics")
                try:
                    final_metrics = {
                        'total_return': (all_portfolio_values[-1]['value'] / initial_capital - 1) * 100 if all_portfolio_values else 0,
                        'max_drawdown': min((v['value'] - max(p['value'] for p in all_portfolio_values[:i+1])) / max(p['value'] for p in all_portfolio_values[:i+1]) * 100 
                                          for i, v in enumerate(all_portfolio_values)) if len(all_portfolio_values) > 1 else 0,
                        'sharpe': np.mean([m['daily_return'] for m in all_metrics if m.get('daily_return') is not None]) / 
                                 np.std([m['daily_return'] for m in all_metrics if m.get('daily_return') is not None]) * np.sqrt(252) 
                                 if len(all_metrics) > 1 and any(m.get('daily_return') is not None for m in all_metrics) else 0
                    }
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Return", f"{final_metrics['total_return']:.2f}%")
                    col2.metric("Max Drawdown", f"{final_metrics['max_drawdown']:.2f}%")
                    col3.metric("Sharpe Ratio", f"{final_metrics['sharpe']:.2f}")
                except Exception as e:
                    st.error(f"Error calculating final metrics: {e}")
            
            # Show final allocation
            with final_allocation_container:
                st.subheader("Final Portfolio Allocation")
                try:
                    if step_summary and 'portfolio' in step_summary:
                        final_allocs = step_summary['portfolio']['allocation']
                        alloc_df = pd.DataFrame([
                            {"Ticker": k, "Allocation": v}
                            for k, v in final_allocs.items() if isinstance(v, (int, float)) and v > 0
                        ])
                        if not alloc_df.empty:
                            st.plotly_chart(px.pie(alloc_df, names="Ticker", values="Allocation", title="Final Portfolio Allocation"))
                        else:
                            st.write("No allocation data available for the final window.")
                    else:
                        st.write("No final allocation data available.")
                except Exception as e:
                    st.error(f"Error displaying final allocation: {e}")
            
            # Show complete metrics history
            with final_history_container:
                st.subheader("Complete Simulation History")
                try:
                    if all_metrics:
                        metrics_df = pd.DataFrame(all_metrics)
                        st.dataframe(metrics_df)
                    else:
                        st.write("No metrics history available.")
                except Exception as e:
                    st.error(f"Error displaying metrics history: {e}")

            # Add a separator between final results and live progress
            st.markdown("---")

        except Exception as e:
            st.error(f"Simulation failed: {e}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Trading Strategy Simulator") 

