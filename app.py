import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
from data.ticker_selector import select_top_tickers
from data.market_data import load_market_data
from agents.strategy.momentum_agent import MomentumAgent
from agents.strategy.mean_reversion_agent import MeanReversionAgent
from agents.strategy.event_driven_agent import EventDrivenAgent
from agents.meta_planner import MetaPlannerAgent
from simulation.simulate import simulate_portfolio
from evaluation.metrics import total_return, sharpe_ratio, max_drawdown, hit_rate
from memory.memory_agent import MemoryAgent
import os
from data.feature_engineering import compute_features

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
                st.error("Ensemble mode not yet implemented in this demo.")
                st.stop()
            agent_cls = agent_map[agent_type]
            agent = agent_cls()
            agent.set_tickers(tickers)

            # Run simulation
            sim_results = simulate_portfolio(
                tickers=tickers,
                all_data=all_data,
                agent=agent,
                start_date=str(start_date),
                end_date=str(end_date),
                window=30,
                step=15,
                initial_cash=initial_capital
            )
            pf_df = sim_results['portfolio_history']
            metrics = sim_results['metrics'] if 'metrics' in sim_results else sim_results.get('final_metrics', {})
            plan_history = sim_results['plan_history']
            step_summaries = sim_results.get('step_summaries', [])

            # Show metrics
            st.subheader("Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Return", f"{metrics.get('total_return', 0)*100:.2f}%")
            col2.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0)*100:.2f}%")
            col3.metric("Sharpe Ratio", f"{metrics.get('sharpe', 0):.2f}")
            col4.metric("Rachev Ratio", f"{metrics.get('rachev', 0):.2f}")

            # Show metrics at each time step
            if step_summaries:
                st.subheader("Step-by-Step Portfolio Metrics")
                metrics_df = pd.DataFrame([
                    {
                        'date': s['date'],
                        'portfolio_value': s['portfolio']['total_value'],
                        'cash': s['portfolio']['cash'],
                        'daily_return': s.get('metrics', {}).get('daily_return', None),
                        'cumulative_return': s.get('metrics', {}).get('cumulative_return', None),
                        **{f"pos_{k}": v for k, v in s['portfolio']['position_values'].items()},
                        **{f"alloc_{k}": v for k, v in s['portfolio']['allocation'].items()},
                        'selected_agent': s.get('selected_agent', None)  # For ensemble
                    }
                    for s in step_summaries
                ])
                st.dataframe(metrics_df)

                # Interactive step slider
                step_idx = st.slider("Select Time Step", 0, len(step_summaries)-1, len(step_summaries)-1)
                step = step_summaries[step_idx]
                st.write(f"**Step {step_idx+1} / {len(step_summaries)} — Date: {step['date']}**")
                st.json(step)

                # Allocation pie chart for selected step
                allocs = step['portfolio']['allocation']
                alloc_df = pd.DataFrame([
                    {"Ticker": k, "Allocation": v}
                    for k, v in allocs.items() if isinstance(v, (int, float)) and v > 0
                ])
                if not alloc_df.empty:
                    st.plotly_chart(px.pie(alloc_df, names="Ticker", values="Allocation", title=f"Allocation at {step['date']}"))
                else:
                    st.write("No allocation data for this step.")

                # Trades at this step
                if step['trades']:
                    st.write("**Trades Executed:**")
                    st.dataframe(pd.DataFrame([
                        {"Ticker": k, **v} for k, v in step['trades'].items()
                    ]))
                else:
                    st.write("No trades executed at this step.")

                # Plot cumulative return
                if 'cumulative_return' in metrics_df.columns:
                    fig3 = px.line(metrics_df, x='date', y='cumulative_return', title="Cumulative Return Over Time")
                    st.plotly_chart(fig3, use_container_width=True)

            # Plot portfolio value
            st.subheader("Portfolio Value Over Time")
            pf_df_reset = pf_df.reset_index()
            fig = px.line(pf_df_reset, x='date', y='value', title="Portfolio Value")
            st.plotly_chart(fig, use_container_width=True)

            # Show allocations for last window
            st.subheader("Final Window Allocation")
            last_plan = pf_df_reset.iloc[-1]['plan']
            allocs = {k: v['weight'] for k, v in last_plan.items() if isinstance(v, dict) and 'weight' in v and isinstance(v['weight'], (int, float)) and v['weight'] > 0}
            alloc_df = pd.DataFrame(list(allocs.items()), columns=["Ticker", "Weight"])
            if not alloc_df.empty:
                fig2 = px.pie(alloc_df, names="Ticker", values="Weight", title="Final Portfolio Allocation")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.write("No allocation data for the final window.")

            # Show plan history table
            st.subheader("Plan History (Last 5 Windows)")
            st.dataframe(pd.DataFrame(plan_history[-5:]))

        except Exception as e:
            st.error(f"Simulation failed: {e}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Trading Strategy Simulator") 

