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
from simulation.simulator import simulate
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
        datetime.now() - timedelta(days=365),
        help="Start date for simulation"
    )
with col2:
    end_date = st.date_input(
        "End Date",
        datetime.now(),
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

# Remove n_tickers slider and use fixed AAPL
tickers = ['AAPL']
st.sidebar.info("Using AAPL for simulation")

# Run simulation button
if st.sidebar.button("Run Simulation", type="primary"):
    with st.spinner("Running simulation..."):
        try:
            ticker = 'AAPL'
            data = load_market_data(ticker)
            if not data:
                st.error("Failed to load market data for AAPL.")
                st.stop()

            # Convert to DataFrame
            df = pd.DataFrame({
                'timestamp': data['timestamps'],
                'open': data['open'],
                'high': data['high'],
                'low': data['low'],
                'close': data['close'],
                'volume': data['volume'],
            })
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            # Filter by user-selected date range
            df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]
            if df.empty:
                st.error("No data available for the selected date range.")
                st.stop()

            # Agent selection
            agent_map = {
                "Momentum": MomentumAgent,
                "Mean Reversion": MeanReversionAgent,
                "Event Driven": EventDrivenAgent
            }
            agent_cls = agent_map[agent_type]
            agent = agent_cls()
            agent.set_tickers([ticker])
            context = {}

            # Simulation loop
            initial_cash = initial_capital
            cash = initial_cash
            holdings = {ticker: 0}
            portfolio_value_history = []
            allocation_history = []
            justifications = []
            dates = df.index.date.unique()
            for current_date in dates:
                # Use all data up to current_date to compute features
                df_slice = df[df.index.date <= current_date]
                features = {ticker: compute_features(df_slice)}
                plan = agent.propose_plan(features, context)
                # Save allocation for export
                allocation_history.append({
                    'date': str(current_date),
                    'plan': plan
                })
                # For now, rebalance fully to the plan weights at close price
                close_price = df_slice['close'].iloc[-1]
                alloc_cash = cash + holdings[ticker] * close_price
                weight = plan.get(ticker, {}).get('weight', 0)
                shares = int((alloc_cash * weight) // close_price)
                holdings[ticker] = shares
                cash = alloc_cash - shares * close_price
                # Portfolio value
                value = cash + holdings[ticker] * close_price
                portfolio_value_history.append({'date': str(current_date), 'value': value})
                # (Optional) Justification
                if hasattr(agent, 'justify_plan'):
                    try:
                        justification = agent.justify_plan(plan, context)
                    except Exception:
                        justification = None
                    justifications.append({'date': str(current_date), 'justification': justification})

            # Calculate metrics
            portfolio_df = pd.DataFrame(portfolio_value_history)
            tr = (portfolio_df['value'].iloc[-1] - initial_cash) / initial_cash
            mdd = (portfolio_df['value'].cummax() - portfolio_df['value']).max() / portfolio_df['value'].cummax().max()
            sr = (portfolio_df['value'].pct_change().mean() / portfolio_df['value'].pct_change().std()) * (252 ** 0.5) if portfolio_df['value'].pct_change().std() != 0 else 0

            # Get SPY benchmark
            spy_data = load_market_data('SPY')
            spy_values = []
            if spy_data:
                spy_df = pd.DataFrame({
                    'date': spy_data['timestamps'],
                    'close': spy_data['close']
                })
                spy_df['date'] = pd.to_datetime(spy_df['date'])
                spy_df = spy_df[(spy_df['date'].dt.date >= start_date) & (spy_df['date'].dt.date <= end_date)]
                if not spy_df.empty:
                    start_price = spy_df['close'].iloc[0]
                    for _, row in spy_df.iterrows():
                        value = initial_cash * (row['close'] / start_price)
                        spy_values.append({'date': row['date'].date(), 'value': value})

            # Display results
            st.header("Simulation Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Final Portfolio Value", f"${portfolio_df['value'].iloc[-1]:,.2f}")
            with col2:
                st.metric("Total Return", f"{tr * 100:.2f}%")
            with col3:
                st.metric("Max Drawdown", f"{mdd * 100:.2f}%")

            # Portfolio vs SPY chart
            st.subheader("Portfolio Performance vs SPY")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=portfolio_df['date'], y=portfolio_df['value'], name='Portfolio', line=dict(color='#00ac69')))
            if spy_values:
                spy_df2 = pd.DataFrame(spy_values)
                fig.add_trace(go.Scatter(x=spy_df2['date'], y=spy_df2['value'], name='SPY', line=dict(color='#ff4b4b')))
            fig.update_layout(xaxis_title="Date", yaxis_title="Portfolio Value ($)", hovermode='x unified', showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

            # Drawdown chart
            st.subheader("Drawdown Curve")
            drawdown = (portfolio_df['value'] - portfolio_df['value'].cummax()) / portfolio_df['value'].cummax()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=portfolio_df['date'], y=drawdown * 100, fill='tozeroy', name='Drawdown', line=dict(color='#ff4b4b')))
            fig.update_layout(xaxis_title="Date", yaxis_title="Drawdown (%)", hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

            # Final allocation pie chart (last day)
            st.subheader("Final Portfolio Allocation")
            last_plan = allocation_history[-1]['plan']
            alloc_df = pd.DataFrame([
                {'symbol': k, 'weight': v['weight']} for k, v in last_plan.items()
            ])
            fig = px.pie(alloc_df, values='weight', names='symbol', title='Portfolio Allocation')
            st.plotly_chart(fig, use_container_width=True)

            # Export options
            st.subheader("Export Results")
            if st.button("Export to CSV"):
                os.makedirs("outputs", exist_ok=True)
                portfolio_df.to_csv("outputs/portfolio_history.csv", index=False)
                pd.DataFrame(allocation_history).to_csv("outputs/allocations.csv", index=False)
                metrics = {
                    'final_value': portfolio_df['value'].iloc[-1],
                    'total_return': tr,
                    'sharpe': sr,
                    'max_drawdown': mdd
                }
                pd.DataFrame([metrics]).to_csv("outputs/metrics.csv", index=False)
                st.success("Results exported to CSV files in the 'outputs' directory")

            # (Optional) Show daily justifications/logs
            if st.checkbox("Show daily agent justifications/logs"):
                st.write(justifications)

        except Exception as e:
            st.error(f"An error occurred during simulation: {str(e)}")
            st.error("Please check that all dependencies are installed and try again.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Trading Strategy Simulator") 