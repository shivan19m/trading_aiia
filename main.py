# from data.market_data import load_market_data
# from agents.strategy.momentum_agent import MomentumAgent
# from agents.strategy.mean_reversion_agent import MeanReversionAgent
# from agents.strategy.event_driven_agent import EventDrivenAgent
# from agents.validator import ValidatorAgent
# from agents.meta_planner import MetaPlannerAgent
# from execution.executor import ExecutorAgent
# from memory.memory_agent import MemoryAgent
# from evaluation.analyzer import PostTradeAnalyzerAgent
# import os 
# from dotenv import load_dotenv
# load_dotenv()

# # Set date range for market data
# start_date = "2024-01-01"
# end_date = "2024-01-31"

# # Load market data (AAPL, daily)
# market_data = load_market_data('AAPL', start_date, end_date, lookback_days=30, interval='1d')
# context = {}  # Mock context, can be extended

# # Instantiate agents
# momentum_agent = MomentumAgent()
# mean_reversion_agent = MeanReversionAgent()
# event_driven_agent = EventDrivenAgent()
# validator_agent = ValidatorAgent()
# meta_planner = MetaPlannerAgent()
# executor_agent = ExecutorAgent()
# memory_agent = MemoryAgent()
# post_trade_analyzer = PostTradeAnalyzerAgent()

# # Each agent proposes and justifies a plan
# agents = [momentum_agent, mean_reversion_agent, event_driven_agent]
# plans = []
# print("\n--- Plan Proposals and Justifications ---")
# for agent in agents:
#     plan = agent.propose_plan(market_data, context)
#     justification = agent.justify_plan(plan, context)
#     plans.append({'agent': agent.__class__.__name__, 'plan': plan, 'justification': justification, 'critiques': []})
#     print(f"\n{agent.__class__.__name__} Plan: {plan}")
#     print(f"Justification: {justification}")

# # Pairwise critique: each agent critiques every other agent's plan
# print("\n--- Socratic Critiques ---")
# for i, agent in enumerate(agents):
#     for j, other in enumerate(agents):
#         if i != j:
#             critique = agent.critique_plan(plans[j]['plan'], context)
#             plans[j]['critiques'].append({'from': agent.__class__.__name__, 'critique': critique})

# # Print critiques for each plan
# for p in plans:
#     print(f"\nCritiques for {p['agent']} Plan:")
#     for c in p['critiques']:
#         print(f"- From {c['from']}: {c['critique']}")

# # Mock constraints (could be more sophisticated)
# constraints = meta_planner.classify_constraints(plans[0]['plan'])

# # Validate each plan
# print("\n--- Constraint Validation ---")
# valid_plans = []
# for p in plans:
#     is_valid, violations = validator_agent.validate_constraints(p['plan'], constraints)
#     if is_valid:
#         print(f"{p['agent']} plan is valid.")
#         valid_plans.append(p)
#     else:
#         print(f"{p['agent']} plan is INVALID. Violations: {violations}")
#         memory_agent.record_violation(p['plan'], violations)

# # MetaPlanner selects the best plan
# print("\n--- MetaPlanner Selection ---")
# if valid_plans:
#     selected = meta_planner.coordinate_planning([p['plan'] for p in valid_plans], market_data)
#     print(f"Selected Plan: {selected}")
# else:
#     print("No valid plans to select.")
#     selected = None

# # Execute the selected plan
# print("\n--- Execution ---")
# if selected:
#     execution_result = executor_agent.execute(selected)
#     print(f"Execution Result: {execution_result}")
#     memory_agent.store_plan(selected, execution_result['status'])
# else:
#     execution_result = None

# # Post-trade analysis
# print("\n--- Post-Trade Analysis ---")
# if selected and execution_result:
#     analysis = post_trade_analyzer.evaluate_performance(selected, execution_result)
#     print(f"Performance Analysis: {analysis}")
#     memory_agent.log_performance(selected, analysis)
#     post_trade_analyzer.update_preferences({meta_planner.__class__.__name__: analysis['score']})

# # Print memory logs
# print("\n--- Memory Logs ---")
# print("Plan History:", memory_agent.get_plan_history())
# print("Performance History:", memory_agent.get_performance_history())
# print("Constraint Violations:", memory_agent.get_constraint_violations())

from data.market_data import load_market_data
from agents.strategy.momentum_agent import MomentumAgent
from agents.strategy.mean_reversion_agent import MeanReversionAgent
from agents.strategy.event_driven_agent import EventDrivenAgent
from agents.validator import ValidatorAgent
from agents.meta_planner import MetaPlannerAgent
from execution.executor import ExecutorAgent
from memory.memory_agent import MemoryAgent
from evaluation.analyzer import PostTradeAnalyzerAgent
from data.feature_engineering import compute_features
import os 
from dotenv import load_dotenv

import numpy as np

load_dotenv()

# --- Configuration --------------------------------------------------------

start_date = "2024-01-01"
end_date   = "2024-01-31"
tickers    = ['AAPL','MSFT','GOOG','AMZN','TSLA','cash']

# --- Load raw market data -------------------------------------------------

# NOTE: This assumes `load_market_data` can load all tickers at once
market_data_dict = {
    ticker: load_market_data(ticker, start_date, end_date, lookback_days=30, interval='1d')
    for ticker in tickers if ticker != 'cash'  # Skip 'cash'
}

# --- Extract momentum features for all tickers ----------------------------

def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

# def extract_momentum_features(price_data_dict):
#     """
#     Extracts basic technical indicators: momentum, RSI, MACD
#     Inputs:
#         price_data_dict: {symbol: pd.DataFrame} where df has 'close' column
#     Returns:
#         {symbol: {feature_name: value, ...}}
#     """
#     features = {}
#     for symbol, df in price_data_dict.items():
#         try:
#             if 'close' not in df.columns or df.isnull().values.any() or len(df) < 30:
#                 raise ValueError("Invalid or insufficient data")

#             df = df.copy()
#             df['momentum'] = df['close'].pct_change(periods=5)
#             df['rsi'] = compute_rsi(df['close'])
#             df['macd'] = compute_ema(df['close'], 12) - compute_ema(df['close'], 26)

#             latest = df.dropna().iloc[-1]
#             features[symbol] = {
#                 'momentum': float(latest['momentum']),
#                 'rsi': float(latest['rsi']),
#                 'macd': float(latest['macd']),
#             }
#         except Exception as e:
#             print(f"[WARN] Feature extraction failed for {symbol}: {e}")
#             features[symbol] = {}
#     return features

# Extracted features used by agents

    
# feature_dict = extract_momentum_features(market_data_dict)
# feature_dict['cash'] = {}  # Optional: add empty dict for 'cash' to prevent key errors

# --- Instantiate agents ---------------------------------------------------

momentum_agent       = MomentumAgent()
mean_reversion_agent = MeanReversionAgent()
event_driven_agent   = EventDrivenAgent()
validator_agent      = ValidatorAgent()
meta_planner         = MetaPlannerAgent()
executor_agent       = ExecutorAgent()
memory_agent         = MemoryAgent()
post_trade_analyzer  = PostTradeAnalyzerAgent()

# --- Assign tickers to agents --------------------------------------------
# Map agent to their feature extractors

agent_feature_map = {
    momentum_agent: momentum_agent.extract_features(market_data_dict),
    mean_reversion_agent: mean_reversion_agent.extract_features(market_data_dict),
    event_driven_agent: event_driven_agent.extract_features(market_data_dict)
}

# Optional: Add empty features for 'cash'
for features in agent_feature_map.values():
    features['cash'] = {}

for ag in [
    momentum_agent,
    mean_reversion_agent,
    event_driven_agent,
    validator_agent,
    meta_planner
]:
    ag.set_tickers(tickers)

# --- 1) Plan Proposals & Justifications -----------------------------------
strategists = [momentum_agent, mean_reversion_agent, event_driven_agent]
plans = []

print("\n--- Plan Proposals and Justifications ---")
for ag in strategists:
    features = agent_feature_map.get(ag, {})  # get agent's own features
    plan = ag.propose_plan(features, context={})
    just = ag.justify_plan(plan, context={})
    plans.append({
        'agent': ag.__class__.__name__,
        'plan': plan,
        'justification': just,
        'critiques': []
    })
    print(f"\n{ag.__class__.__name__} Plan: {plan}")
    print(f"Justification: {just}")

# --- 2) Socratic Critiques ------------------------------------------------

print("\n--- Socratic Critiques ---")
for i, ag in enumerate(strategists):
    for j, other in enumerate(strategists):
        if i == j:
            continue
        critique = ag.critique_plan(plans[j]['plan'], context={})
        plans[j]['critiques'].append({
            'from': ag.__class__.__name__,
            'critique': critique
        })

for p in plans:
    print(f"\nCritiques for {p['agent']} Plan:")
    for c in p['critiques']:
        print(f"- From {c['from']}: {c['critique']}")

# --- 3) Classify Constraints ----------------------------------------------

constraints = meta_planner.classify_constraints(plans[0]['plan'])
print(f"\nConstraints: {constraints}")

# --- 4) Constraint Validation ---------------------------------------------

print("\n--- Constraint Validation ---")
valid_plans = []
for p in plans:
    is_valid, violations = validator_agent.validate_constraints(p['plan'], constraints)
    print(f"{p['agent']} → valid={is_valid}, violations={violations}")
    if is_valid:
        valid_plans.append(p)

# --- 5) MetaPlanner Selection ---------------------------------------------

print("\n--- MetaPlanner Selection ---")
if not valid_plans:
    print("No valid plans to select.")
    selected = None
else:
    # Extract raw plans from the valid plan objects
    candidate_plans = [p['plan'] for p in valid_plans]
    agent_names = [p['agent'] for p in valid_plans]

    # Let MetaPlanner evaluate and select the best plan
    selected = meta_planner.coordinate_planning(candidate_plans, market_data_dict)

    # Print scores if available (optional - MetaPlanner may log internally)
    print(f"\nEvaluating {len(candidate_plans)} valid plans:")
    for name, plan in zip(agent_names, candidate_plans):
        print(f"- Candidate from {name}: {plan}")

    print(f"\nSelected Plan: {selected}")
    
# --- 6) Execute -----------------------------------------------------------

print("\n--- Execution ---")
if selected:
    result = executor_agent.execute(selected)
    print(f"Execution Result: {result}")
    memory_agent.store_plan(selected, result.get('status'))
else:
    result = None

# --- 7) Post-Trade Analysis ----------------------------------------------

print("\n--- Post-Trade Analysis ---")
if selected and result:
    analysis = post_trade_analyzer.evaluate_performance(selected, result)
    print(f"Performance Analysis: {analysis}")
    memory_agent.log_performance(selected, analysis)
    post_trade_analyzer.update_preferences({
        meta_planner.__class__.__name__: analysis['score']
    })

# --- 8) Memory Logs -------------------------------------------------------

print("\n--- Memory Logs ---")
print("Plan History:        ", memory_agent.get_plan_history())
print("Performance History: ", memory_agent.get_performance_history())
print("Constraint Violations:", memory_agent.get_constraint_violations())