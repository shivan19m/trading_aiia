from data.market_data import load_market_data
from agents.strategy.momentum_agent import MomentumAgent
from agents.strategy.mean_reversion_agent import MeanReversionAgent
from agents.strategy.event_driven_agent import EventDrivenAgent
from agents.validator import ValidatorAgent
from agents.meta_planner import MetaPlannerAgent
from execution.executor import ExecutorAgent
from memory.memory_agent import MemoryAgent
from evaluation.analyzer import PostTradeAnalyzerAgent

from dotenv import load_dotenv
load_dotenv()

# Load market data (AAPL, 7-day hourly)
market_data = load_market_data('AAPL')
context = {}  # Mock context, can be extended

# Instantiate agents
momentum_agent = MomentumAgent()
mean_reversion_agent = MeanReversionAgent()
event_driven_agent = EventDrivenAgent()
validator_agent = ValidatorAgent()
meta_planner = MetaPlannerAgent()
executor_agent = ExecutorAgent()
memory_agent = MemoryAgent()
post_trade_analyzer = PostTradeAnalyzerAgent()

# Each agent proposes and justifies a plan
agents = [momentum_agent, mean_reversion_agent, event_driven_agent]
plans = []
print("\n--- Plan Proposals and Justifications ---")
for agent in agents:
    plan = agent.propose_plan(market_data, context)
    justification = agent.justify_plan(plan, context)
    plans.append({'agent': agent.__class__.__name__, 'plan': plan, 'justification': justification, 'critiques': []})
    print(f"\n{agent.__class__.__name__} Plan: {plan}")
    print(f"Justification: {justification}")

# Pairwise critique: each agent critiques every other agent's plan
print("\n--- Socratic Critiques ---")
for i, agent in enumerate(agents):
    for j, other in enumerate(agents):
        if i != j:
            critique = agent.critique_plan(plans[j]['plan'], context)
            plans[j]['critiques'].append({'from': agent.__class__.__name__, 'critique': critique})

# Print critiques for each plan
for p in plans:
    print(f"\nCritiques for {p['agent']} Plan:")
    for c in p['critiques']:
        print(f"- From {c['from']}: {c['critique']}")

# Mock constraints (could be more sophisticated)
constraints = meta_planner.classify_constraints(plans[0]['plan'])

# Validate each plan
print("\n--- Constraint Validation ---")
valid_plans = []
for p in plans:
    is_valid, violations = validator_agent.validate_constraints(p['plan'], constraints)
    if is_valid:
        print(f"{p['agent']} plan is valid.")
        valid_plans.append(p)
    else:
        print(f"{p['agent']} plan is INVALID. Violations: {violations}")
        memory_agent.record_violation(p['plan'], violations)

# MetaPlanner selects the best plan
print("\n--- MetaPlanner Selection ---")
if valid_plans:
    selected = meta_planner.coordinate_planning([p['plan'] for p in valid_plans], market_data)
    print(f"Selected Plan: {selected}")
else:
    print("No valid plans to select.")
    selected = None

# Execute the selected plan
print("\n--- Execution ---")
if selected:
    execution_result = executor_agent.execute(selected)
    print(f"Execution Result: {execution_result}")
    memory_agent.store_plan(selected, execution_result['status'])
else:
    execution_result = None

# Post-trade analysis
print("\n--- Post-Trade Analysis ---")
if selected and execution_result:
    analysis = post_trade_analyzer.evaluate_performance(selected, execution_result)
    print(f"Performance Analysis: {analysis}")
    memory_agent.log_performance(selected, analysis)
    post_trade_analyzer.update_preferences({meta_planner.__class__.__name__: analysis['score']})

# Print memory logs
print("\n--- Memory Logs ---")
print("Plan History:", memory_agent.get_plan_history())
print("Performance History:", memory_agent.get_performance_history())
print("Constraint Violations:", memory_agent.get_constraint_violations()) 