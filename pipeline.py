# pipeline.py

import os
from datetime import datetime, timedelta
from data.market_data import load_market_data
from data.feature_engineering import compute_features
from agents.strategy.momentum_agent import MomentumAgent
from agents.strategy.mean_reversion_agent import MeanReversionAgent
from agents.strategy.event_driven_agent import EventDrivenAgent
from agents.validator import ValidatorAgent
from agents.meta_planner import MetaPlannerAgent
from execution.executor import ExecutorAgent
from memory.memory_agent import MemoryAgent
from evaluation.analyzer import PostTradeAnalyzerAgent

class TradingPipeline:
    """
    Encapsulates the full backtest pipeline: feature extraction, plan proposal,
    critique, validation, meta-planning, execution simulation, and analysis.
    """
    def __init__(self, tickers, lookback_days=30, interval='1d'):
        self.tickers = tickers
        self.lookback_days = lookback_days
        self.interval = interval

        # Instantiate agents and assign tickers
        self.momentum_agent       = MomentumAgent();       self.momentum_agent.set_tickers(tickers)
        self.mean_reversion_agent = MeanReversionAgent();  self.mean_reversion_agent.set_tickers(tickers)
        self.event_driven_agent   = EventDrivenAgent();    self.event_driven_agent.set_tickers(tickers)
        self.validator_agent      = ValidatorAgent();      self.validator_agent.set_tickers(tickers)
        self.meta_planner         = MetaPlannerAgent();    self.meta_planner.set_tickers(tickers)
        self.executor_agent       = ExecutorAgent()
        self.memory_agent         = MemoryAgent()
        self.analyzer             = PostTradeAnalyzerAgent()

    def load_data(self, start_date, end_date):
        """
        Loads OHLCV data for each ticker in the time window.
        """
        data = {}
        for ticker in self.tickers:
            if ticker.lower() == 'cash':
                continue
            df = load_market_data(
                ticker, start_date, end_date,
                lookback_days=self.lookback_days,
                interval=self.interval
            )
            data[ticker] = df
        return data

    def extract_features(self, market_data_dict):
        """
        Extract features for each strategy agent.
        Returns a dict mapping agent -> feature_dict.
        """
        return {
            self.momentum_agent:       self.momentum_agent.extract_features(market_data_dict),
            self.mean_reversion_agent: self.mean_reversion_agent.extract_features(market_data_dict),
            self.event_driven_agent:   self.event_driven_agent.extract_features(market_data_dict),
        }

    def run_batch(self, start_date, end_date):
        """
        Execute one batch: from data loading through analysis.
        Returns (selected_plan, execution_result).
        """
        # 1. Load data & features
        market_data_dict = self.load_data(start_date, end_date)
        features_dict = {}
        for ticker in self.tickers:
            if ticker.lower() == 'cash':
                continue
            df = market_data_dict[ticker]
            features = self._get_features_for_ticker(ticker, start_date, end_date)
            print(f"Features for {ticker} in window {start_date} to {end_date}: {features}")  # Debug print
            features_dict[ticker] = features

        # 2. Plan proposals & justifications
        strategists = [
            self.momentum_agent,
            self.mean_reversion_agent,
            self.event_driven_agent
        ]
        plans = []
        for ag in strategists:
            plan = ag.propose_plan(features_dict, context={})
            justification = ag.justify_plan(plan, context={})
            print(f"\n=== {ag.__class__.__name__} Plan ===")
            print("Allocation:")
            for symbol, details in plan.items():
                print(f"  {symbol}: {details['weight']:.2%} - {details['reason']}")
            print(f"Justification: {justification}")
            plans.append({
                'agent': ag.__class__.__name__,
                'plan': plan,
                'justification': justification,
                'critiques': []
            })

        # 3. Socratic critiques
        for i, ag in enumerate(strategists):
            for j, _ in enumerate(strategists):
                if i == j:
                    continue
                critique = ag.critique_plan(plans[j]['plan'], context={})
                print(f"\n=== {ag.__class__.__name__} Critique of {plans[j]['agent']} ===")
                print(critique)
                plans[j]['critiques'].append({
                    'from': ag.__class__.__name__,
                    'critique': critique
                })

        # 4. Constraint validation
        constraints = self.meta_planner.classify_constraints(plans[0]['plan'])
        valid_plans = []
        for p in plans:
            is_valid, violations = self.validator_agent.validate_constraints(
                p['plan'], constraints
            )
            if is_valid:
                valid_plans.append(p)
            else:
                print(f"\n=== Constraint Violations for {p['agent']} ===")
                print(violations)
                self.memory_agent.record_violation(p['plan'], violations)

        # 5. Meta-planning
        selected = None
        if valid_plans:
            candidate_plans = [p['plan'] for p in valid_plans]
            selected = self.meta_planner.coordinate_planning(
                candidate_plans, market_data_dict
            )
            print("\n=== Selected Plan ===")
            for symbol, details in selected.items():
                print(f"  {symbol}: {details['weight']:.2%} - {details['reason']}")

        # 6. Execution & analysis
        execution_result = None
        if selected:
            execution_result = self.executor_agent.execute(selected)
            self.memory_agent.store_plan(selected, execution_result.get('status'))
            analysis = self.analyzer.evaluate_performance(selected, execution_result)
            print("\n=== Execution Analysis ===")
            print(f"Status: {execution_result.get('status')}")
            print(f"Score: {analysis.get('score', 'N/A')}")
            self.memory_agent.log_performance(selected, analysis)
            self.analyzer.update_preferences({
                self.meta_planner.__class__.__name__: analysis['score']
            })

        return selected, execution_result

    def _get_features_for_ticker(self, ticker, window_start, window_end):
        if ticker == 'cash':
            # Return default features for cash position
            return {
                'typical_price': 1.0,
                'dollar_volume': 0.0,
                'prev_close': 1.0,
                'prev_high': 1.0,
                'prev_low': 1.0,
                'rsi': 50.0,
                'macd': 0.0,
                'macd_signal': 0.0,
                'zscore': 0.0,
                'bb_upper': 1.0,
                'bb_lower': 1.0,
                'bb_ma': 1.0,
                'avg_vol_ratio': 0.0,
                'vwap': 1.0,
                'trades': 0.0,
                'pre_market': 1.0,
                'after_market': 1.0,
                'dividend': 0.0,
                'split': 0.0
            }
        
        # Load market data for the ticker
        df = load_market_data(
            ticker, window_start, window_end,
            lookback_days=self.lookback_days,
            interval=self.interval
        )
        
        # Compute features using the DataFrame
        features = compute_features(df, symbol=ticker)
        return features