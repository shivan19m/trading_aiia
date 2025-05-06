# pipeline.py

import os
from datetime import datetime, timedelta
from data.market_data import load_market_data
from data.feature_engineering import compute_features  # UPDATED: unified feature builder
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
    def __init__(
        self,
        tickers,
        lookback_days=35,
        interval='1d',
        # UPDATED: hyperparameters exposed for optimization
        momentum_window=5,
        meanrev_z_thresh=1.5,
        event_weight_cap=0.2,
    ):
        self.tickers = tickers
        self.lookback_days = lookback_days
        self.interval = interval

        # UPDATED: store hyperparams locally
        self.momentum_window = momentum_window
        self.meanrev_z_thresh = meanrev_z_thresh
        self.event_weight_cap = event_weight_cap

        # Instantiate agents with hyperparams
        self.momentum_agent       = MomentumAgent(window=self.momentum_window)        # UPDATED
        self.mean_reversion_agent = MeanReversionAgent(z_thresh=self.meanrev_z_thresh)     # UPDATED
        self.event_driven_agent   = EventDrivenAgent(weight_cap=self.event_weight_cap)     # UPDATED

        for ag in [self.momentum_agent, self.mean_reversion_agent, self.event_driven_agent]:
            ag.set_tickers(tickers)

        self.validator_agent      = ValidatorAgent();      self.validator_agent.set_tickers(tickers)
        self.meta_planner         = MetaPlannerAgent();    self.meta_planner.set_tickers(tickers)
        self.executor_agent       = ExecutorAgent()
        self.memory_agent         = MemoryAgent()
        self.analyzer             = PostTradeAnalyzerAgent()

    def load_data(self, start_date, end_date):
        """
        Loads OHLCV data for each ticker in the time window (with lookback).
        """
        data = {}
        for ticker in self.tickers:
            if ticker.lower() == 'cash':
                continue
            df = load_market_data(
                ticker, start_date, end_date,
                lookback_days=self.lookback_days,
                interval=self.interval,
                force_refresh=False
            )
            print(f"{ticker}: Loaded data from {df.index.min()} to {df.index.max()}, shape={df.shape}")
            data[ticker] = df
        return data

    # UPDATED: unified feature extraction instead of per-agent loops
    def extract_features(self, market_data_dict):
        """
        UPDATED: 一次性计算所有 tickers 特征，再分发给各策略。
        """
        # 1) 统一调用 compute_features，传入超参
        all_features = compute_features(
            market_data_dict,
            momentum_window=self.momentum_window,
            meanrev_z_thresh=self.meanrev_z_thresh,
            event_weight_cap=self.event_weight_cap
        )
        # 2) 确保 cash 始终存在
        all_features['cash'] = {}

        # 3) 分发给每个 agent（它们会挑自己需要的字段）
        return {
            self.momentum_agent:       all_features,
            self.mean_reversion_agent: all_features,
            self.event_driven_agent:   all_features,
        }
        
    def run_batch(self, start_date, end_date):
        """
        Execute one batch: from data loading through analysis.
        """
        # 1. Load market data & unified features
        market_data_dict = self.load_data(start_date, end_date)
        features_dict = self.extract_features(market_data_dict)  # UPDATED

        # 2. Plan proposals & justifications (LLM → rule → equal fallback)
        strategists = [
            self.momentum_agent,
            self.mean_reversion_agent,
            self.event_driven_agent
        ]
        plans = {}
        for ag in strategists:
            plan = ag.propose_plan(features_dict, context={})
            # UPDATED: rule-based fallback if plan too uniform
            if all(abs(plan[s]['weight'] - 1/len(self.tickers)) < 1e-3 for s in plan):
                plan = ag.apply_rule_fallback(features_dict)  # each agent should implement
            plans[ag.__class__.__name__] = plan
            justification = ag.justify_plan(plan, context={})
            print(f"\n=== {ag.__class__.__name__} Plan ===")
            for symbol, d in plan.items():
                print(f"  {symbol}: {d['weight']:.2%} - {d['reason']}")
            print(f"Justification: {justification}")

        # 3. Local scoring (no Socratic pairwise loops) — UPDATED
        def local_score(plan):
            score = 0.0
            for sym, alloc in plan.items():
                w = alloc['weight']
                feat = features_dict.get(sym, {})
                momentum = abs(feat.get('momentum', 0))
                zscore = abs(feat.get('zscore', 0))
                score += w * (momentum - 0.1 * zscore)
            return score

        scores = {name: local_score(p) for name, p in plans.items()}

        # 4. Constraint validation
        constraints = self.meta_planner.classify_constraints(next(iter(plans.values())))
        valid = {
            name: p
            for name, p in plans.items()
            if self.validator_agent.validate_constraints(p, constraints)[0]
        }

        if not valid:
            print("No valid plans.")
            return None, None

        # 5. Meta-planning: pick max local score, optional aggregated LLM summary — UPDATED
        selected_name = max(valid, key=lambda n: scores[n])
        selected_plan = valid[selected_name]
        print(f"\n=== Selected Plan: {selected_name} ===")
        for symbol, d in selected_plan.items():
            print(f"  {symbol}: {d['weight']:.2%} - {d['reason']}")

        # OPTIONAL: aggregated LLM evaluation
        # summary = self.meta_planner.summarize_selection(valid, scores, selected_name)
        # print("MetaPlanner summary:", summary)

        # 6. Execution & performance analysis
        result = self.executor_agent.execute(selected_plan)
        self.memory_agent.store_plan(selected_plan, result.get('status'))
        perf = self.analyzer.evaluate_performance(selected_plan, result)
        print(f"\n=== Execution Analysis ===\nStatus: {result['status']}\nScore: {perf.get('score')}")

        # UPDATED: record continuous metrics
        metrics = {
            'cumulative_return': perf.get('cumulative_return'),
            'sharpe_ratio': perf.get('sharpe_ratio'),
            'max_drawdown': perf.get('max_drawdown'),
        }
        self.memory_agent.log_performance(selected_plan, perf)
        self.analyzer.update_preferences({self.meta_planner.__class__.__name__: perf['score']})

        return selected_plan, metrics

    # _get_features_for_ticker no longer used; removed for decoupling — UPDATED