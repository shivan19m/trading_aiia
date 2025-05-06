# optimizer.py

import optuna
from pipeline import TradingPipeline

class StrategyOptimizer:
    """
    Online hyperparameter optimizer using Optuna.
    Each trial proposes a set of parameters for the pipeline,
    runs a single batch, and reports back the performance metric.
    """
    def __init__(self, tickers):
        self.tickers = tickers
        self.study = optuna.create_study(direction="maximize")
        self.best_params = None

    def _objective(self, trial):
        # 1. Sample hyperparameters
        momentum_window   = trial.suggest_int("momentum_window", 3, 10)
        meanrev_threshold = trial.suggest_float("meanrev_threshold", 0.3, 0.8)
        event_weight_cap  = trial.suggest_float("event_weight_cap", 0.1, 0.5)

        # 2. Instantiate a fresh pipeline with these params
        pipeline = TradingPipeline(self.tickers)
        # Inject sampled params into agents:
        pipeline.momentum_agent.lookback = momentum_window
        pipeline.mean_reversion_agent.confidence_threshold = meanrev_threshold
        pipeline.event_driven_agent.max_event_weight = event_weight_cap

        # 3. Run one batch on a fixed validation window (now full year)
        val_start = "2024-01-01"
        val_end   = "2024-12-31"
        selected, result = pipeline.run_batch(val_start, val_end)

        # 4. Extract a scalar performance metric (e.g. Sharpe or raw return)
        if result and result.get("status") == "executed":
            perf = pipeline.analyzer.evaluate_performance(selected, result)
            return perf["score"]
        else:
            return 0.0  # failed plan gets zero score

    def optimize(self, n_trials=20):
        """
        Run n_trials of Optuna to find the best hyperparameters.
        """
        self.study.optimize(self._objective, n_trials=n_trials)
        self.best_params = self.study.best_params
        print("🏆 Best hyperparameters:", self.best_params)
        return self.best_params
