# driver.py

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pipeline import TradingPipeline
from optimizer import StrategyOptimizer

# Load environment
load_dotenv()

if __name__ == '__main__':
    # 1. 定义标的与优化器
    tickers = ['AAPL','MSFT','GOOG','AMZN','TSLA','cash']
    optimizer = StrategyOptimizer(tickers)
    
    # 2. 在第一个窗口上进行超参搜索
    print("=== Running hyperparameter optimization on validation window ===")
    best_params = optimizer.optimize(n_trials=30)
    
    # 3. 用最优超参驱动后续回测
    window = timedelta(days=30)
    step   = timedelta(days=15)
    start  = datetime(2024, 1, 1)
    end    = datetime(2024, 12, 31)
    current = start

    # 创建主pipeline并注入最佳参数
    pipeline = TradingPipeline(tickers)
    # 根据优化结果设置 agent 参数
    pipeline.momentum_agent.lookback = best_params["momentum_window"]
    pipeline.mean_reversion_agent.confidence_threshold = best_params["meanrev_threshold"]
    pipeline.event_driven_agent.max_event_weight = best_params["event_weight_cap"]

    # 4. 批次回测循环
    while current + window <= end:
        batch_start = current.strftime('%Y-%m-%d')
        batch_end   = (current + window).strftime('%Y-%m-%d')
        print(f"\n=== Backtest window: {batch_start} to {batch_end} ===")

        selected, result = pipeline.run_batch(batch_start, batch_end)
        print("Selected Plan:", selected)
        print("Execution Result:", result)

        current += step