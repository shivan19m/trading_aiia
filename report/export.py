import pandas as pd
import os

def save_csvs(portfolio_values, allocations, metrics, out_dir="outputs"):
    """
    Save portfolio, allocations, and performance metrics to CSV files.
    """
    os.makedirs(out_dir, exist_ok=True)
    pf_path = os.path.join(out_dir, "portfolio.csv")
    alloc_path = os.path.join(out_dir, "allocations.csv")
    metrics_path = os.path.join(out_dir, "performance_summary.csv")
    pd.DataFrame(portfolio_values).to_csv(pf_path, index=False)
    pd.DataFrame(allocations).to_csv(alloc_path, index=False)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"Saved portfolio values to {pf_path}")
    print(f"Saved allocations to {alloc_path}")
    print(f"Saved performance summary to {metrics_path}") 