import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_portfolio_vs_spy(portfolio_values, spy_values, save_path=None):
    """
    Line chart comparing portfolio and SPY value over time.
    portfolio_values, spy_values: list of dicts with 'date' and 'value'
    """
    pf_df = pd.DataFrame(portfolio_values)
    spy_df = pd.DataFrame(spy_values)
    plt.figure(figsize=(10, 6))
    plt.plot(pf_df['date'], pf_df['value'], label='Portfolio')
    plt.plot(spy_df['date'], spy_df['value'], label='SPY', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Portfolio Value vs SPY')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_drawdown_curve(portfolio_values, save_path=None):
    """
    Plot drawdown curve over time.
    """
    pf_df = pd.DataFrame(portfolio_values)
    values = pf_df['value'].values
    running_max = pd.Series(values).cummax()
    drawdown = (values - running_max) / running_max
    plt.figure(figsize=(10, 4))
    plt.plot(pf_df['date'], drawdown, color='red')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.title('Portfolio Drawdown Curve')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_final_allocation(final_allocations, save_path=None):
    """
    Pie chart of last day's weights per asset.
    final_allocations: list of dicts with 'symbol' and 'weight'
    """
    labels = [a['symbol'] for a in final_allocations]
    sizes = [a['weight'] for a in final_allocations]
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Final Portfolio Allocation')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show() 