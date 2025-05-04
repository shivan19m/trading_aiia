def validate_plan_structure(plan: dict, tickers: list) -> dict:
    """
    Validate and normalize a trading plan structure.
    Ensures each ticker and 'cash' has a dict with at least a 'weight' key.
    Fills missing tickers with weight 0.0 and dummy reason.
    
    Args:
        plan: The trading plan dict to validate
        tickers: List of valid ticker symbols
        
    Returns:
        Normalized plan dict with all required fields
    """
    # Ensure plan is a dict
    if not isinstance(plan, dict):
        plan = {}
    
    # Initialize missing tickers with zero weight
    for ticker in tickers:
        if ticker not in plan:
            plan[ticker] = {'weight': 0.0, 'reason': 'Not mentioned'}
        elif not isinstance(plan[ticker], dict):
            plan[ticker] = {'weight': 0.0, 'reason': 'Invalid format'}
        elif 'weight' not in plan[ticker]:
            plan[ticker]['weight'] = 0.0
        if 'reason' not in plan[ticker]:
            plan[ticker]['reason'] = 'No reason provided'
    
    # Ensure cash allocation exists
    if 'cash' not in plan:
        plan['cash'] = {'weight': 0.0, 'reason': 'No allocation'}
    elif not isinstance(plan['cash'], dict):
        plan['cash'] = {'weight': 0.0, 'reason': 'Invalid format'}
    elif 'weight' not in plan['cash']:
        plan['cash']['weight'] = 0.0
    if 'reason' not in plan['cash']:
        plan['cash']['reason'] = 'No reason provided'
    
    # Normalize weights to sum to 1.0
    total_weight = sum(v['weight'] for k, v in plan.items())
    if total_weight > 0:
        for k, v in plan.items():
            v['weight'] = v['weight'] / total_weight
    
    return plan 