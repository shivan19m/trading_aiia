import os
from dotenv import load_dotenv
load_dotenv()  # Ensure .env is loaded for API key
import openai
import json
from agents.base import BaseAgent
import random
import re
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MomentumAgent(BaseAgent):
    """
    Agent that generates trade plans based on price acceleration and volume trends using GPT-4.
    """
    def __init__(self):
        super().__init__()  # Initialize base class
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.tickers = []

    def set_tickers(self, tickers: list):
        """Set the list of tickers to consider."""
        self.tickers = tickers

    def propose_plan(self, features: Dict[str, Any], context: str, memory_agent=None) -> Dict[str, Dict[str, Any]]:
        """
        Generate a portfolio allocation plan based on momentum indicators.
        """
        if not self.tickers:
            logger.error("No tickers set for MomentumAgent. Cannot generate plan.")
            raise ValueError("No tickers set for MomentumAgent.")

        try:
            # Prepare features for each ticker
            ticker_features = {}
            for ticker in self.tickers:
                if ticker in features:
                    ticker_features[ticker] = features[ticker]
                else:
                    logger.warning(f"Missing features for {ticker}")

            # Construct prompt
            prompt = f"""
            Based on the following technical indicators and market context, generate a portfolio allocation plan.
            Focus on momentum strategies - identify strong trends and allocate accordingly.
            
            Context: {context}
            
            Features:
            {json.dumps(ticker_features, indent=2)}
            
            Generate a JSON allocation plan with weights summing to 1.0.
            Include a 'reason' for each allocation.
            Example format:
            {{
                "AAPL": {{"weight": 0.3, "reason": "Strong momentum indicators"}},
                "cash": {{"weight": 0.1, "reason": "Holding cash for opportunities"}}
            }}
            """

            # Get GPT response
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000
            )
            gpt_response = response.choices[0].message.content

            # Extract JSON using regex
            plan_match = re.search(r'\{(?:[^{}]*|\{[^{}]*\})*\}\s*$', gpt_response)
            if not plan_match:
                logger.error("Failed to find JSON in GPT response")
                return {}

            plan_str = plan_match.group(0)
            logger.debug(f"Extracted plan JSON: {plan_str[:100]}...")

            # Parse JSON
            plan = json.loads(plan_str)
            
            # Validate plan structure
            if not self._validate_plan(plan):
                logger.error("Plan validation failed")
                return {}

            return plan

        except Exception as e:
            logger.error(f"Error in propose_plan: {str(e)}")
            return {}

    def _validate_plan(self, plan: Dict[str, Dict[str, Any]]) -> bool:
        """Validate the structure of the generated plan."""
        try:
            if not isinstance(plan, dict):
                return False
            
            total_weight = 0
            for symbol, alloc in plan.items():
                if not isinstance(alloc, dict):
                    return False
                if 'weight' not in alloc or 'reason' not in alloc:
                    return False
                if not isinstance(alloc['weight'], (int, float)):
                    return False
                total_weight += alloc['weight']
            
            # Allow small floating point error
            return abs(total_weight - 1.0) < 0.01
            
        except Exception as e:
            logger.error(f"Plan validation error: {str(e)}")
            return False

    def _generate_fallback_plan(self, features):
        """Generate a simple fallback plan if GPT fails."""
        plan = {}
        n_tickers = len(features)
        if n_tickers > 0:
            weight = 1.0 / n_tickers
            for symbol in features:
                plan[symbol] = {
                    'weight': weight,
                    'reason': 'Fallback: Equal weight allocation'
                }
        else:
            plan['cash'] = {
                'weight': 1.0,
                'reason': 'Fallback: 100% cash'
            }
        return self.validate_plan(plan)
    
    def justify_plan(self, plan, context):
        """
        Provide justification for the proposed plan.
        """
        if not plan:
            return "No plan to justify."
        lines = []
        for symbol, alloc in plan.items():
            reason = alloc.get('reason', 'No reason provided')
            weight = alloc.get('weight', 0)
            lines.append(f"{symbol}: weight {weight:.2f}, reason: {reason}")
        return "; ".join(lines)

    def critique_plan(self, plan, context):
        """
        Critique another agent's plan from a momentum perspective.
        """
        if not plan:
            return "No plan to critique."
        critiques = []
        for symbol, alloc in plan.items():
            weight = alloc.get('weight', 0)
            reason = alloc.get('reason', '')
            if weight > 0.5:
                critiques.append(f"{symbol}: High allocation ({weight:.2f})—momentum strategies typically diversify more unless a very strong trend is present.")
            if 'momentum' not in reason.lower():
                critiques.append(f"{symbol}: Reason does not mention momentum—consider trend strength.")
        if not critiques:
            return "Plan aligns with momentum principles."
        return "; ".join(critiques)

    def validate_constraints(self, plan, constraints):
        """
        Validate the plan against explicit, implicit, and derived constraints.
        Returns (is_valid, violations)
        """
        violations = []
        if not plan:
            violations.append("Plan is empty.")
            return False, violations
        total_weight = sum(alloc.get('weight', 0) for alloc in plan.values())
        if abs(total_weight - 1.0) > 0.01:
            violations.append(f"Total weight {total_weight:.2f} does not sum to 1.0.")
        for symbol, alloc in plan.items():
            if alloc.get('weight', 0) < 0:
                violations.append(f"{symbol}: Negative weight.")
            if alloc.get('weight', 0) > 1:
                violations.append(f"{symbol}: Weight exceeds 1.")
        is_valid = len(violations) == 0
        return is_valid, violations

    def execute(self, plan):
        """
        Simulate execution of the plan (mock logic).
        """
        if not plan:
            return {'status': 'failed', 'reason': 'No plan to execute.'}
        # Simulate success
        return {'status': 'success', 'plan': plan} 