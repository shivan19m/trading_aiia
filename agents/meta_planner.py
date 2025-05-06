# from .base import BaseAgent
# import random

# class MetaPlannerAgent(BaseAgent):
#     """
#     Assigns agent roles, coordinates planning, and classifies constraints.
#     """
#     def assign_roles(self, market_regime, agent_history):
#         """
#         Assign roles to agents based on market regime and historical agent success.
#         TODO: Use LLM or advanced logic for adaptive assignment.
#         """
#         # Mock: Randomly assign all agents for now
#         return ['MomentumAgent', 'MeanReversionAgent', 'EventDrivenAgent']

#     def coordinate_planning(self, agent_plans, market_data):
#         """
#         Coordinate and ensemble agent plans. Selects the best plan based on mock scoring.
#         TODO: Implement real scoring/ensembling logic.
#         """
#         # Mock: Randomly select one plan as the final plan
#         if not agent_plans:
#             return None
#         return random.choice(agent_plans)

#     def classify_constraints(self, plan):
#         """
#         Classify constraints as explicit, implicit, or derived.
#         TODO: Use LLM or rules for constraint classification.
#         """
#         # Mock: Return dummy constraints
#         return {
#             'explicit': ['max_exposure', 'max_leverage'],
#             'implicit': ['market_liquidity'],
#             'derived': ['risk_adjusted_return']
#         }

#     # Implement abstract methods with pass or simple mocks
#     def propose_plan(self, market_data, context):
#         pass

#     def justify_plan(self, plan, context):
#         pass

#     def critique_plan(self, plan, context):
#         pass

#     def validate_constraints(self, plan, constraints):
#         pass

#     def execute(self, plan):
#         pass 


from .base import BaseAgent
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from openai import OpenAI
import os
import logging

logger = logging.getLogger(__name__)

class MetaPlannerAgent(BaseAgent):
    """
    Meta agent that selects the most promising trading plan using ML + LLM ensemble evaluation.
    """

    def __init__(self, use_llm=True, use_ml=True):
        super().__init__()
        self.use_llm = use_llm
        self.use_ml = use_ml
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.ml_model = LogisticRegression()
        self._ml_trained = False

    def train_model(self, plans, labels):
        """
        Train logistic regression on historical plan scores.
        Inputs:
            - plans: list of plan dicts
            - labels: list of binary or scalar scores (e.g. [1, 0, 1])
        """
        X = [self._extract_features(p) for p in plans]
        self.ml_model.fit(X, labels)
        self._ml_trained = True

    def _extract_features(self, plan):
        """
        Convert a plan dict into numerical features for ML.
        Features: weight mean, std, var, max, min
        """
        weights = [v['weight'] for v in plan.values()]
        return [
            np.mean(weights),
            np.std(weights),
            np.var(weights),
            np.max(weights),
            np.min(weights)
        ]

    def coordinate_planning(self, plans, market_data):
        """
        Coordinate between different trading plans and select the best one.
        """
        scores = []
        for plan in plans:
            score = self._evaluate_plan(plan, market_data)
            scores.append(score)
        
        self.logger.info(f"[MetaPlanner] Plan scores: {scores}")
        
        # Select plan with highest score
        best_idx = scores.index(max(scores))
        return plans[best_idx]

    def _evaluate_plan(self, plan, market_data):
        """
        Evaluate a plan based on multiple criteria.
        """
        score = 0.0
        
        # 1. Diversification score (0-0.3)
        weights = [details['weight'] for details in plan.values()]
        if weights:
            # Penalize extreme concentration
            max_weight = max(weights)
            diversification_score = 0.3 * (1 - max_weight)
            score += diversification_score
        
        # 2. Cash allocation score (0-0.2)
        cash_weight = plan.get('cash', {}).get('weight', 0)
        cash_score = 0.2 * (0.1 <= cash_weight <= 0.3)  # Reward reasonable cash allocation
        score += cash_score
        
        # 3. Market condition alignment (0-0.3)
        market_score = 0.0
        for symbol, details in plan.items():
            if symbol == 'cash':
                continue
            if symbol in market_data:
                df = market_data[symbol]
                # Check if plan aligns with recent trend
                recent_return = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1)
                weight = details['weight']
                if (recent_return > 0 and weight > 0.1) or (recent_return < 0 and weight < 0.1):
                    market_score += 0.1
        score += min(0.3, market_score)
        
        # 4. Risk management score (0-0.2)
        risk_score = 0.0
        for symbol, details in plan.items():
            if symbol == 'cash':
                continue
            if symbol in market_data:
                df = market_data[symbol]
                # Check volatility
                volatility = df['close'].pct_change().std()
                weight = details['weight']
                if volatility > 0.02 and weight < 0.2:  # High volatility, low weight
                    risk_score += 0.1
        score += min(0.2, risk_score)
        
        return score

    def classify_constraints(self, plan):
        return {
            'explicit': ['max_exposure', 'max_leverage'],
            'implicit': ['market_liquidity'],
            'derived': ['risk_adjusted_return']
        }

    def propose_plan(self, market_data, context): pass
    def justify_plan(self, plan, context): pass
    def critique_plan(self, plan, context): pass
    def validate_constraints(self, plan, constraints): pass
    def execute(self, plan): pass