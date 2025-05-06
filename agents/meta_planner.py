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

    def coordinate_planning(self, agent_plans, market_data):
        """
        Choose the best plan using ML + LLM hybrid scoring
        """
        if not agent_plans:
            return None

        plan_scores = []

        for plan in agent_plans:
            score = 0.0

            # Use ML if trained
            if self.use_ml and self._ml_trained:
                try:
                    features = np.array(self._extract_features(plan)).reshape(1, -1)
                    score += float(self.ml_model.predict_proba(features)[0][1])  # class 1 prob
                except Exception as e:
                    logger.warning(f"[ML] Scoring failed: {e}")

            # Use LLM for additional scoring
            if self.use_llm:
                try:
                    prompt = (
                        "Evaluate the quality of this portfolio allocation plan for a short-term trading strategy.\n"
                        "Score it from 0 to 10 based on diversification, exposure, and risk/reward balance.\n"
                        "Plan:\n" + str(plan)
                    )
                    response = self.call_openai_with_backoff(
                        client=self.client,
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a financial risk evaluation assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.2,
                        max_tokens=100
                    )
                    llm_score = self._extract_score_from_text(response.choices[0].message.content)
                    if llm_score is not None:
                        score += llm_score / 10.0  # Normalize to [0,1]
                except Exception as e:
                    logger.warning(f"[LLM] Scoring failed: {e}")

            plan_scores.append(score)

        # Pick the best scoring plan
        best_idx = np.argmax(plan_scores)
        logger.info(f"[MetaPlanner] Plan scores: {plan_scores}")
        return agent_plans[best_idx]

    def _extract_score_from_text(self, text):
        """
        Try to extract a numeric score (0-10) from LLM text output
        """
        import re
        match = re.search(r"(\d+(\.\d+)?)", text)
        if match:
            score = float(match.group(1))
            if 0 <= score <= 10:
                return score
        return None

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