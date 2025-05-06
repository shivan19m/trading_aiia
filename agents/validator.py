# from agents.base import BaseAgent

# class ValidatorAgent(BaseAgent):
#     """
#     Agent that performs static (pre-trade) and dynamic (post-execution) constraint checks.
#     Rejects plans with too high quantity, low confidence, or blacklisted symbols.
#     """
#     BLACKLISTED_SYMBOLS = {'GME', 'AMC', 'BBBY'}
#     MIN_CONFIDENCE = 0.6
#     MAX_QUANTITY = 1000
#     MAX_WEIGHT = 0.5
#     MAX_TOTAL_WEIGHT = 1.0
import os
import json
import logging
from typing import Dict, Any
from openai import OpenAI
from .base import BaseAgent
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidatorAgent(BaseAgent):
    DEFAULT_MAX_WEIGHT = 1.0
    DEFAULT_MIN_CONFIDENCE = 0.5

    def __init__(self, use_llm=True, use_ml=True):
        super().__init__()
        self.max_weight = self.DEFAULT_MAX_WEIGHT
        self.min_confidence = self.DEFAULT_MIN_CONFIDENCE
        self.use_llm = use_llm
        self.use_ml = use_ml
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Example fallback ML model for anomaly detection (dummy setup)
        self.ml_model = RandomForestClassifier()
        self._is_model_trained = False  # Will flip to True once trained

    def validate_constraints(self, plan: Dict[str, Dict[str, Any]], constraints: Dict[str, Any]):
        violations = []
        total_weight = 0.0

        for symbol, info in plan.items():
            weight = info.get('weight', 0.0)
            reason = info.get('reason', '').strip()

            if not (0.0 <= weight <= 1.0):
                violations.append(f"{symbol}: weight {weight:.2f} out of bounds [0, 1]")
            if symbol.lower() == 'cash' and weight > 0.5:
                violations.append(f"{symbol}: weight {weight:.2f} exceeds cash limit of 0.5")
            if reason == "":
                violations.append(f"{symbol}: missing allocation reason")

            total_weight += weight

        if total_weight > self.max_weight:
            violations.append(f"Total weight {total_weight:.2f} exceeds max allowed {self.max_weight:.2f}")

        # Optional: simple ML-based anomaly detection
        if self.use_ml and self._is_model_trained:
            X = np.array([[info.get('weight', 0.0)] for info in plan.values()])
            prediction = self.ml_model.predict(X)
            if np.any(prediction == -1):
                violations.append("ML model flagged anomaly in weight distribution")

        # Optional: LLM-based critique for explanation
        if self.use_llm and violations:
            try:
                prompt = (
                    "A portfolio plan has the following issues:\n" +
                    "\n".join(f"- {v}" for v in violations) +
                    "\n\nExplain the financial risk implications and how to revise the plan to satisfy constraints."
                )
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a financial risk management assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=300
                )
                llm_suggestion = response.choices[0].message.content.strip()
                violations.append("LLM Suggestion: " + llm_suggestion)
            except Exception as e:
                violations.append(f"LLM Suggestion: LLM call failed: {e}")

        return len(violations) == 0, violations

    def train_ml_model(self, X_train, y_train):
        self.ml_model.fit(X_train, y_train)
        self._is_model_trained = True

    def propose_plan(self, market_data, context):
        raise NotImplementedError("ValidatorAgent does not propose plans.")

    def justify_plan(self, plan, context):
        raise NotImplementedError("ValidatorAgent does not justify plans.")

    def critique_plan(self, plan, context):
        raise NotImplementedError("ValidatorAgent does not critique plans.")

    def execute(self, plan):
        raise NotImplementedError("ValidatorAgent does not execute plans.")


    # def validate_constraints(self, plan, constraints):
    #     """
    #     Check explicit, implicit, and derived constraints.
    #     Reject if quantity > 1000, confidence < 0.6, or symbol is blacklisted.
    #     """
    #     violations = []
    #     # Quantity check
    #     if plan.get('quantity', 0) > self.MAX_QUANTITY:
    #         violations.append(f"Quantity {plan.get('quantity')} exceeds max allowed {self.MAX_QUANTITY}.")
    #     # Confidence check
    #     if plan.get('confidence', 1.0) < self.MIN_CONFIDENCE:
    #         violations.append(f"Confidence {plan.get('confidence')} is below minimum {self.MIN_CONFIDENCE}.")
    #     # Blacklist check
    #     if plan.get('symbol', '').upper() in self.BLACKLISTED_SYMBOLS:
    #         violations.append(f"Symbol {plan.get('symbol')} is blacklisted.")
    #     # TODO: Add more constraint checks as needed
    #     is_valid = len(violations) == 0
    #     return is_valid, violations

    # def validate_constraints(self, plan, constraints):
    #     """
    #     Validate constraints for a portfolio plan (multiple assets).
    #     Reject if any symbol is blacklisted, or its metadata violates constraints.
    #     """
    #     violations = []

    #     for symbol, meta in plan.items():
    #         # Quantity check (if applicable)
    #         quantity = meta.get('quantity', 0)
    #         if quantity > self.MAX_QUANTITY:
    #             violations.append(f"{symbol}: Quantity {quantity} exceeds max allowed {self.MAX_QUANTITY}.")

    #         # Confidence check (if applicable)
    #         confidence = meta.get('confidence', 1.0)
    #         if confidence < self.MIN_CONFIDENCE:
    #             violations.append(f"{symbol}: Confidence {confidence:.2f} is below minimum {self.MIN_CONFIDENCE}.")

    #         # Blacklist check
    #         if symbol.upper() in self.BLACKLISTED_SYMBOLS:
    #             violations.append(f"{symbol}: Symbol is blacklisted.")

    #     is_valid = len(violations) == 0
    #     return is_valid, violations

    # def propose_plan(self, market_data, context):
    #     pass

    # def justify_plan(self, plan, context):
    #     pass

    # def critique_plan(self, plan, context):
    #     pass

    # def execute(self, plan):
    #     pass 