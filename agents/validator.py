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
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
from .base import BaseAgent

import json

class ValidatorAgent(BaseAgent):
    DEFAULT_MAX_WEIGHT = 1.0
    DEFAULT_MIN_CONFIDENCE = 0.5

    def __init__(self):
        super().__init__()
        self.max_weight = self.DEFAULT_MAX_WEIGHT
        self.min_confidence = self.DEFAULT_MIN_CONFIDENCE
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def validate_constraints(self, plan, constraints):
        violations = []

        # 1. Check each asset weight is valid
        total_weight = 0.0
        for symbol, info in plan.items():
            weight = info.get('weight', 0)
            reason = info.get('reason', '').strip()

            if not (0.0 <= weight <= 1.0):
                violations.append(f"{symbol}: weight {weight:.2f} is out of bounds [0, 1].")

            if symbol.lower() == 'cash' and weight > 0.5:
                violations.append(f"{symbol}: weight {weight:.2f} exceeds max 0.5")

            if reason == "":
                violations.append(f"{symbol}: missing reason for allocation.")

            total_weight += weight

        # 2. Check total exposure
        if total_weight > self.max_weight:
            violations.append(f"Total portfolio weight {total_weight:.2f} exceeds limit of {self.max_weight:.2f}")

        # 3. Try using LLM for extra suggestions
        llm_suggestion = ""
        if violations:
            try:
                prompt = (
                    "A portfolio plan has the following issues:\n"
                    + "\n".join(f"- {v}" for v in violations)
                    + "\n\nExplain why these are problematic in financial terms and suggest specific adjustments to meet constraints."
                )
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a financial risk management assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=300
                )
                llm_suggestion = response.choices[0].message.content.strip()
            except Exception as e:
                llm_suggestion = f"LLM Suggestion: Could not get LLM explanation: {e}"
            violations.append(llm_suggestion)

        is_valid = len(violations) == 0
        return is_valid, violations

    def _explain_and_tune(self, plan, violations):
        """
        Use OpenAI API to explain violations and suggest adjustments.
        """
        prompt = (
            f"You are a quantitative risk manager.\n"
            f"Given this trading plan: {plan}\n"
            f"And these violations: {violations}\n"
            "Explain why they occur and recommend how to adjust weights or parameters to satisfy constraints."
        )
        try:
            resp = client.chat.completions.create(model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert risk management assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=200)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"Could not get LLM explanation: {e}"

    # No-op implementations inherited from BaseAgent for propose_plan, justify_plan, etc.
    def propose_plan(self, market_data, context):
        raise NotImplementedError("ValidatorAgent only validates plans, does not propose.")

    def justify_plan(self, plan, context):
        raise NotImplementedError("ValidatorAgent does not justify proposals.")

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