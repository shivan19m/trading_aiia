from agents.base import BaseAgent

class ValidatorAgent(BaseAgent):
    """
    Agent that performs static (pre-trade) and dynamic (post-execution) constraint checks.
    Rejects plans with too high quantity, low confidence, or blacklisted symbols.
    """
    BLACKLISTED_SYMBOLS = {'GME', 'AMC', 'BBBY'}
    MIN_CONFIDENCE = 0.6
    MAX_QUANTITY = 1000

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
    
    def validate_constraints(self, plan, constraints):
        """
        Validate constraints for a portfolio plan (multiple assets).
        Reject if any symbol is blacklisted, or its metadata violates constraints.
        """
        violations = []

        for symbol, meta in plan.items():
            # Quantity check (if applicable)
            quantity = meta.get('quantity', 0)
            if quantity > self.MAX_QUANTITY:
                violations.append(f"{symbol}: Quantity {quantity} exceeds max allowed {self.MAX_QUANTITY}.")

            # Confidence check (if applicable)
            confidence = meta.get('confidence', 1.0)
            if confidence < self.MIN_CONFIDENCE:
                violations.append(f"{symbol}: Confidence {confidence:.2f} is below minimum {self.MIN_CONFIDENCE}.")

            # Blacklist check
            if symbol.upper() in self.BLACKLISTED_SYMBOLS:
                violations.append(f"{symbol}: Symbol is blacklisted.")

        is_valid = len(violations) == 0
        return is_valid, violations

    def propose_plan(self, market_data, context):
        pass

    def justify_plan(self, plan, context):
        pass

    def critique_plan(self, plan, context):
        pass

    def execute(self, plan):
        pass 