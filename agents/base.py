from abc import ABC, abstractmethod
from utils.helpers import validate_plan_structure

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the financial planning system.
    """
    def __init__(self):
        self.tickers = None  # Will be set by the system
    
    def set_tickers(self, tickers):
        """Set the list of valid tickers for this agent."""
        self.tickers = tickers
    
    def validate_plan(self, plan):
        """
        Validate and normalize a trading plan structure.
        Ensures each ticker and 'cash' has a dict with at least a 'weight' key.
        """
        if self.tickers is None:
            raise ValueError("Tickers not set for agent. Call set_tickers() first.")
        return validate_plan_structure(plan, self.tickers)
    
    @abstractmethod
    def propose_plan(self, market_data, context):
        """
        Generate a trading plan based on market data and context.
        Returns a validated plan dict with weights for each ticker and cash.
        """
        pass

    @abstractmethod
    def justify_plan(self, plan, context):
        """
        Provide justification for the proposed plan.
        TODO: Integrate LLM-based justification.
        """
        pass

    @abstractmethod
    def critique_plan(self, plan, context):
        """
        Critique another agent's plan.
        TODO: Implement Socratic reasoning and critique logic.
        """
        pass

    @abstractmethod
    def validate_constraints(self, plan, constraints):
        """
        Validate the plan against explicit, implicit, and derived constraints.
        TODO: Implement constraint validation logic.
        """
        pass

    @abstractmethod
    def execute(self, plan):
        """
        Execute the plan (or simulate execution).
        TODO: Integrate with ExecutorAgent and real execution logic.
        """
        pass 